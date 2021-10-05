"""
Input-Output module. Helps read and write data between persistent
and volatile memory
"""
import gc
import logging
import logging.handlers
import os
import shutil
import sys
from collections import OrderedDict, namedtuple
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from itertools import accumulate
from multiprocessing import cpu_count
from multiprocessing.sharedctypes import RawArray
from os.path import isdir, join, split, splitext
from pathlib import Path
from tempfile import TemporaryFile
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import cupy as cp
#import cv2 as cv
import numpy as np
import yaml
from PIL import Image, ImageMath, UnidentifiedImageError
from skimage.io import imread, imsave
from tensorflow.keras.utils import to_categorical
from tifffile import imsave as tifsave
from tifffile import imread as tifread
from tqdm.auto import tqdm
from tf_mmciad.utils.window_ops import sliding_window as win_slider

logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler(sys.stderr)
f_path = Path("/nb_projects/logfiles/io.log")
f_path.mkdir(parents=True, exist_ok=True)
logger.setLevel(logging.DEBUG)
f_handler = logging.handlers.RotatingFileHandler(
    f_path / "debug.log", maxBytes=2*1024**2, backupCount=5
)
f_handler.setLevel(logging.DEBUG)
c_handler.setLevel(logging.DEBUG)
log_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
f_handler.setFormatter(log_format)
c_handler.setFormatter(log_format)
logger.addHandler(f_handler)
logger.addHandler(c_handler)
# PIL.Image.DecompressionBombError: could be decompression bomb DOS attack.
logger.info("PIL.Image.MAX_IMAGE_PIXELS = None")
Image.MAX_IMAGE_PIXELS = None

Size = namedtuple("Size", ["x", "y"])

var_dict = {}


def float_to_palette(image: Image) -> Image:
    """Convert float input to 8-bit
    """
    return ImageMath.eval("convert(int(a * 255), 'P')", a=image)


def init_worker(arrs, arr_shapes, stats, lock):
    """Initialize the worker .

    Args:
        arrs ([type]): [description]
        arr_shapes ([type]): [description]
        stats ([type]): [description]
        lock ([type]): [description]
    """
    var_dict["arrs"] = arrs
    var_dict["shape"] = arr_shapes
    var_dict["stats"] = stats
    tqdm.set_lock(lock)


def create_samples(
    path: Union[Path, str],
    filter_dict: dict = None,
    prefix="train",
    output_dir=None,
    duplication_list=None,
    tile_size=(208, 208),
    overlap=20,
):
    """ Read all sample slides and subdivide into square tiles.
    The resulting tiles are saved as .tif files in corresponding directories.

    Parameters
    ----------
    path : str
        Identifies the root data folder containing the whole slide images, e.g.
        .../path/train-N8b.tif
        .../path/gt/train-N8b.png
    bg_color : list of ints
        3-element list containing the RGB color code for the background class
    ignore_color : list of ints
        3-element list containing the RGB color code for the ignore class
    prefix : str
        Filename prefix, e.g.
        'train', 'test', 'val'
    output_dir : str, optional
        Relative path for output tiles
    tile_size : tuple of ints, optional
        Pixel size of the tiles created
    overlap : int, optional
        Amount of overlap between neighbouring tiles given in percentage of the
        tile size
    """

    assert len(tile_size) == 2, "tile_size must give exactly two dimensions"
    assert isinstance(tile_size[0], int) and isinstance(
        tile_size[1], int
    ), "Wrong type, must be int"
    size = Size._make(tile_size)
    assert isinstance(overlap, int), "Wrong type, must be int"
    assert 0 <= overlap < 100, "value outside valid range (0-99)"
    overlap_step = Size(
        x=size.x // (100 // (100 - overlap)), y=size.y // (100 // (100 - overlap))
    )
    output_dir = prefix if output_dir is None else output_dir
    path = Path(path)
    gt_savepath = Path(path / output_dir / "gt").resolve()
    gt_savepath.mkdir(exist_ok=True)
    image_fetcher = ImageFetcher(path, path / "gt", prefix, "tif", "png")
    est_total_tiles = image_fetcher.estimate_num_tiles(overlap_step)
    with tqdm(total=est_total_tiles) as pbar:
        for (name, input_slide, target_slide) in image_fetcher.yield_pair():
            width, height = target_slide.size
            x_coords, y_coords = np.mgrid[
                0 : width : overlap_step.x, 0 : height : overlap_step.y
            ]
            points = np.c_[x_coords.ravel(), y_coords.ravel()]
            num_points = points.shape[0]
            for current_point in range(num_points):
                x, y = points[current_point, :]
                if y > height - size.y:
                    y = height - size.y
                if x > width - size.x:
                    x = width - size.x
                dx = x + size.x
                dy = y + size.y
                res_x = input_slide[x:dx, y:dy].swapaxes(0,1)
                res_y: Image.Image = target_slide.crop((x, y, dx, dy))
                # Check if res_y should be discarded according to filter_dict
                keep = True
                if isinstance(filter_dict, dict):
                    for _, (color, prob, lower, upper) in filter_dict.items():
                        if keep and check_class(
                            res_y,
                            color,
                            probability=prob,
                            lower_threshold=lower,
                            upper_threshold=upper,
                        ):
                            keep = False
                if keep:
                    imsave(
                        path / output_dir / f"{name}_{current_point:0>4d}.png",
                        res_x,
                        check_contrast=False,
                    )
                    res_y.save(
                        gt_savepath / f"{name}_{current_point:0>4d}.png",
                        format="PNG",
                        mode="P",
                        compress_level=1,
                    )
                if keep and duplication_list is not None:
                    for color in duplication_list:
                        if duplicate_tile(res_y, color):
                            imsave(
                                Path(path,
                                output_dir,
                                f"{name}_dup_{current_point:0>4d}.png"),
                                res_x,
                                check_contrast=False,
                            )
                            res_y.save(
                                Path(gt_savepath,
                                f"{name}_dup_{current_point:0>4d}.png"),
                                format="PNG",
                                mode="P",
                                compress_level=1,
                            )
                pbar.update(1)


class ImageFetcher:
    """Gathers image paths for generating pairs of input and target

    Arguments:
        input_path {str} -- Path to input image dir
        target_path {str} -- Path to target image dir
        prefix {[type]} -- Image data prefix
        input_ext {str} -- Input file extension
        target_ext {str} -- Target file extension

    Returns:
        ImageFetcher instance
    """

    def __init__(
        self, input_path: Path, target_path: Path, prefix, input_ext: str, target_ext: str
    ):
        """Create new instance of ImageFetcher object

        Arguments:
            input_path {str} -- Path to input image dir
            target_path {str} -- Path to target image dir
            prefix {[type]} -- Image data prefix
            input_ext {str} -- Input file extension
            target_ext {str} -- Target file extension
        """
        self.input_path = input_path
        self.target_path = target_path
        self.prefix = prefix
        self.input_ext = input_ext
        self.target_ext = target_ext
        self.input_dict = self.fetch_list(self.input_path, self.prefix, self.input_ext)
        self.target_dict = self.fetch_list(
            self.target_path, self.prefix, self.target_ext
        )
        if len(self.input_dict) != len(self.target_dict):
            raise ValueError(
                "Input/target length mismatch "
                + f"({len(self.input_dict)} vs {len(self.target_dict)})"
            )

    def __len__(self) -> int:
        """Return length of source directory
        """
        return len(self.input_dict)

    def fetch_list(self, path: Path, prefix: str, ext="tif") -> Dict[str, Path]:
        """fetch filenames from path. The keys are truncated file IDs

        Parameters
        ----------
        path : str
            path to whole slide images
        prefix : str
            image data set qualifier (usually one of 'train' or 'test')
        ext : str, default "tif"
            filename extension to look for

        Returns
        -------
        dict
            dictionary of image arrays
        """

        slide_files = sorted(path.glob(f"{prefix}*.{ext}"), key=lambda x: str(x).lower())
        file_dict = OrderedDict()
        for file in slide_files:
            file_id = (
                "-".join(file.stem.split("-")[1:]).replace("-corrected", "")
                if prefix
                else str(file.stem)
            )
            file_dict[file_id] = file
        return file_dict

    def yield_pair(self) -> Iterator[Tuple[str, np.ndarray, Image.Image]]:
        """Yield example-target pair

        Yields:
            tuple[str, np.array, Image]: N-D matrices of example image
                and corresponding target image
        """
        input_kwargs = {
            "kwarg": {"plugin": "tifffile"},
        }
        target_kwargs = {
            "kwarg": {"mode": "P"},
        }

        for name, input_img_path in self.input_dict.items():
            target_img_path = self.target_dict[name]
            yield (
                name,
                imread(input_img_path, **input_kwargs["kwarg"]).swapaxes(0,1),
                self.open(target_img_path),
            )

    @staticmethod
    def open(image: Path) -> Image.Image:
        """Open an image file.

        Args:
            image (Path): path to image file
        Returns:
            Image.Image: PIL Image object
        """
        try:
            img = Image.open(image)
        except UnidentifiedImageError:
            img = Image.fromarray(tifread(image))
        return img

    def estimate_num_tiles(self, overlap_step: Size) -> int:
        """Estimate the total number of tiles in the input file.

        Args:
            overlap_step (Size): amount of overlap for neighbouring tiles

        Returns:
            int: Number of tiles that will result from the listed images
        """
        list_of_images = list(self.input_dict.values())
        total_tiles = 0
        for name in list_of_images:
            img = self.open(name)
            dim_y, dim_x = img.size
            x_coords, y_coords = np.mgrid[
                0 : dim_x : overlap_step.x, 0 : dim_y : overlap_step.y
            ]
            points = np.c_[x_coords.ravel(), y_coords.ravel()]
            total_tiles += points.shape[0]
        return total_tiles


def duplicate_tile(tile, label_color):
    """Check if tile should be duplicated.

    Args:
        tile ([type]): [description]
        label_color ([type]): [description]

    Returns:
        [type]: [description]
    """
    detection_threshold = 0.05
    if isinstance(tile, Image.Image):
        total_pixels = sum(tile.histogram())
        class_pixels = tile.histogram()[label_color]
        if class_pixels >= total_pixels * detection_threshold:
            return True
        return False
    segmap = np.logical_and.reduce(tile == label_color, axis=-1)
    if segmap.sum() >= segmap.size * detection_threshold:
        return True
    return False


def check_class(
    segmap, class_color, probability=0.9, lower_threshold=0.9, upper_threshold=None
):
    """ Filter input based on how much of a given class is present in the input image.
    Returns True if image should be filtered.

    Parameters
    ----------
    :param segmap: The input image to be filtered
    :type segmap: array-like
    :param class_color: list of list of length 3 with RGB color code matching
                        the class color checking against
    :type class_color: list of list of ints
    :param probability: probability of image being filtered if above a
                        certain threshold (optional, defaults to 0.9)
    :type probability: float
    :param lower_threshold: Threshold for fraction of segmap allowed to
                            have class_color before being a candidate
                            for filtering (optional, defaults to 0.9)
    :type lower_threshold: float
    :param upper_threshold: Threshold for fraction of segmap above
                            which it will be filtered (optional,
                            defaults to None)
    :type upper_threshold: float or None
    """
    if isinstance(segmap, Image.Image):
        total_pixels = sum(segmap.histogram())
        class_pixels = segmap.histogram()[class_color]
        if (
            upper_threshold is not None
            and class_pixels >= total_pixels * upper_threshold
        ):
            return True
        if class_pixels >= total_pixels * lower_threshold:
            return np.random.rand() > 1 - probability
        return False
    template = np.zeros(segmap.shape[:-1])
    template += np.logical_and.reduce(segmap == class_color, axis=-1)
    if (
        upper_threshold is not None
        and template.sum() >= template.size * upper_threshold
    ):
        return True
    if template.sum() >= template.size * lower_threshold:
        return np.random.rand() > 1 - probability
    return False


def load_slides_as_dict(
    path,
    prefix="N8b",
    mean_arr=None,
    std_arr=None,
    input_max=255,
    gt_path=None,
    num_cls=None,
    colors=None,
    cuda=False,
):
    """Loads a dictionary of slides from a file .

    Args:
        path ([type]): [description]
        prefix (str, optional): [description]. Defaults to "N8b".
        mean_arr ([type], optional): [description]. Defaults to None.
        std_arr ([type], optional): [description]. Defaults to None.
        input_max (int, optional): [description]. Defaults to 255.
        gt_path ([type], optional): [description]. Defaults to None.
        num_cls ([type], optional): [description]. Defaults to None.
        colors ([type], optional): [description]. Defaults to None.
        cuda (bool, optional): [description]. Defaults to False.

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]

    Yields:
        [type]: [description]
    """
    if gt_path and num_cls is None:
        raise ValueError("Required input missing: num_cls")
    if gt_path and colors is None:
        raise ValueError("Required input missing: num_cls")
    ftype = "*.tif"
    if (mean_arr is None or std_arr is None) and (std_arr != mean_arr):
        raise ValueError("Both m and s or neither must be supplied")
    input_img_files = sorted(glob(join(path, prefix + ftype)))
    slide_names = [
        "-".join(splitext(split(file)[1])[0].split("-")[1:]).replace("-corrected", "")
        for file in input_img_files
    ]
    input_slides = {
        name: imread(path).astype("float") / 1.0
        for name, path in zip(slide_names, input_img_files)
    }
    memmap = False
    for img in input_slides.values():
        if img.size > 3072 ** 2:
            memmap = True
    if memmap:
        with TemporaryFile(prefix="input_", dir=path + "../tmp") as tfile:
            memmap_input_slides = {
                name: np.memmap(tfile, dtype=np.floating, mode="w+", shape=img.shape,)
                for name, img in input_slides.items()
            }
            for name, img in memmap_input_slides.items():
                memmap_input_slides[name][:] = input_slides[name][:]
                input_slides.pop(name)
        input_slides = memmap_input_slides

    if mean_arr is not None and std_arr is not None:
        if memmap and not cuda:
            for name, img in tqdm(input_slides.items(), desc="Processing input images"):
                # tmp = np.memmap(
                #     TemporaryFile(dir=path + "../tmp")[1],
                #     dtype=np.floating,
                #     mode="w+",
                #     shape=img.shape,
                # )
                tmp = RawArray("d", int(np.prod(img.shape)))
                tmp_np = np.frombuffer(tmp, dtype=np.floating).reshape(img.shape)
                img_ra = RawArray("d", int(np.prod(img.shape)))
                img_np = np.frombuffer(img_ra, dtype=np.floating).reshape(img.shape)
                np.copyto(img_np, img)
                scaffold: np.array = np.zeros(img.shape[:1])
                n_processes = cpu_count()
                splits = np.array_split(scaffold, n_processes)
                chunks = [np.array((), dtype=np.bool)] + splits[:-1]
                offsets = list(accumulate([len(i) for i in chunks]))
                chunks = list(zip(splits, offsets))
                del splits
                arrs = [tmp_np, img_np]
                stats = [mean_arr, std_arr, input_max]
                assert isinstance(stats[0], np.ndarray)
                with ProcessPoolExecutor(
                    n_processes,
                    initializer=init_worker,
                    initargs=(arrs, img.shape, stats, tqdm.get_lock()),
                ) as executor:
                    list(
                        tqdm(
                            executor.map(parallel_norm, chunks),
                            total=len(chunks),
                            desc="Processing image chunks",
                        )
                    )
                input_slides[name] = tmp_np
            # for i in memmap_input_slides.values():
            #    os.unlink(i.filename)
            del memmap_input_slides
        elif memmap and cuda:
            for name, img in tqdm(input_slides.items(), desc="Processing input images"):
                with TemporaryFile(prefix="raw_", dir=path + "../tmp") as raw_fp:
                    tmp = np.memmap(
                        raw_fp, dtype=np.floating, mode="w+", shape=img.shape,
                    )
                    for (x, y, dx, dy, window) in win_slider(
                        img, 3072, (3072, 3072)
                    ):
                        img = cp.array(window)
                        cp.cuda.Stream.null.synchronize()
                        img[x:dx, y:dy] = (window - mean_arr) / std_arr
                        cp.cuda.Stream.null.synchronize()
        else:
            for i in input_slides.keys():
                input_slides[i] = (input_slides[i] - mean_arr) / std_arr  # / input_max
    if gt_path:
        target_img_files = sorted(glob(join(path, gt_path, prefix + "*.png")))
        slide_names = [
            "-".join(splitext(split(file)[1])[0].split("-")[1:]).replace(
                "-corrected", ""
            )
            for file in target_img_files
        ]
        target_color = {
            name: imread(path).astype("uint8")[:, :, :3]
            for name, path in zip(slide_names, target_img_files)
        }
        if memmap:
            with TemporaryFile(prefix="target_color_", dir=path + "../tmp") as ttfile:
                memmap_target_color = {
                    name: np.memmap(
                        ttfile, dtype=np.floating, mode="w+", shape=img.shape,
                    )
                    for name, img in target_color.items()
                }
                for name, img in memmap_target_color.items():
                    memmap_target_color[name][:] = target_color[name][:]
                    target_color.pop(name)
            target_color = memmap_target_color
        target_sparse = {
            name: np.expand_dims(np.zeros(slide.shape[:2]), axis=-1)
            for name, slide in target_color.items()
        }
        if memmap and not cuda:
            with TemporaryFile(prefix="target_sparse_", dir=path + "../tmp") as tsfile:
                memmap_target_sparse = {
                    name: np.memmap(tsfile, dtype=np.uint8, mode="w+", shape=img.shape,)
                    for name, img in target_sparse.items()
                }
                for name, img in memmap_target_sparse.items():
                    memmap_target_sparse[name][:] = target_sparse[name][:]
                    target_sparse.pop(name)
            target_sparse = memmap_target_sparse
            with TemporaryFile(
                prefix="target_cat_", dir=path + "../tmp"
            ) as target_file:
                memmap_target = {
                    name: np.memmap(
                        target_file,
                        dtype=np.uint8,
                        mode="w+",
                        shape=img.shape[:-1] + (num_cls,),
                    )
                    for name, img in target_sparse.items()
                }
            target = memmap_target
            for name, img in tqdm(
                target_color.items(), desc="Processing target images"
            ):
                tmp = RawArray("d", int(np.prod(img.shape)))
                tmp_np = np.frombuffer(tmp, dtype=np.floating).reshape(img.shape)
                img_ra = RawArray("B", int(np.prod(target_sparse[name].shape)))
                img_np = np.frombuffer(img_ra, dtype=np.uint8).reshape(
                    target_sparse[name].shape
                )
                np.copyto(img_np, target_sparse[name])
                target_raw = RawArray("B", int(np.prod(target[name].shape)))
                target_np = np.frombuffer(target_raw, dtype=np.uint8).reshape(
                    target[name].shape
                )
                np.copyto(target_np, target[name])
                arrs = [tmp_np, img_np, target_np]
                shapes = [img.shape, target_sparse[name].shape, target[name].shape]
                stats = [num_cls, colors]
                with ProcessPoolExecutor(
                    n_processes,
                    initializer=init_worker,
                    initargs=(arrs, shapes, stats, tqdm.get_lock()),
                ) as executor:
                    list(
                        tqdm(
                            executor.map(parallel_categorical, chunks),
                            total=len(chunks),
                            desc="Processing image chunks",
                        )
                    )
                target_sparse[name] = tmp_np
                target[name] = target_np
        elif memmap and cuda:
            pass
        else:
            for cls_ in range(num_cls):
                color = colors[cls_, :]
                for name in target_sparse.keys():
                    target_sparse[name] += np.expand_dims(
                        np.logical_and.reduce(target_color[name] == color, axis=-1)
                        * cls_,
                        axis=-1,
                    )
            target = {
                name: to_categorical(slide, num_classes=num_cls)
                for name, slide in target_sparse.items()
            }
        return input_slides, target
    return input_slides


def parallel_categorical(chunk):
    """Convert a chunk of one-hot encoded array to a categorical array .

    Args:
        chunk ([type]): [description]
    """
    arr_chunk, offset = chunk
    it = np.nditer(arr_chunk, flags=["multi_index", "zerosize_ok"])
    tmp = np.frombuffer(var_dict["arrs"][0]).reshape(var_dict["shape"][0])
    img = np.frombuffer(var_dict["arrs"][1], dtype=np.uint8).reshape(
        var_dict["shape"][1]
    )
    target = np.frombuffer(var_dict["arrs"][2], dtype=np.uint8).reshape(
        var_dict["shape"][2]
    )
    num_cls = var_dict["stats"][0]
    colors = var_dict["stats"][1]
    while not it.finished:
        idx = np.array(it.multi_index)
        idx[0] += offset
        for cls_ in range(num_cls):
            color = colors[cls_, :]
            img[idx] += np.expand_dims(
                np.logical_and.reduce(tmp[idx] == color, axis=-1) * cls_, axis=-1,
            ).astype(np.uint8)
        target[idx] = to_categorical(img[idx], num_classes=num_cls)
        it.iternext()


def parallel_norm(chunk):
    """Perform normalization of a chunk of an array.

    Args:
        chunk ([type]): [description]
    """
    arr_chunk, offset = chunk
    it = np.nditer(arr_chunk, flags=["multi_index", "zerosize_ok"])
    tmp = np.frombuffer(var_dict["arrs"][0]).reshape(var_dict["shape"])
    img = np.frombuffer(var_dict["arrs"][1]).reshape(var_dict["shape"])
    mean_arr = var_dict["stats"][0]
    std_arr = var_dict["stats"][1]
    # input_max = var_dict['stats'][2]
    while not it.finished:
        idx = np.array(it.multi_index)
        idx[0] += offset
        tmp[idx] = (img[idx] - mean_arr) / std_arr  # / input_max
        it.iternext()


def move_files_in_dir(src_dir: str, dst_dir: str, pattern=None) -> None:
    """Move files in src_dir to dst_dir .

    Args:
        src_dir (str): [description]
        dst_dir (str): [description]
        pattern ([type], optional): [description]. Defaults to None.
    """
    # Check if both the are directories
    if isdir(src_dir) and isdir(dst_dir):
        # Iterate over all the files in source directory
        if pattern is None:
            files = glob(src_dir + "*")
            with tqdm(total=len(files)) as pbar:
                for file_path in files:
                    # Move each file to destination Directory
                    shutil.move(file_path, dst_dir)
                    pbar.update(1)
        if isinstance(pattern, str):
            files = glob(src_dir + pattern)
            with tqdm(total=len(files)) as pbar:
                for file_path in files:
                    # Move each file to destination Directory
                    shutil.move(file_path, dst_dir)
                    pbar.update(1)
        if isinstance(pattern, list):
            with tqdm(total=len(pattern)) as pbar:
                for group in pattern:
                    files = glob(src_dir + group)
                    with tqdm(total=len(files)) as inner_pbar:
                        for file_path in files:
                            # Move each file to destination Directory
                            shutil.move(file_path, dst_dir)
                            inner_pbar.update(1)
                    pbar.update(1)
    else:
        print("src_dir & dst_dir should be directories")

def dump(obj, method=print):
    """Dump all attributes of an object to a key-value formatted string.
    each attribute is given as the parameter to 'method' (defaults to 'print')

    Args:
        obj ([type]): [description]
        method ([type], optional): [description]. Defaults to print.
    """
    for attr in dir(obj):
        if (attr.startswith('__') and attr.endswith('__')) or attr == 'img_tiler':
            continue
        if attr in ('tile_list', '_tile_list', 'tile_coords'):
            method("len(obj.%s) = %r" % (attr, len(getattr(obj, attr))))
        else:
            method("obj.%s = %r" % (attr, getattr(obj, attr)))

class ImageTiler:
    """Create tiles from a TIFF image file and store them as a
    collection of TIFF files.
    If the specified tile path is not empty, ImageTiler will
    check if the directory contains any TIFF files that matches
    the naming pattern generated by ImageTiler and store the file
    rather than generating the tiles again.
    """

    def __init__(
        self,
        image_path: Path,
        tile_dimensions: Iterable[int],
        tile_path: Path,
        tile_overlap: int = 128,
        force_extension: Optional[str] = None,
    ):
        self.basename, self.extension = image_path.stem, image_path.suffix
        self.width, self.height = tile_dimensions
        self.overlap = tile_overlap
        self.tile_path = tile_path
        self._tile_list: List[Path] = list()
        self.tile_coords: Dict[str, Tuple[int, int, int, int]] = dict()
        self.img = Image.open(image_path)
        self.img_width, self.img_height = self.img.size
        if force_extension:
            if force_extension not in ["png", "jpg", "tif", "bmp"]:
                raise ValueError(f"Invalid extension '{force_extension}'")
            self.extension = f".{force_extension}"

        dump(self, method=logger.debug)

        self._set_tile_list()
        if self._tile_list:
            try:
                self._mimick_tile_generation()
            except IndexError:
                self._generate_tiles()    
        if not self._tile_list:
            self._generate_tiles()

    @property
    def tile_list(self):
        """Return list of tile file paths

        Returns:
            List[str]: list of tile paths
        """
        if not self._tile_list:
            self._set_tile_list()
        return self._tile_list

    def _set_tile_list(self):
        """Creates the list of tiles (empty if no tiles exist)

        Raises
        ------
        NotADirectoryError
            Raised if tile_path points to a file
        """
        if self.tile_path.is_dir():
            self._tile_list = sorted(
                self.tile_path.glob(f"{self.basename}_*{self.extension}")
            )
        elif self.tile_path.is_file():
            raise NotADirectoryError(f"The directory name is invalid: {self.tile_path}")
        else:
            self.tile_path.mkdir(exist_ok=True)

    def _mimick_tile_generation(self):
        for k, (_, bbox) in enumerate(self._crop(sim=True)):
            fp = self._tile_list[k]
            self.tile_coords[fp] = bbox

    def _generate_tiles(self):
        """Generate tiles from the input image and append the filename
        to tile_list.
        """
        for k, (tile, bbox) in enumerate(self._crop()):
            # img = Image.new("RGB", (self.height, self.width), (0, 0, 0))
            # img.paste(tile)
            if self.tile_path.stem == self.basename:
                fp = Path(self.tile_path, f"tile_{k:04d}{self.extension}")
            else:
                fp = Path(self.tile_path, f"{self.basename}_{k:04d}{self.extension}")
            self._tile_list.append(fp)
            self.tile_coords[fp] = bbox
            tile.save(fp)

    def _crop(self, sim=False) -> Iterator[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """Divide the specified image into tiles no larger than the
        specified height and width and yield them in sequence.

        Yields
        -------
        Image._ImageCrop
            A tile of the specified image
        Tuple[int]
            Corner coordinates of the yielded crop box
        """
        if sim:
            dummy = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        for y in range(0, self.img_height - self.overlap, self.height - self.overlap):
            for x in range(0, self.img_width - self.overlap, self.width - self.overlap):
                dx: int = min(x + self.width, self.img_width)
                dy: int = min(y + self.height, self.img_height)
                box = (x, y, dx, dy)
                if sim:
                    yield dummy, box
                yield self.img.crop(box), box

    def dump(self, fp: str):
        """Write tile information to file

        Args:
            fp (str): file path
        """
        with open(fp, mode="w") as file:
            data = {
                "tile_coords": self.tile_coords,
                "tile_dimensions": [self.height, self.width],
                "overlap": self.overlap,
                "img_dimensions": [self.img_width, self.img_height],
            }
            yaml.safe_dump(data, file)

    def load(self, fp: str):
        """Load tile information from file

        Args:
            fp (str): file path
        """
        with open(fp, mode="r") as file:
            data = yaml.safe_load(file)
        self.tile_coords = data["tile_coords"]
        self.width, self.height = data["tile_dimensions"]
        self.overlap = data["overlap"]
        self.img_width, self.img_height = data["img_dimensions"]
        self._tile_list = sorted([Path(s) for s in self.tile_coords.keys()])

    def remove(self):
        """Remove all tiles
        """
        for file in self._tile_list:
            file.unlink(missing_ok=True)
        self.tile_path.rmdir()

    def __getitem__(self, index):
        """x.__getitem__(y) <==> x[y]
        """
        key = self._tile_list.__getitem__(index)
        return key, self.tile_coords.__getitem__(key)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        """Implement iter(self).
        """
        return self.tile_coords.__iter__()

    def items(self):
        """D.items() -> a set-like object providing a view on D's items
        """
        return self.tile_coords.items()


class ImageAssembler:
    """Assemble a list of image tiles to a complete image
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        tile_info: Union[str, Path, ImageTiler],
        ext: str = None,
    ):
        """Initialize the instance of this TileTiler .

        Args:
            output_path (str, Path): path to recombined output image
            tile_info (str, Path, ImageTiler): ImageTiler object or path to a
                YAML dump of an ImageTiler object
            ext (str, optional): file extension for output image. If omitted, the
                extension is inferred from the extension of the tiles.
                Defaults to None.
        """
        self.img_path = Path(output_path)
        self.width, self.height = 0, 0
        self.overlap = 0
        self.img_width, self.img_height = 0, 0
        self._tile_list = None
        self.ext = ""
        if not isinstance(tile_info, (str, Path)):
            self.img_tiler: ImageTiler = tile_info
            self._tile_list = self.img_tiler._tile_list
            self.tile_coords = self.img_tiler.tile_coords
            self.width, self.height = self.img_tiler.width, self.img_tiler.height
            self.overlap = self.img_tiler.overlap
            self.img_width = self.img_tiler.img_width
            self.img_height = self.img_tiler.img_height
            self.ext = self._get_extension(self._tile_list[0])
        else:
            self.load(tile_info)
        if ext:
            self.ext = ext
        dump(self, method=logger.debug)

    def load(self, file_path: Union[str, Path]):
        """Load tile information from file
        """
        with open(file_path, mode="r") as file:
            data = yaml.safe_load(file)
        self.tile_coords = data["tile_coords"]
        self.width, self.height = data["tile_dimensions"]
        self.overlap = data["overlap"]
        self.img_width, self.img_height = data["img_dimensions"]
        self._tile_list = list([Path(s) for s in self.tile_coords.keys()])
        self.ext = self._get_extension(self._tile_list[0])

    def _get_extension(self, tile_path):
        return os.path.splitext(tile_path)[-1]

    def read_tiff(self, path: Union[str, Path]) -> cp.ndarray:
        """
        path - Path to the multipage-tiff file

        returns: n-channel image with the leading dimension being the channels
        """
        #_, img = cv.imreadmulti(path, flags=cv.IMREAD_ANYDEPTH)
        img = tifread(path)
        ndimg = cp.array(img)
        return ndimg

    def merge(self, colors: np.ndarray=None, **pil_kwargs):
        """Merge tiles into complete image

        For softmaxed input, supply a color vector.

        For PNG output, add the following keywords:
            format='PNG',
            mode='P'
        """
        img = Image.new("RGB", (self.img_width, self.img_height), (0, 0, 0))
        multi_frame = False
        logger.debug(f"{colors = }")
        if colors is not None:
            palette = colors.flatten().tolist()
            # color_map = np.ndarray(shape=(256 * 256 * 256), dtype="int32")
            # color_map[:] = -1
            # color_codes = {i: colors[i] for i in range(len(colors))}
            # for idx, rgb in color_codes.items():
            #     rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            #     color_map[rgb] = idx
        else:
            palette = None
        logger.debug(f"{palette = }")
        # def to_palette(image):
        #     image = image.dot(np.array([65536, 256, 1], dtype="int32"))
        #     return color_map[image].astype("uint8")

        for _, (fp, bbox) in enumerate(sorted(self.tile_coords.items())):
            fp = Path(fp).with_suffix(self.ext)
            x, y = bbox[:2]
            if not multi_frame:
                with Image.open(fp) as tile:
                    if tile.n_frames > 1:
                        multi_frame = True
                    else:
                        tile = Image.fromarray(tile)

            if multi_frame:
                tile = self.read_tiff(fp)
                #logger.debug(f"{tile.shape = }")
                if colors is not None:
                    if img.mode == "RGB":
                        img = img.convert(mode="P")
                        img.putpalette(palette)
                    sparse_tile = cp.argmax(tile, axis=-1)
                    #logger.debug(f"{sparse_tile.shape = }")
                    pil_kwargs["format"] = "PNG"
                    pil_kwargs["mode"] = "P"
                    pil_kwargs["compress_level"] = 1
                    uint_tile: np.ndarray = sparse_tile.get()
                    uint_tile = uint_tile.astype(np.uint8)
                    tile = Image.fromarray(uint_tile, mode="P")
                    tile.putpalette(palette)

            mask = np.full(tile.size, 255, dtype="uint8")
            if y > 0:
                mask[: self.overlap // 2, :] = 0
            if x > 0:
                mask[:, : self.overlap // 2] = 0
            mask = Image.fromarray(mask.T)
            box = bbox[:2]
            try:
                img.paste(tile, box, mask)
            except ValueError as err:
                logger.error("Error during paste operation")
                logger.error(f"Current tile path: {fp}")
                logger.error(f"{img.size = }, {tile.size = }, {box = }, {mask.size = }")
                logger.error(f"{img.mode = }, {tile.mode = }")
                logger.exception(f"{type(tile) = }, {type(box) = }, {type(mask) = }", exc_info=err)
                img.paste(tile, tuple([float(x), float(y)]), mask)

            # TODO Generalize to other formats
        if colors is not None:
            img = img.convert(mode="P")
            img.putpalette(palette)
        self.img_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing to {self.img_path} ...")
        img.save(self.img_path, **pil_kwargs)
        logger.info(f"Finished writing to {self.img_path}")

    def merge_multichannel(self):
        """Merge the image tiles and save each channelas a separate image.

        Returns:
            [type]: [description]
        """
        meta_name, _ = os.path.splitext(self._tile_list[0])
        meta_path = meta_name + self.ext
        meta_img = Image.open(meta_path)
        n_frames = meta_img.n_frames
        meta_img.close()
        path_list = []
        logger.info("Writing to disk ...")
        for frame in range(n_frames):
            img = Image.new("F", (self.img_width, self.img_height), 0.0)
            for _, (fp, (x, y, _, _)) in enumerate(sorted(self.tile_coords.items())):
                fp = f"{os.path.splitext(fp)[0]}{self.ext}"
                tile = Image.open(fp)
                tile.seek(frame)

                mask = np.full((self.width, self.height), 255, dtype="uint8")
                if y > 0:
                    mask[: self.overlap // 2, :] = 0
                if x > 0:
                    mask[:, : self.overlap // 2] = 0
                mask = Image.fromarray(mask)

                img.paste(tile, (x, y), mask)
                tile.close()

                del tile
                gc.collect()
                # TODO Generalize to other formats
            name, ext = os.path.splitext(self.img_path)
            name += f"_ch_{frame:0>2}"
            channel_path = name + ext
            arr = np.array(img)
            tifsave(channel_path, arr, bigtiff=True)
            # img.save(channel_path, bigtiff=True)
            path_list.append(channel_path)
        logger.info("Finished writing to disk")
        return path_list
