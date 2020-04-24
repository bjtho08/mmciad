"""Input-Output module. Helps read and write data between persistent
and volatile memory
"""
import os
from os import makedirs
from tempfile import mkstemp
from os.path import join, split, splitext, isdir
from glob import glob
import shutil
from collections import namedtuple, OrderedDict
import numpy as np
from tqdm.auto import tqdm
from skimage.io import imread, imsave
from tensorflow.keras.utils import to_categorical
from PIL import Image

# PIL.Image.DecompressionBombError: could be decompression bomb DOS attack.

Image.MAX_IMAGE_PIXELS = None

Size = namedtuple("Size", ["x", "y"])


def create_samples(
    path: str,
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
        Amount of overlap between neighbouring tiles given in percentage of the tile size
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
    makedirs(join(path, output_dir, "gt", ""), exist_ok=True)
    image_fetcher = ImageFetcher(path, join(path, "gt"), prefix, "tif", "png")
    with tqdm(total=len(image_fetcher)) as pbar:
        for name, input_slide, target_slide in image_fetcher.yield_pair():
            dim_x, dim_y = target_slide.shape[:2]
            x_coords, y_coords = np.mgrid[
                0 : dim_x : overlap_step.x, 0 : dim_y : overlap_step.y
            ]
            points = np.c_[x_coords.ravel(), y_coords.ravel()]
            num_points = points.shape[0]
            for current_point in range(num_points):
                x, y = points[current_point, :]
                res_x = input_slide[x : x + size.x, y : y + size.y, :]
                res_y = target_slide[x : x + size.x, y : y + size.y, :]
                change = False
                if (x + size.x) > dim_x:
                    x = dim_x - size.x
                    change = True
                if (y + size.y) > dim_y:
                    y = dim_y - size.y
                    change = True
                if change:
                    res_x = input_slide[x : x + size.x, y : y + size.y, :]
                    res_y = target_slide[x : x + size.x, y : y + size.y, :]
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
                        path + output_dir + f"/{name}_{current_point:0>4d}.png",
                        res_x,
                        check_contrast=False,
                    )
                    imsave(
                        path + output_dir + f"/gt/{name}_{current_point:0>4d}.png",
                        res_y,
                        check_contrast=False,
                    )
                if keep and duplication_list is not None:
                    for color in duplication_list:
                        if duplicate_tile(res_y, color):
                            imsave(
                                path
                                + output_dir
                                + f"/{name}_dup_{current_point:0>4d}.png",
                                res_x,
                                check_contrast=False,
                            )
                            imsave(
                                path
                                + output_dir
                                + f"/gt/{name}_dup_{current_point:0>4d}.png",
                                res_y,
                                check_contrast=False,
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
        self, input_path: str, target_path: str, prefix, input_ext: str, target_ext: str
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
        assert len(self.input_dict) == len(
            self.target_dict
        ), f"Input/target length mismatch ({len(self.input_dict)} vs {len(self.target_dict)})"

    def __len__(self) -> int:
        """Return length of source directory
        """
        return len(self.input_dict)

    def fetch_list(self, path: str, prefix: str, ext="tif") -> dict:
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

        search_string = [prefix, "*." + ext]
        slide_files = sorted(glob(join(path, "".join(search_string))), key=str.lower)
        file_dict = OrderedDict()
        for file in slide_files:
            file_id = "-".join(splitext(split(file)[1])[0].split("-")[1:]).replace(
                "-corrected", ""
            )
            file_dict[file_id] = file
        return file_dict

    def yield_pair(self):
        input_kwargs = {
            "kwarg": {"plugin": "tifffile"},
        }
        target_kwargs = {
            "kwarg": {"pilmode": "RGB"},
        }

        for name, input_img_path in self.input_dict.items():
            target_img_path = self.target_dict[name]
            yield (
                name,
                imread(input_img_path, **input_kwargs["kwarg"]),
                imread(target_img_path, **target_kwargs["kwarg"]),
            )


def duplicate_tile(tile, label_color):
    detection_threshold = 0.05
    segmap = np.logical_and.reduce(tile == label_color, axis=-1)
    if segmap.sum() >= segmap.size * detection_threshold:
        return True
    return False


def check_class(
    segmap, class_colors, probability=0.9, lower_threshold=0.9, upper_threshold=None
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
    assert hasattr(class_colors, "__iter__"), "class_colors must be a list of lists!"
    template = np.zeros(segmap.shape[:-1])
    for class_color in class_colors:
        template += np.logical_and.reduce(segmap == class_color, axis=-1)
    if (
        upper_threshold is not None
        and template.sum() >= template.size * upper_threshold
    ):
        return True
    if template.sum() >= template.size * lower_threshold:
        return np.random.rand() > 1 - probability
    return False


# def load_slides(
#     path, colorvec: list, prefix="N8b", m=None, s=None, gt_path=None, num_cls=None
# ):
#     if gt_path and num_cls is None:
#         raise ValueError("Required input missing: num_cls")
#     ftype = "*.tif"
#     if m is None:
#         m = np.array([[[[0.0, 0.0, 0.0]]]])
#     if s is None:
#         s = np.array([[[[1.0, 1.0, 1.0]]]])
#     input_img_files = glob(join(path, prefix + ftype))
#     X = np.asarray(
#         [
#             imread(input_img_files[i]).astype("float") / 255.0
#             for i in range(len(input_img_files))
#         ]
#     )
#     for i in range(len(X)):
#         X[i] = (X[i] - m) / s
#         x_min = X[i].min()
#         x_max = X[i].max()
#         X[i] = (X[i] - x_min) / (x_max - x_min)
#     if gt_path:
#         target_img_files = glob(join(path, prefix, "*.png"))
#         target_img_samples = imread(target_img_files[0])[:, :, :3]
#         target_categorical = np.expand_dims(
#             np.zeros(target_img_samples.shape[:2]), axis=-1
#         )
#         for cls_ in range(num_cls):
#             color = colorvec[cls_, :]
#             target_categorical += np.expand_dims(
#                 np.logical_and.reduce(target_img_samples == color, axis=-1) * cls_,
#                 axis=-1,
#             )
#         Y = to_categorical(target_categorical, num_classes=num_cls)
#         return X, Y
#     return X


def load_slides_as_dict(
    path,
    prefix="N8b",
    mean_list=None,
    std_list=None,
    input_max=255,
    gt_path=None,
    num_cls=None,
    colors=None,
):
    if gt_path and num_cls is None:
        raise ValueError("Required input missing: num_cls")
    if gt_path and colors is None:
        raise ValueError("Required input missing: num_cls")
    ftype = "*.tif"
    if (mean_list is None or std_list is None) and (std_list != mean_list):
        raise ValueError("Both m and s or neither must be supplied")
    input_img_files = sorted(glob(join(path, prefix + ftype)))
    slide_names = [
        "-".join(splitext(split(file)[1])[0].split("-")[1:]).replace("-corrected", "")
        for file in input_img_files
    ]
    input_slides = {
        name: imread(path).astype("float") / 255.0
        for name, path in zip(slide_names, input_img_files)
    }
    for img in input_slides.values():
        if img.size > 3072 ** 2:
            memmap = True
    if memmap:
        memmap_input_slides = {
            name: np.memmap(
                mkstemp(dir=path + "../tmp")[1],
                dtype=np.floating,
                mode="w+",
                shape=img.shape,
            )
            for name, img in input_slides.items()
        }
        for name, img in memmap_input_slides.items():
            memmap_input_slides[name][:] = input_slides[name][:]
            input_slides.pop(name)
        input_slides = memmap_input_slides

    if mean_list is not None and std_list is not None:
        if memmap:
            for i in tqdm(input_slides.keys()):
                tmp = np.memmap(
                    mkstemp(dir=path + "../tmp")[1],
                    dtype=np.floating,
                    mode="w+",
                    shape=img.shape,
                )
                scaffold: np.array = np.zeros(img.shape[:1])
                pbar = tqdm(total=scaffold[:].shape[0])
                it = np.nditer(scaffold, flags=["multi_index"])
                while not it.finished:
                    tmp[it.multi_index] = (
                        (input_slides[i][it.multi_index] - mean_list) / std_list
                    ) / input_max
                    pbar.update(1)
                    it.iternext()
                pbar.close()
                tmp.flush()
                input_slides[i] = tmp
            for i in memmap_input_slides.values():
                os.unlink(i.filename)
            del memmap_input_slides
        else:
            for i in input_slides.keys():
                input_slides[i] = ((input_slides[i] - mean_list) / std_list) / input_max
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
        target_sparse = {
            name: np.expand_dims(np.zeros(slide.shape[:2]), axis=-1)
            for name, slide in target_color.items()
        }
        for cls_ in range(num_cls):
            color = colors[cls_, :]
            for name in target_sparse.keys():
                target_sparse[name] += np.expand_dims(
                    np.logical_and.reduce(target_color[name] == color, axis=-1) * cls_,
                    axis=-1,
                )
        target = {
            name: to_categorical(slide, num_classes=num_cls)
            for name, slide in target_sparse.items()
        }
        return input_slides, target
    return input_slides


def move_files_in_dir(src_dir: str, dst_dir: str, pattern=None) -> None:
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
