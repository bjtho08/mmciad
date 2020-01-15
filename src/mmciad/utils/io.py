"""Input-Output module. Helps read and write data between persistent
and volatile memory
"""
from os import makedirs
from os.path import join, split, splitext
from glob import glob
from collections import namedtuple
import numpy as np
from skimage.io import imread, imsave
from keras.utils import to_categorical

Size = namedtuple("Size", ["x", "y"])


def create_samples(
    path, filter_dict=None, prefix="train", output_dir=None, duplication_list=None
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
    """
    size = Size(x=208, y=208)
    output_dir = prefix if output_dir is None else output_dir
    makedirs(join(path, output_dir, "gt", ""), exist_ok=True)
    input_images = read_images(path, prefix)
    target_images = read_images(path, prefix, True)
    input_target_pairs = {
        name: (input_image, target_images[name])
        for name, input_image in input_images.items()
    }
    for name, (input_slide, target_slide) in input_target_pairs.items():
        dim_x, dim_y = target_slide.shape[:2]
        x_coordinates, y_coordinates = np.mgrid[0:dim_x:160, 0:dim_y:160]
        points = np.c_[x_coordinates.ravel(), y_coordinates.ravel()]
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
                    path + output_dir + f"/{name}_{current_point:0>4d}.tif",
                    res_x.astype(np.single),
                    check_contrast=False,
                )
                imsave(
                    path + output_dir + f"/gt/{name}_{current_point:0>4d}.tif",
                    res_y.astype(np.ubyte),
                    check_contrast=False,
                )
            if keep and duplication_list is not None:
                for color in duplication_list:
                    if duplicate_tile(res_y, color):
                        imsave(
                            path + output_dir + f"/{name}_1{current_point:0>4d}.tif",
                            res_x.astype("float"),
                            check_contrast=False,
                        )
                        imsave(
                            path + output_dir + f"/gt/{name}_1{current_point:0>4d}.tif",
                            res_y.astype("uint8"),
                            check_contrast=False,
                        )


def read_images(path, prefix, target=False):
    """Read whole slide images into a dictionary. The keys are truncated file IDs

    Parameters
    ----------
    path : str
        path to whole slide images
    prefix : str
        image data set qualifier (usually one of 'train' or 'test')
    target : bool, optional
        defines whether to look for input or target images, by default False

    Returns
    -------
    dict
        dictionary of image arrays
    """
    path = [path]
    ext = "*.tif"
    dtype = np.single
    kwarg = {"fastij": True}
    denom = 255.0
    if target:
        path.append("gt")
        ext = "*.png"
        dtype = np.ubyte
        kwarg = {"pilmode": "RGB"}
        denom = 1
    slide_files = sorted(glob(join(*path, prefix + ext)), key=str.lower)
    slide_names = {
        "-".join(splitext(split(file)[1])[0].split("-")[1:]).replace(
            "-corrected", ""
        ): file
        for file in slide_files
    }
    images = {
        name: imread(path, **kwarg).astype(dtype) / denom
        for name, path in slide_names.items()
    }
    return images


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


def load_slides(
    path, colorvec: list, prefix="N8b", m=None, s=None, gt_path=None, num_cls=None
):
    if gt_path and num_cls is None:
        raise ValueError("Required input missing: num_cls")
    ftype = "*.tif"
    if m is None:
        m = np.array([[[[0.0, 0.0, 0.0]]]])
    if s is None:
        s = np.array([[[[1.0, 1.0, 1.0]]]])
    input_img_files = glob(join(path, prefix + ftype))
    X = np.asarray(
        [
            imread(input_img_files[i]).astype("float") / 255.0
            for i in range(len(input_img_files))
        ]
    )
    for i in range(len(X)):
        X[i] = (X[i] - m) / s
        x_min = X[i].min()
        x_max = X[i].max()
        X[i] = (X[i] - x_min) / (x_max - x_min)
    if gt_path:
        target_img_files = glob(join(path, prefix, "*.png"))
        target_img_samples = imread(target_img_files[0])[:, :, :3]
        target_categorical = np.expand_dims(
            np.zeros(target_img_samples.shape[:2]), axis=-1
        )
        for cls_ in range(num_cls):
            color = colorvec[cls_, :]
            target_categorical += np.expand_dims(
                np.logical_and.reduce(target_img_samples == color, axis=-1) * cls_,
                axis=-1,
            )
        Y = to_categorical(target_categorical, num_classes=num_cls)
        return X, Y
    return X


def load_slides_as_dict(
    path,
    prefix="N8b",
    mean_list=None,
    std_list=None,
    input_hist_range=None,
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
    if mean_list is not None and std_list is not None:
        for i in input_slides.keys():
            input_slides[i] = (input_slides[i] - mean_list) / std_list
        x_min = input_hist_range[0]
        x_max = input_hist_range[1]
        for i in input_slides.keys():
            input_slides[i] = (input_slides[i] - x_min) / (x_max - x_min)
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
