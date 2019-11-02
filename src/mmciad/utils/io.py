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
from .preprocessing import calculate_stats

Size = namedtuple("Size", ["x", "y"])


def read_samples(path, colors, prefix="train", n_samples=None, num_cls=None):
    """ Read all sample slides and subdivide into square tiles.

    Parameters
    ----------
    path : str
        Identifies the root data folder containing the whole slide images, e.g.
        .../path/train-N8b.tif
        .../path/gt/train-N8b.png
    colors : np.ndarray
        n x 3 array containing the RGB colors for each of the n classes
    prefix : str
        Filename prefix, e.g.
        'train', 'test', 'val'
    n_samples : int or None
        Number of sample tiles to extract from each whole slide image.

        If n_samples == None: the images will be subdivided into a regular grid
        with overlapping tiles.

        if n_samples > 0: n_slides*n_samples tiles will be created by creating
        coordinate paris at random and extracting the tiles from these locations.
    """
    if num_cls is None:
        raise ValueError("Required input missing: num_cls")
    size = (208, 208)
    X_files = sorted(glob(join(path, prefix + "*.tif")), key=str.lower)
    Y_files = sorted(glob(join(path, "gt", prefix + "*.png")), key=str.lower)
    num_img = len(X_files)
    assert len(X_files) == len(Y_files)
    X_samples = [
        np.asarray(imread(X_files[i]).astype("float") / 255.0) for i in range(num_img)
    ]
    means, stds = calculate_stats(X_samples)
    for i in range(num_img):
        X_samples[i] = (X_samples[i] - means) / stds
    x_range = np.asarray(
        [[X_samples[i].min(), X_samples[i].max()] for i in range(num_img)],
    )
    x_min = x_range.min()
    x_max = x_range.max()
    for i in range(num_img):
        X_samples[i] = (X_samples[i] - x_min)/(x_max - x_min)
    Y_samples = np.asarray([imread(Y_files[i])[:, :, :3] for i in range(num_img)])
    Y_class = rgb_to_indexed(Y_samples, i, num_img, colors, num_cls)
    X = []
    Y = []
    if n_samples is not None:
        for i in range(num_img):
            X_, Y_ = Y_samples[i].shape[:2]
            max_shape = np.array(
                (np.asarray(Y_samples[i].shape[:2]) - np.array(size)) / 200
            )
            points = (
                np.c_[
                    np.random.randint(max_shape[0], size=n_samples),
                    np.random.randint(max_shape[1], size=n_samples),
                ]
                * 200
            )
            for n in range(n_samples):
                x, y = points[n, :]
                X.append(X_samples[i][x : x + size[0], y : y + size[1], :])
                Y.append(Y_class[i][x : x + size[0], y : y + size[1], :])
    else:
        for i in range(num_img):
            X_, Y_ = Y_samples[i].shape[:2]
            px, py = np.mgrid[0:X_:160, 0:Y_:160]
            points = np.c_[px.ravel(), py.ravel()]
            pr = points.shape[0]
            for n in range(pr):
                x, y = points[n, :]
                res_x = X_samples[i][x : x + size[0], y : y + size[1], :]
                res_y = Y_class[i][x : x + size[0], y : y + size[1], :]
                change = False
                if (x + size[0]) > X_:
                    x = X_ - size[0]
                    change = True
                if (y + size[1]) > Y_:
                    y = Y_ - size[1]
                    change = True
                if change:
                    res_x = X_samples[i][x : x + size[0], y : y + size[1], :]
                    res_y = Y_class[i][x : x + size[0], y : y + size[1], :]
                X.append(res_x)
                Y.append(res_y)
    X = np.asarray(X, dtype="float")
    Y = np.asarray(Y, dtype="uint8")
    return X, Y, means, stds, points

def rgb_to_indexed(Y_samples, i, num_img, colors, num_cls):
    Y_class = [
        np.expand_dims(np.zeros(Y_samples[i].shape[:2]), axis=-1)
        for i in range(num_img)
    ]
    for i in range(num_img):
        for cls_ in range(colors.shape[0]):
            color = colors[cls_, :]
            Y_class[i] += np.expand_dims(
                np.logical_and.reduce(Y_samples[i] == color, axis=-1) * cls_, axis=-1
            )
        Y_class[i] = to_categorical(Y_class[i], num_classes=num_cls)
    return Y_class


def create_samples(path, bg_color, ignore_color, prefix="train", duplication_list=None):
    """ Read all sample slides and subdivide into square tiles. The resulting tiles are saved
    as .tif files in corresponding directories.

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
    """
    size = Size(x=208, y=208)
    makedirs(join(path, prefix, "gt", ""), exist_ok=True)
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
            # Check if res_y contains any pixels with the ignore label
            keep = True
            if keep and check_class(res_y, ignore_color, upper_threshold=1.0):
                keep = False
                # Check if res_y contains enough pixels with background label
            if keep and check_class(res_y, bg_color, upper_threshold=1.0):
                keep = False
            if keep and check_class(
                res_y, bg_color, probability=0.5, lower_threshold=0.7
            ):
                keep = False
            if keep:
                imsave(
                    path + prefix + f"/{name}_{current_point:0>4d}.tif",
                    res_x.astype(np.single),
                    check_contrast=False,
                )
                imsave(
                    path + prefix + f"/gt/{name}_{current_point:0>4d}.tif",
                    res_y.astype(np.ubyte),
                    check_contrast=False,
                )
            if keep and duplication_list is not None:
                for color in duplication_list:
                    if duplicate_tile(res_y, color):
                        imsave(
                            path + prefix + f"/{name}_1{current_point:0>4d}.tif",
                            res_x.astype("float"),
                            check_contrast=False,
                        )
                        imsave(
                            path + prefix + f"/gt/{name}_1{current_point:0>4d}.tif",
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
    slide_files = (sorted(glob(join(*path, prefix + ext)), key=str.lower))
    slide_names = {
        '-'.join(splitext(split(file)[1])[0].split("-")[1:]).replace("-corrected", ""): file
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
    segmap, class_color, probability=0.9, lower_threshold=0.9, upper_threshold=None
):
    """ Filter input based on how much of a given class is present in the input image.
    Returns True if image should be filtered.

    Parameters
    ----------
    :param segmap: The input image to be filtered
    :type segmap: array-like
    :param class_color: list of length 3 with RGB color code matching
                        the class color checking against
    :type class_color: list of ints
    :param probability: Probability of image being filtered if above a
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
    segmap = np.logical_and.reduce(segmap == class_color, axis=-1)
    if upper_threshold is not None and segmap.sum() >= segmap.size * upper_threshold:
        return True
    if segmap.sum() >= segmap.size * lower_threshold:
        return np.random.rand() > 1 - probability
    return False


def load_slides(path, colorvec: list, prefix="N8b", m=None, s=None, load_gt=True, num_cls=None):
    if load_gt and num_cls is None:
        raise ValueError("Required input missing: num_cls")
    ftype = "*.tif"
    if m is None:
        m = np.array([[[[0.0, 0.0, 0.0]]]])
    if s is None:
        s = np.array([[[[1.0, 1.0, 1.0]]]])
    X_files = glob(join(path, prefix + ftype))
    X = np.asarray(
        [imread(X_files[i]).astype("float") / 255.0 for i in range(len(X_files))]
    )
    for i in range(len(X)):
        X[i] = (X[i] - m) / s
        x_min = X[i].min()
        x_max = X[i].max()
        X[i] = (X[i] - x_min)/(x_max - x_min)
    if load_gt:
        Y_files = glob(join(path, prefix, "*.png"))
        Y_samples = imread(Y_files[0])[:, :, :3]
        Y_class = np.expand_dims(np.zeros(Y_samples.shape[:2]), axis=-1)
        for cls_ in range(num_cls):
            color = colorvec[cls_, :]
            Y_class += np.expand_dims(
                np.logical_and.reduce(Y_samples == color, axis=-1) * cls_, axis=-1
            )
        Y = to_categorical(Y_class, num_classes=num_cls)
        return X, Y
    return X


def load_slides_as_dict(
    path, prefix="N8b", m=None, s=None, x_minmax=None, load_gt=True, num_cls=None, colors=None
):
    if load_gt and num_cls is None:
        raise ValueError("Required input missing: num_cls")
    if load_gt and colors is None:
        raise ValueError("Required input missing: num_cls")
    ftype = "*.tif"
    if (m is None or s is None) and (s != m):
        raise ValueError("Both m and s or neither must be supplied")
    X_files = sorted(glob(join(path, prefix + ftype)))
    slide_names = [
        '-'.join(splitext(split(file)[1])[0].split("-")[1:]).replace("-corrected", "")
        for file in X_files
    ]
    X = {
        name: imread(path).astype("float") / 255.0
        for name, path in zip(slide_names, X_files)
    }
    if m is not None and s is not None:
        for i in X.keys():
            X[i] = (X[i] - m) / s
        x_min = x_minmax[0]
        x_max = x_minmax[1]
        for i in X.keys():
            X[i] = (X[i] - x_min)/(x_max - x_min)
    if load_gt:
        Y_files = sorted(glob(join(path, 'gt', prefix + "*.png")))
        slide_names = [
            '-'.join(splitext(split(file)[1])[0].split("-")[1:]).replace("-corrected", "")
            for file in Y_files
        ]
        Y_color = {
            name: imread(path).astype("uint8")[:, :, :3]
            for name, path in zip(slide_names, Y_files)
        }
        Y_sparse = {
            name: np.expand_dims(np.zeros(slide.shape[:2]), axis=-1)
            for name, slide in Y_color.items()
        }
        for cls_ in range(num_cls):
            color = colors[cls_, :]
            for name in Y_sparse.keys():
                Y_sparse[name] += np.expand_dims(
                    np.logical_and.reduce(Y_color[name] == color, axis=-1) * cls_,
                    axis=-1,
                )
        Y = {
            name: to_categorical(slide, num_classes=num_cls)
            for name, slide in Y_sparse.items()
        }
        return X, Y
    return X
