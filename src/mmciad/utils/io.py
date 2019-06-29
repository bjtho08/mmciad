"""Input-Output module. Helps read and write data between persistent
and volatile memory
"""
from os.path import join, split, splitext
from glob import glob
import numpy as np
from skimage.io import imread, imsave
from keras.utils import to_categorical
from .aux_tools import calculate_stats


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
        for c in range(3):
            X_samples[i][..., c] = (X_samples[i][..., c] - means[c]) / stds[c]
    Y_samples = np.asarray([imread(Y_files[i])[:, :, :3] for i in range(num_img)])
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


def create_samples(path, bg_color, ignore_color, prefix="train"):
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
    size = (208, 208)
    X_files = sorted(glob(join(path, prefix + "*.tif")), key=str.lower)
    Y_files = sorted(glob(join(path, "gt", prefix + "*.png")), key=str.lower)
    num_img = len(X_files)
    X_samples = np.asarray(
        [imread(X_files[i]).astype("float") / 255.0 for i in range(num_img)]
    )
    Y_samples = np.asarray([imread(Y_files[i])[:, :, :3] for i in range(num_img)])
    X = []  # np.zeros(shape=(n_samples*num_img, size[0], size[1], 3), dtype="float")
    Y = (
        []
    )  # np.zeros(shape=(n_samples*num_img, size[0], size[1], num_cls), dtype="uint8")
    for i in range(num_img):
        X_, Y_ = Y_samples[i].shape[:2]
        px, py = np.mgrid[0:X_:160, 0:Y_:160]
        points = np.c_[px.ravel(), py.ravel()]
        pr = points.shape[0]
        for n in range(pr):
            x, y = points[n, :]
            res_x = X_samples[i][x : x + size[0], y : y + size[1], :]
            res_y = Y_samples[i][x : x + size[0], y : y + size[1], :]
            change = False
            if (x + size[0]) > X_:
                x = X_ - size[0]
                change = True
            if (y + size[1]) > Y_:
                y = Y_ - size[1]
                change = True
            if change:
                res_x = X_samples[i][x : x + size[0], y : y + size[1], :]
                res_y = Y_samples[i][x : x + size[0], y : y + size[1], :]
            # Check if res_y contains any pixels with the ignore label
            keep = True
            if keep and check_class(res_y, ignore_color, probability=1):
                keep = False
                # Check if res_y contains enough pixels with background label
            if keep and check_class(res_y, bg_color):
                keep = False
            if keep and check_class(
                res_y, bg_color, probability=0.5, lower_threshold=0.7
            ):
                keep = False
            if keep:
                X.append(res_x)
                Y.append(res_y)
    X = np.asarray(X, dtype="float")
    Y = np.asarray(Y, dtype="uint8")
    for i in range(len(X)):  ## Check_contrast will be available from version 0.15
        imsave(path + prefix + "/X_{}.tif".format(i), X[i, ...], check_contrast=False)
        imsave(
            path + prefix + "/gt/X_{}.tif".format(i), Y[i, ...], check_contrast=False
        )


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


def load_slides(path, prefix="N8b", m=None, s=None, load_gt=True, num_cls=None):
    if load_gt and num_cls is None:
        raise ValueError("Required input missing: num_cls")
    ftype = "*.tif"
    if m is None:
        m = [0.0, 0.0, 0.0]
    if s is None:
        s = [1.0, 1.0, 1.0]
    X_files = glob(join(path, prefix + ftype))
    X = np.asarray(
        [imread(X_files[i]).astype("float") / 255.0 for i in range(len(X_files))]
    )
    for i in range(len(X)):
        for c in range(3):
            X[i][..., c] = (X[i][..., c] - m[c]) / s[c]
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
    path, prefix="N8b", m=None, s=None, load_gt=True, num_cls=None, colors=None
):
    if load_gt and num_cls is None:
        raise ValueError("Required input missing: num_cls")
    if load_gt and colors is None:
        raise ValueError("Required input missing: num_cls")
    ftype = "*.tif"
    if m is None:
        m = [0.0, 0.0, 0.0]
    if s is None:
        s = [1.0, 1.0, 1.0]
    X_files = sorted(glob(join(path, prefix + ftype)))
    slide_names = [
        '-'.join(splitext(split(file)[1])[0].split("-")[1:]).replace("-corrected", "")
        for file in X_files
    ]
    X = {
        name: imread(path).astype("float") / 255.0
        for name, path in zip(slide_names, X_files)
    }
    for i in X.keys():
        for c in range(3):
            X[i][..., c] = (X[i][..., c] - m[c]) / s[c]
    if load_gt:
        Y_files = sorted(glob(join(path, prefix, "*.png")))
        Y_color = {
            name: imread(path).astype("float")[:, :, :3] / 255.0
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
