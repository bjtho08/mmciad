import os
from os.path import join
from glob import glob
import numpy as np

from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
from imgaug import augmenters as iaa
from keras.utils import to_categorical


def calculate_stats(input_tiles=None, path=None, prefix="train", local=True):
    """Calculate mean and standard deviation for input image dataset,
    either local (per channel, default) or global (all channels).
    
    Parameters
    ----------
    input_tiles : list of array-like, optional
        data structure containing image arrays, by default None

    path : str, optional
        path to folder containing the dataset, by default None.
        If path is specified, param X will not be used.

    prefix : str, optional
        filename prefix, by default "train"

    local : bool, optional
        determines statistics are calculated per channel or
        not, by default True
    
    Returns
    -------
    lists or ints
        calculated statistics. if local is True, the function
        returns two lists of ints, otherwise two ints
    """ 


    if isinstance(path, str):
        input_files = glob(join(path, prefix + "*.tif"))
        input_tiles = [
            imread(i).reshape(-1, imread(i).shape[-1]).astype(np.float) / 255.0
            for i in input_files
        ]
    input_tiles = np.vstack(input_tiles)
    input_tiles = input_tiles[None, None, :]
    if local:
        means, stds = (
            input_tiles.mean(axis=-2, keepdims=True),
            input_tiles.std(axis=-2, keepdims=True),
        )
        input_tiles_center = (input_tiles - means) / stds
        mins = input_tiles_center.min(axis=-2, keepdims=True)
        maxs = input_tiles_center.max(axis=-2, keepdims=True)
        return means, stds, mins, maxs
    if not local:
        means, stds = (input_tiles.mean(keepdims=True), input_tiles.std(keepdims=True))
        input_tiles_center = (input_tiles - means) / stds
        mins = input_tiles_center.min(axis=-2, keepdims=True)
        maxs = input_tiles_center.max(axis=-2, keepdims=True)
        return means, stds, mins, maxs

def histogram_augment(images, random_state, parents, hooks):
    for img in images:
        pass
    return images

def preaugment(img_list):
    seq = iaa.Sequential([ # augmenters that will affect the input image pixel values
        iaa.OneOf([iaa.Add((-0.07, 0.07)), iaa.Multiply((0.8, 1.2))]),
        iaa.Dropout(p=(0.1, 0.5), per_channel=True)
    ])
    return seq.augment_images(img_list)

def augmentor(img, segmap):
    dtype = img.dtype
    preseq = iaa.Sequential([ # augmenters that will affect the input image pixel values
        iaa.OneOf([iaa.Add((-0.07, 0.07)), iaa.Multiply((0.8, 1.2))]),
        iaa.Dropout(p=(0.1, 0.5), per_channel=True),
    ])
    seq = iaa.SomeOf( # augmenters that will distort input and target images in identical fashion
        (0, None),
        [
            iaa.Fliplr(1),
            iaa.Flipud(1),
            iaa.Affine(rotate=(-90, 90), mode="reflect"),
            iaa.Affine(scale=(0.8, 1.2), mode="reflect"),
            iaa.PiecewiseAffine(
                scale=(0.01, 0.05), nb_rows=6, nb_cols=6, mode="reflect"
            ),
            iaa.ElasticTransformation(alpha=(0, 80), sigma=(8.0), mode="reflect"),
        ]
    )
    seq_det = seq.to_deterministic()
    img_aug = seq_det.augment_images(preseq.augment_images(img.astype("float32").astype(dtype)))
    segmap_aug = seq_det.augment_images(segmap)
    return img_aug, segmap_aug


def calculate_class_weights(path, class_list, colordict, ignore=None, prefix="train"):
    label_files = glob(os.path.join(path, prefix, "gt", "*.tif"))
    num_img = len(label_files)
    num_classes = len(class_list)
    target_tiles = np.asarray(
        [imread(label_files[i])[:, :, :3] for i in range(num_img)]
    )
    target_categorical = [
        np.expand_dims(np.zeros(target_tiles[i].shape[:2]), axis=-1)
        for i in range(num_img)
    ]
    for i in range(num_img):
        for index, class_ in enumerate(class_list):
            color = colordict[class_]
            target_categorical[i] += np.expand_dims(
                np.logical_and.reduce(target_tiles[i] == color, axis=-1) * index,
                axis=-1,
            )
        target_categorical[i] = to_categorical(
            target_categorical[i], num_classes=num_classes
        )
    target_tiles = np.asarray(target_categorical, dtype="uint8")
    target_tiles = np.argmax(target_tiles, axis=-1)
    target_tiles_flat = target_tiles.reshape(target_tiles.size)
    if ignore is not None:
        mask = np.ones_like(target_tiles_flat, dtype=bool)
        mask[target_tiles_flat == ignore] = False
        target_tiles_flat = target_tiles_flat[mask]
    class_weights = compute_class_weight(
        "balanced", np.unique(target_tiles_flat), target_tiles_flat
    )
    cls_wgts = dict(zip(class_list, class_weights))
    if ignore is not None:
        cls_wgts[ignore] = 0
    return cls_wgts


def class_ratio(path, class_list, colordict, prefix="train"):
    label_files = glob(os.path.join(path, prefix, "gt", "*.tif"))
    num_img = len(label_files)
    num_classes = len(class_list)
    target_tiles = np.asarray(
        [imread(label_files[i])[:, :, :3] for i in range(num_img)]
    )
    target_categorical = [
        np.expand_dims(np.zeros(target_tiles[i].shape[:2]), axis=-1)
        for i in range(num_img)
    ]
    for i in range(num_img):
        for index, class_ in enumerate(class_list):
            color = colordict[class_]
            target_categorical[i] += np.expand_dims(
                np.logical_and.reduce(target_tiles[i] == color, axis=-1) * index,
                axis=-1,
            )
        target_categorical[i] = to_categorical(
            target_categorical[i], num_classes=num_classes
        )
    target_tiles = np.asarray(target_categorical, dtype="uint8")
    target_tiles = np.argmax(target_tiles, axis=-1)
    target_tiles_flat = target_tiles.reshape(target_tiles.size)
    class_ratios = np.bincount(target_tiles_flat)
    return {
        label: count / target_tiles_flat.size
        for label, count in zip(class_list, class_ratios)
    }


def merge_labels(img, remap_pattern: dict):
    """Remap values of an 8-bit image according to the supplied dict()
    
    Parameters
    ----------
    img : Array-like
        Target labels to be remapped
    remap_pattern : dict
        Dict with the format:
        {out_lbl_1: [in_lbl_1, ..., in_lbl_n],}
    
    Example
    -------
    >>> target_image = np.random.randint(11, size=(208, 208, 1), dtype="uint8")
    >>> print(np.max(target_image.max))
    10
    >>> print(target_image.shape)
    (208, 208, 1)
    >>> pattern = {0: [0, 2, 4, 5], 1: [3, 6, 7, 8], 2: [1, 9, 10]}
    >>> new_target = merge_labels(target_image, pattern)
    >>> print(np.max(new_target))
    2
    >>> print(new_target.shape)
    (208, 208, 1)
    """

    output_img = np.zeros_like(img).astype("uint8")
    # assert set(range(img.max())) == {
    #     val for label in remap_pattern.values() for val in label
    # }, "pattern must remap all original labels!"
    for k, v in remap_pattern.items():
        output_img[np.isin(img, v)] = k
    return output_img
