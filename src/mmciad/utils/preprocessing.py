"""Preprocessing module handles all data processing preceding model training
"""
import os
from os.path import join
from glob import glob
import numpy as np
from tqdm import tqdm

from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


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
    pixel_count = 0
    num_channels = 3 if local else 1
    img_sum = np.zeros(num_channels)
    img_sum_squared = np.zeros(num_channels)
    img_min = np.ones(num_channels)
    img_max = np.zeros(num_channels)

    with tqdm(total=len(input_files)) as pbar:
        for img_path in input_files:
            img = imread(img_path)
            img = img/255. if not np.issubdtype(img.dtype, np.floating) else img
            pixel_count += img.size/img.shape[-1]
            img_sum += np.sum(img, axis=(0, 1))
            img_sum_squared += np.square(img_sum)
            with np.nditer(img_min, flags=['c_index'], op_flags=['readwrite']) as it:
                for elem in it:
                    new_min = np.min(img[it.index])
                    elem[...] = new_min if new_min < elem else elem
            with np.nditer(img_max, flags=['c_index'], op_flags=['readwrite']) as it:
                for elem in it:
                    new_max = np.amax(img[it.index])
                    elem[...] = new_max if new_max > elem else elem
            pbar.update(1)
     
    img_mean = img_sum / pixel_count
    img_std = np.sqrt(img_sum_squared / pixel_count - np.square(img_mean))
    return img_mean, img_std, img_min, img_max


def augmentor(img, segmap):
    dtype = img.dtype
    segmap = [SegmentationMapsOnImage(i, shape=segmap[0].shape) for i in segmap]
    preseq = iaa.Sequential(
        [  # augmenters that will affect the input image pixel values
            iaa.OneOf([iaa.Add((-0.07, 0.07)), iaa.Multiply((0.8, 1.2))]),
            iaa.Dropout(p=(0.1, 0.5), per_channel=True),
        ]
    )
    afrot = iaa.Affine(rotate=(-90, 90), mode="reflect")
    afscale = iaa.Affine(scale=(0.8, 1.2), mode="reflect")
    eltrans = iaa.ElasticTransformation(alpha=(50, 200), sigma=(40.0), mode="reflect")
    afrot._mode_segmentation_maps = "reflect"
    afscale._mode_segmentation_maps = "reflect"
    eltrans._mode_segmentation_maps = "reflect"
    seq = iaa.SomeOf(  # augmenters that applies symmetrically
        (0, None), [iaa.Fliplr(1), iaa.Flipud(1), afrot, afscale, eltrans]
    )
    seq_det = seq.to_deterministic()
    img_aug = seq_det.augment_images(
        preseq.augment_images(img.astype("float32").astype(dtype))
    )
    segmap_aug = seq_det.augment_segmentation_maps(segmap)
    segmap_aug = [i.get_arr() for i in segmap_aug]
    return img_aug, segmap_aug

def tf_augmentor(img, segmap):
    dtype = img.dtype
    segmap = SegmentationMapsOnImage(segmap, shape=segmap.shape)
    preseq = iaa.Sequential(
        [  # augmenters that will affect the input image pixel values
            iaa.OneOf([iaa.Add((-20, 20)), iaa.Multiply((0.8, 1.2))]),
            iaa.Dropout(p=(0.01, 0.5), per_channel=False),
        ]
    )
    afrot = iaa.Affine(rotate=(-90, 90), mode="reflect")
    afscale = iaa.Affine(scale=(0.8, 1.2), mode="reflect")
    eltrans = iaa.ElasticTransformation(alpha=(50, 200), sigma=(40.0), mode="reflect")
    afrot._mode_segmentation_maps = "reflect"
    afscale._mode_segmentation_maps = "reflect"
    eltrans._mode_segmentation_maps = "reflect"
    seq = iaa.SomeOf(  # augmenters that applies symmetrically
        (0, None), [iaa.Fliplr(1), iaa.Flipud(1), afrot, afscale, eltrans]
    )
    seq_det = seq.to_deterministic()
    img_aug = seq_det.augment_images(preseq.augment_images(img))
    segmap_aug = seq_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr()
    return img_aug, segmap_aug


def calculate_class_weights(path, class_list, colordict, ignore=None, prefix="train"):
    label_files = glob(os.path.join(path, prefix, "gt", "*.tif"))
    num_img = len(label_files)
    target_tiles = np.asarray(
        [imread(label_files[i])[:, :, :3] for i in range(num_img)]
    )
    target_sparse = [np.zeros(target_tiles[i].shape[:2]) for i in range(num_img)]
    for i in range(num_img):
        for index, class_ in enumerate(class_list):
            color = colordict[class_]
            target_sparse[i] += (
                np.logical_and.reduce(target_tiles[i] == color, axis=-1) * index
            )
    target_tiles = np.asarray(target_sparse, dtype="uint8")
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
    target_sparse = [
        np.expand_dims(np.zeros(target_tiles[i].shape[:2]), axis=-1)
        for i in range(num_img)
    ]
    for i in range(num_img):
        for index, class_ in enumerate(class_list):
            color = colordict[class_]
            target_sparse[i] += np.expand_dims(
                np.logical_and.reduce(target_tiles[i] == color, axis=-1) * index,
                axis=-1,
            )
    target_tiles = np.asarray(target_sparse, dtype="uint8")
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
    for class_int, label in remap_pattern.items():
        output_img[np.isin(img, label)] = class_int
    return output_img
