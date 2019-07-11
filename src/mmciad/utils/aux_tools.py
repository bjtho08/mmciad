import os
from os.path import join
from glob import glob
import numpy as np

from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
from imgaug import augmenters as iaa
from keras.utils import to_categorical

def calculate_stats(path, prefix='train'):
    if isinstance(path, str):
        X_files = glob(join(path, prefix+"*.tif"))
        X = [imread(i).reshape(-1, imread(i).shape[-1]).astype('float')/255. for i in X_files]
    if isinstance(path, list):
        X = path
    means = [0]*3
    stds = [0]*3
    sums = [None]*3
    for c in range(3):
        sums[c] = np.ravel(np.vstack(X)[..., c])
        means[c] = sums[c].mean()
        stds[c] = sums[c].std()
    return means, stds


def augmentor(img, segmap):
    seq = iaa.SomeOf((0, None), [
        iaa.Fliplr(1),
        iaa.Flipud(1),
        iaa.Affine(rotate=(-90, 90), mode="reflect"),
        iaa.Affine(scale=(0.8, 1.2), mode="reflect"),
        iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=6, nb_cols=6, mode='reflect'),
        iaa.ElasticTransformation(alpha=(0, 80), sigma=(8.0), mode="reflect")
    ])
    seq_det = seq.to_deterministic()
    img_aug = seq_det.augment_images(img)
    segmap_aug = seq_det.augment_images(segmap)
    return img_aug, segmap_aug

def calculate_class_weights(path, class_list, colordict, ignore=None, prefix='train'):
    label_files = glob(os.path.join(path, prefix, 'gt', "*.tif"))
    num_img = len(label_files)
    num_classes = len(class_list)
    Y = np.asarray([imread(label_files[i])[:, :, :3] for i in range(num_img)])
    Y_class = [
        np.expand_dims(np.zeros(Y[i].shape[:2]), axis=-1) for i in range(num_img)
    ]
    for i in range(num_img):
        for index, class_ in enumerate(class_list):
            color = colordict[class_]
            Y_class[i] += np.expand_dims(
                np.logical_and.reduce(Y[i] == color, axis=-1) * index, axis=-1)
        Y_class[i] = to_categorical(Y_class[i], num_classes=num_classes)
    Y = np.asarray(Y_class, dtype='uint8')
    Y = np.argmax(Y, axis=-1)
    Y_flat = Y.reshape(Y.size)
    if ignore is not None:
        mask = np.ones_like(Y_flat, dtype=bool)
        mask[Y_flat == ignore] = False
        Y_flat = Y_flat[mask]
    class_weights = compute_class_weight('balanced', np.unique(Y_flat), Y_flat)
    cls_wgts = dict(zip(class_list, class_weights))
    if ignore is not None:
        cls_wgts[ignore] = 0
    return cls_wgts
