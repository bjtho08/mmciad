"""Create generators for supplying keras models with data
"""
import os.path as osp
from glob import glob
import numpy as np
from skimage.io import imread
from keras.utils import Sequence, to_categorical
from .preprocessing import augmentor, merge_labels


class DataGenerator(Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        path,
        colorvec,
        means,
        stds,
        x_min,
        x_max,
        id_list=None,
        batch_size=32,
        dim=(208, 208),
        n_channels=3,
        n_classes=10,
        shuffle=True,
        augmenter=True,
        remap_labels=None,
    ):
        "Initialization"
        self.dim = dim
        self.path = path
        self.colorvec = colorvec
        self.means = means
        self.stds = stds
        self.x_min = x_min
        self.x_max = x_max
        self.batch_size = batch_size
        self.id_list = id_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augmenter
        self.remap_labels = remap_labels

        if self.id_list is None:
            self.id_list = [
                osp.splitext(osp.basename(i))[0]
                for i in glob(osp.join(self.path, "*.tif"))
            ]

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.id_list) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        id_list_temp = [self.id_list[k] for k in indexes]

        # Generate data
        input_img_batch, target_batch = self.__data_generation(id_list_temp)

        return input_img_batch, target_batch

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.id_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, id_list_temp):
        "Generates data containing batch_size samples"
        # input_img_batch : (n_samples, *dim, n_channels)
        # Initialization
        input_img_batch = np.empty((self.batch_size, *self.dim, self.n_channels))
        target_batch = np.empty((self.batch_size, *self.dim, 3), dtype=np.uint8)
        target_batch_class = [
            np.zeros((*self.dim, 1), dtype=np.uint8) for _ in range(self.batch_size)
        ]

        # Generate data
        for i, sample_id in enumerate(id_list_temp):
            # Store sample
            input_img_batch[i] = imread(self.path + sample_id + ".tif")
            input_img_batch = input_img_batch.astype(np.float32, copy=False)
            input_img_batch[i] = (input_img_batch[i] - self.means) / self.stds
            input_img_batch[i] = (input_img_batch[i] - self.x_min) / (
                self.x_max - self.x_min
            )
            # Store class
            target_batch[i,] = imread(self.path + "gt/" + sample_id + ".tif").astype(
                "int64"
            )
            for cls_ in range(self.n_classes):
                color = self.colorvec[cls_, :]
                target_batch_class[i] += np.expand_dims(
                    np.logical_and.reduce(target_batch[i,] == color, axis=-1) * cls_,
                    axis=-1,
                ).astype("uint8")
            if isinstance(self.remap_labels, dict):
                self.n_classes = len(self.remap_labels)
                target_batch_class[i] = merge_labels(
                    target_batch_class[i], self.remap_labels
                )
        if self.augment:
            assert np.issubdtype(input_img_batch.dtype, np.float32)
            input_img_batch, target_batch_class = augmentor(
                input_img_batch, target_batch_class
            )
        target_batch = [
            to_categorical(target, num_classes=self.n_classes)
            for target in target_batch_class
        ]
        return (
            np.asarray(input_img_batch, dtype="float64"),
            np.asarray(target_batch, dtype="uint8"),
        )
