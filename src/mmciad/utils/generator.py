"""Create generators for supplying keras models with data
"""
import os.path as osp
from glob import glob
import numpy as np
from skimage.io import imread
from tensorflow.keras.utils import Sequence, to_categorical
from .preprocessing import augmentor


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
        list_IDs=None,
        batch_size=32,
        dim=(208, 208),
        n_channels=3,
        n_classes=10,
        shuffle=True,
        augmenter=True,
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
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augmenter

        if self.list_IDs is None:
            self.list_IDs = [
                osp.splitext(osp.basename(i))[0]
                for i in glob(osp.join(self.path, "*.tif"))
            ]

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 3), dtype=np.uint8)
        y_class = [
            np.zeros((*self.dim, 1), dtype=np.uint8) for _ in range(self.batch_size)
        ]

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = imread(self.path + ID + ".tif").astype("float")
            X[i] = (X[i] - self.means) / self.stds
            X[i] = (X[i] - self.x_min)/(self.x_max - self.x_min)
            # Store class
            y[i,] = imread(self.path + "gt/" + ID + ".tif").astype("int64")
            for cls_ in range(self.n_classes):
                color = self.colorvec[cls_, :]
                y_class[i] += np.expand_dims(
                    np.logical_and.reduce(y[i,] == color, axis=-1) * cls_, axis=-1
                ).astype("uint8")
            y_class[i] = to_categorical(y_class[i], num_classes=self.n_classes)
        if self.augment:
            X, y = augmentor(X, y_class)

        return np.asarray(X, dtype="float"), np.asarray(y, dtype="uint8")
