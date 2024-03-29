"""Create generators for supplying keras models with data
"""
from pathlib import Path
import numpy as np
from skimage.io import imread
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tf_mmciad.utils.preprocessing import augmentor, tf_augmentor, merge_labels

RGB = 3
INDEXED = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        path,
        color_dict,
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
        augment=True,
        remap_labels=None,
    ):
        "Initialization"
        self.dim = dim
        self.path = Path(path)
        self.color_dict = color_dict
        self.means = means
        self.stds = stds
        self.x_min = x_min
        self.x_max = x_max
        self.batch_size = batch_size
        self.id_list = id_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.remap_labels = remap_labels
        if self.id_list is None:
            self.id_list = [i.stem for i in sorted(self.path.glob("*.png"))]

        self.__selftest()
        self.on_epoch_end()

    # def __next__(self):
    #     data = self.__getitem__(self.n)
    #     self.n += 1

    #     if self.n >= self.__len__():
    #         self.on_epoch_end()
    #         self.n = 0
    #     return data

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.id_list) // self.batch_size

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        if index > self.__len__():
            raise IndexError(f"Index out of bounds ({index} > {self.__len__()})")
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
        input_img_batch = np.empty(
            (self.batch_size, *self.dim, self.n_channels), dtype=np.float32
        )
        target_batch_class = [
            np.empty((*self.dim,), dtype=np.uint8) for _ in range(self.batch_size)
        ]

        # Generate data
        for i, sample_id in enumerate(id_list_temp):
            # Store sample
            input_img_batch[i] = imread(self.path / f"{sample_id}.png")
            input_img_batch = input_img_batch.astype(np.float32, copy=False)
            input_img_batch[i] = (input_img_batch[i] - self.means) / self.stds
            # input_img_batch[i] = (input_img_batch[i] - self.x_min) / (
            #    self.x_max - self.x_min
            # )
            # Store class
            unproc_img = np.array(
                Image.open(self.path / "gt" / f"{sample_id}.png")
            )
            shape = unproc_img.shape
            np.putmask(unproc_img, unproc_img == 0, 1)
            proc_img = unproc_img.reshape(shape)
            target_batch_class[i] = proc_img
            if isinstance(self.remap_labels, dict):
                self.n_classes = len(self.remap_labels)
                target_batch_class[i] = merge_labels(
                    target_batch_class[i], self.remap_labels
                )
        if self.augment:
            dtype = input_img_batch.dtype
            if not np.issubdtype(dtype, np.floating):
                raise ValueError(f"img dtype, {dtype}, does not match np.floating")
            input_img_batch, target_batch_class = augmentor(
                input_img_batch, target_batch_class
            )
        target_batch = [
            to_categorical(target, num_classes=self.n_classes)
            for target in target_batch_class
        ]
        if self.batch_size == 1:
            return (
                np.squeeze(np.asarray(input_img_batch, dtype="float64")),
                np.squeeze(np.asarray(target_batch, dtype="uint8")),
            )
        return (
            np.asarray(input_img_batch, dtype="float64"),
            np.asarray(target_batch, dtype="uint8"),
        )

    def __selftest(self):
        """Generator self test. Returns None when nothing fails

        Raises:
            TypeError: Wrong data type for batch_size
            ValueError: Invalid value for batch_size
            FileNotFoundError: No matching files found in specified path
        """
        if not isinstance(self.batch_size, int):
            raise TypeError(
                f"Batch size: {type(self.batch_size)} is not an instance of int()"
            )
        if self.batch_size < 1:
            raise ValueError(f"Batch size: {self.batch_size} cannot be less than 1")
        if len(self.id_list) < 1:
            raise FileNotFoundError(
                f"Empty dataset ({len(self.id_list)}). Verify path ({self.path.resolve()})"
            )


class DataSet(object):
    """[summary]

    Parameters
    ----------
    object : [type]
        [description]
    """

    def __init__(
        self,
        path,
        color_dict,
        means,
        stds,
        x_min,
        x_max,
        batch_size=32,
        dim=(208, 208),
        n_channels=3,
        n_classes=10,
        augment=True,
    ):
        "Initialization"
        self.dim = dim
        self.path = path
        self.color_dict = color_dict
        self.means = means
        self.stds = stds
        self.x_min = x_min
        self.x_max = x_max
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augment = augment
        self.file_list = tf.data.Dataset.list_files(path)
        self.data = self.file_list.map(self.process_path, num_parallel_calls=AUTOTUNE)

    def __len__(self):
        "Denotes the number of batches per epoch"
        num_files = tf.data.experimental.cardinality(self.file_list).numpy()
        return int(np.floor(num_files / self.batch_size))

    def decode_img(self, file_path: tf.Tensor) -> tf.Tensor:
        """Read and decode images in path

        Parameters
        ----------
        file_path : tf.Tensor
            path to images

        Returns
        -------
        tf.Tensor
            Decoded images
        """
        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.convert_to_tensor(img)
        return img

    def process_path(self, file_path: tf.Tensor):
        """Read and decode images in path

        Parameters
        ----------
        file_path : tf.Tensor
            Path to example data

        Returns
        -------
        tf.Tensor
            Example-target pairs
        """
        gt_path = tf.strings.regex_replace(file_path, r"(/)([0-9]{4})", r"\1gt\1\2")
        ground_truth = self.decode_img(gt_path)
        gt = ground_truth.eval()
        gt_class = np.zeros((*self.dim, 1), dtype=np.uint8)
        for cls_, color in enumerate(self.color_dict.values()):
            gt_class += np.expand_dims(
                np.logical_and.reduce(gt == color, axis=-1) * cls_, axis=-1,
            )
        ground_truth = to_categorical(gt_class, num_classes=self.n_classes)
        ground_truth = tf.convert_to_tensor(ground_truth)
        # load the raw data from the file as a string
        img = self.decode_img(file_path)
        return img, ground_truth

    def augment_fn(self):
        """Augment data in dataset
        """

        def augment(examples, targets):
            ex_shape = examples.shape
            ta_shape = targets.shape
            [examples, targets] = tf.numpy_function(
                tf_augmentor, [examples, targets], [examples.dtype, targets.dtype]
            )
            examples.set_shape(ex_shape)
            targets.set_shape(ta_shape)
            return examples, targets

        return augment

    def prepare_for_training(self, size=None):
        """Prepare generator for training

        Parameters
        ----------
        size : int, optional
            Batch size, by default None
        """
        self.data = self.data.map(self.augment_fn())
        self.on_epoch_end()
        if size is None:
            size = self.batch_size
        self.data = self.data.batch(size)
        # if self.augment:
        self.data = self.data.prefetch(buffer_size=AUTOTUNE)

    def on_epoch_end(self):
        """Shuffle data on epoch end
        """
        self.data = self.data.shuffle(buffer_size=1)


# path = '/nb_projects/AAA_ml/data/training/*.png'
# val_ds = DataSet(path, None, 1, 1, 0, 255, dim=(384, 384))
# # labeled_val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# val_ds.prepare_for_training(12)

# for img, gt in val_ds.data.take(4):
#     print(f"image shape: {img.shape}, dtype: {img.dtype}")
#     print(f"gt shape: {gt.shape}")
#     plt.subplot(1,2,1)
#     plt.imshow(img[0])
#     plt.subplot(1,2,2)
#     plt.imshow(gt[0])
#     plt.show()
