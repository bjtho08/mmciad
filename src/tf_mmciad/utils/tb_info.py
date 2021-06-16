import os

import io
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import Callback
# Depending on your keras version:-
#from tensorflow.keras.engine.training import GeneratorEnqueuer, Sequence, OrderedEnqueuer
from tensorflow.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer


def make_image_tensor(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Adapted from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    else:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)

def convert_palette_to_rgb(image, color_dict):
    colorvec = np.asarray([color_code for color_code in color_dict.values()])
    return colorvec[image]

class TensorboardWriter():

    def __init__(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        assert (os.path.isdir(outdir))
        self.outdir = outdir
        self.writer = tf.summary.create_file_writer(self.outdir,
                                            flush_millis=10000)

    def save_image(self, tag, image, global_step=None):
        with self.writer.as_default():
            tf.summary.image(tag, image, global_step)

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()


class ModelDiagnoser(Callback):

    def __init__(self,
                 data_generator,
                 batch_size,
                 num_samples,
                 output_dir,
                 color_dict,
                 normalization_mean,
                 normalization_std):
        self.batch_size = batch_size
        self.data_generator = data_generator
        self.color_dict = color_dict
        self.num_samples = num_samples
        self.tensorboard_writer = TensorboardWriter(output_dir + "/images")
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        is_sequence = isinstance(self.data_generator, Sequence)
        if is_sequence:
            self.enqueuer = OrderedEnqueuer(self.data_generator,
                                            use_multiprocessing=False,
                                            shuffle=False)
        else:
            self.enqueuer = GeneratorEnqueuer(self.data_generator,
                                              use_multiprocessing=False,
                                              wait_time=0.01)
        self.enqueuer.start(workers=4, max_queue_size=4)

    def on_epoch_end(self, epoch, logs=None):
        output_generator = self.enqueuer.get()
        steps_done = 0
        total_steps = int(np.ceil(np.divide(self.num_samples, self.batch_size)))
        sample_index = 0
        while steps_done < total_steps:
            generator_output = next(output_generator)
            x, y = generator_output[:2]
            y_pred = self.model.predict(x)
            y_pred = np.argmax(y_pred, axis=-1)
            y_true = np.argmax(y, axis=-1)

            for i in range(0, len(y_pred)):
                n = steps_done * self.batch_size + i
                if n >= self.num_samples:
                    return
                img = np.expand_dims(x[i, :, :, :],0)
                img = (img*self.normalization_std + self.normalization_mean)
                img = np.asarray(img, dtype="uint8")  # mean is the training images normalization mean
                #img = img[:, :, :, [2, 1, 0]]  # reordering of channels

                pred = y_pred[i]
                pred = pred.reshape(img.shape[0:3])
                print(pred.max(axis=(1,2)))
                pred = np.asarray(pred, dtype="uint8")
                print(pred.max(axis=(1,2)))
                pred = convert_palette_to_rgb(pred, self.color_dict)
                print(pred.max(axis=(1,2)))

                ground_truth = y_true[i]
                ground_truth = np.asarray(ground_truth.reshape(img.shape[0:3]), dtype="uint8")
                ground_truth = convert_palette_to_rgb(ground_truth, self.color_dict)


                self.tensorboard_writer.save_image("Epoch-{}/{}/x"
                                                   .format(epoch, sample_index), img, 1)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y"
                                                   .format(epoch, sample_index), ground_truth, 1)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y_pred"
                                                   .format(epoch, sample_index), pred, 1)
                sample_index += 1

            steps_done += 1

    def on_train_end(self, logs=None):
        self.enqueuer.stop()
        self.tensorboard_writer.close()