"""Adds a few custom callbacks. Currently write_logs() and
PatchedModelCheckpoint()
"""
import warnings
import os
import logging
from time import sleep

import numpy as np
from twilio.rest import Client

from tensorflow.keras.callbacks import Callback, TensorBoard
import tensorflow.keras.backend as K


class LossAndAccTextingCallback(Callback):
    def __init__(self) -> None:
        account_sid = "AC53419d4da859fe95426b41af641cda11"
        auth_token = "7e45a775d99632d25e890afc78c6a177"
        self.client = Client(account_sid, auth_token)

    def on_epoch_end(self, epoch, logs=None):
        message = self.client.messages.create(
            body=f"Loss for epoch {epoch} is {logs['loss']:2.4f} and acc is {logs['acc']:1:4f}",  # Message you send
            from_="+17034207403",  # Provided phone number
            to="+4528129716",  # Your phone number
        )
        print(message.sid)


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = len(self.batch_gen)   # Number of times to call next() on the generator.
        self.batch_size = self.batch_gen.batch_size

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = self.batch_gen[s]
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * self.batch_size,) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * self.batch_size,) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)


# ...

# callbacks = [TensorBoardWrapper(gen_val, nb_steps=5, log_dir=self.cfg['cpdir'], histogram_freq=1,
#                                batch_size=32, write_graph=False, write_grads=True)]
# ...


class MemoryCallback(Callback):
    def __init__(self):
        self.current_mem_usage = 1
        self.prev_mem_usage = 1
        
    def on_epoch_end(self, epoch, log={}):
        self.prev_mem_usage = self.current_mem_usage
        self.current_mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000 # func returns kilobytes
        increase = (self.current_mem_usage / self.prev_mem_usage) * 100
        if epoch == 0:
            print(f"Total memory usage: {self.convert_bytes(self.current_mem_usage)}")
        else:
            print(f"Total memory usage: {self.convert_bytes(self.current_mem_usage)} ({increase-100} % increase)")
        
    def convert_bytes(self, num):
        """
        this function will convert bytes to MB.... GB... etc
        """
        step_unit = 1024.0 #1024 bad the size

        for x in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']:
            if num < step_unit:
                return "%3.1f %s" % (num, x)
            num /= step_unit


class ValidationHook(Callback):
    def __init__(self, validation_generator):
        super(ValidationHook, self).__init__()
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        self.validation_generator.on_epoch_end()
        # print("Called 'on_epoch_end()' for 'validation_generator'")


class PatchedModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        period=1,
    ):
        super(PatchedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.previous_filepath = None
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.logger = logging.getLogger("Training Log")

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                "ModelCheckpoint mode %s is unknown, fallback to auto mode." % (mode),
                RuntimeWarning,
            )
            self.logger.warning(
                "ModelCheckpoint mode %s is unknown, fallback to auto mode.", mode
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        "Can save best model only with %s available, "
                        "skipping." % (self.monitor),
                        RuntimeWarning,
                    )
                    self.logger.warning(
                        "Can save best model only with %s available, skipping.",
                        self.monitor,
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                        self.logger.info(
                            "\nEpoch %5d: %s improved from %0.5f to %0.5f, saving ...",
                            epoch + 1,
                            self.monitor,
                            self.best,
                            current,
                        )
                        self.best = current

                        saved_correctly = False
                        if self.previous_filepath and os.path.exists(
                            self.previous_filepath
                        ):
                            os.remove(self.previous_filepath)
                        while not saved_correctly:
                            try:
                                if self.save_weights_only:
                                    self.model.save_weights(filepath, overwrite=True)
                                else:
                                    self.model.save(filepath, overwrite=True)
                                saved_correctly = True
                                self.previous_filepath = filepath
                            except OSError as error:
                                print(
                                    f"Error while trying to save the model: {error}."
                                    + "\nTrying again..."
                                )
                                sleep(5)
                    else:
                        if self.verbose > 0:
                            print(
                                "\nEpoch %05d: %s did not improve from %0.5f"
                                % (epoch + 1, self.monitor, self.best)
                            )
            else:
                if self.verbose > 0:
                    print("\nEpoch %05d: saving model to %s" % (epoch + 1, filepath))
                saved_correctly = False
                if self.previous_filepath and os.path.exists(self.previous_filepath):
                    os.remove(self.previous_filepath)
                while not saved_correctly:
                    try:
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        saved_correctly = True
                        self.previous_filepath = filepath
                    except OSError as error:
                        print(
                            f"Error while trying to save the model: {error}."
                            + "\nTrying again..."
                        )
                        sleep(5)


class DeadReluDetector(Callback):
    """Reports the number of dead ReLUs after each training epoch
    ReLU is considered to be dead if it did not fire once for entire training set

    # Arguments
        x_train: Training dataset to check whether or not neurons fire
        verbose: verbosity mode
            True means that even a single dead neuron triggers a warning message
            False means that only significant number of dead neurons (10% or more)
            triggers a warning message
    """

    def __init__(self, x_train=None, verbose=False):
        super(DeadReluDetector, self).__init__()
        self.x_train = x_train
        self.verbose = verbose
        self.dead_neurons_share_threshold = 0.1

    @staticmethod
    def is_relu_layer(layer):
        # Should work for all layers with relu
        # activation. Tested for Dense and Conv2D
        return layer.get_config().get("activation", None) == "relu"

    def get_relu_activations(self):
        model_input = self.model.input
        is_multi_input = isinstance(model_input, list)
        if not is_multi_input:
            model_input = [model_input]

        funcs = {}
        for index, layer in enumerate(self.model.layers):
            if not layer.get_weights():
                continue
            funcs[index] = K.function(
                model_input + [K.learning_phase()], [layer.output]
            )

        if is_multi_input:
            list_inputs = []
            list_inputs.extend(self.x_train)
            list_inputs.append(1.0)
        else:
            list_inputs = [self.x_train, 1.0]

        layer_outputs = {}
        for index, func in funcs.items():
            layer_outputs[index] = func(list_inputs)[0]

        for layer_index, layer_activations in layer_outputs.items():
            if self.is_relu_layer(self.model.layers[layer_index]):
                layer_name = self.model.layers[layer_index].name
                # layer_weight is a list [W] (+ [b])
                layer_weight = self.model.layers[layer_index].get_weights()

                # with kernel and bias, the weights are saved as a list [W, b].
                # If only weights, it is [W]
                if not isinstance(layer_weight, list):
                    raise ValueError(
                        "'Layer_weight' should be a list, "
                        "but was {}".format(type(layer_weight))
                    )

                # there are no weights for current layer; skip it
                # this is only legitimate if layer is "Activation"
                if len(layer_weight) == 0:
                    continue

                layer_weight_shape = np.shape(layer_weight[0])
                yield [layer_index, layer_activations, layer_name, layer_weight_shape]

    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.x_train, "batch_size"):
            batches = len(self.x_train)
            train = []
            for batch in range(batches):
                x, _ = self.x_train[batch]
                train.extend(x)
            self.x_train = np.asarray(train)

        for relu_activation in self.get_relu_activations():
            layer_index = relu_activation[0]
            activation_values = relu_activation[1]
            layer_name = relu_activation[2]
            layer_weight_shape = relu_activation[3]

            shape_act = activation_values.shape

            weight_len = len(layer_weight_shape)
            act_len = len(shape_act)

            # should work for both Conv and Flat
            if K.image_data_format() == "channels_last":
                # features in last axis
                axis_filter = -1
            else:
                # features before the convolution axis, for weight_
                # len the input and output have to be subtracted
                axis_filter = -1 - (weight_len - 2)

            total_featuremaps = shape_act[axis_filter]

            axis = []
            for i in range(act_len):
                if i not in (axis_filter, len(shape_act) + axis_filter):
                    axis.append(i)
            axis = tuple(axis)

            dead_neurons = np.sum(np.sum(activation_values, axis=axis) == 0)

            dead_neurons_share = float(dead_neurons) / float(total_featuremaps)
            if (
                self.verbose and dead_neurons > 0
            ) or dead_neurons_share >= self.dead_neurons_share_threshold:
                str_warning = (
                    "Layer {} (#{}) has {} " "dead neurons ({:.2%})!"
                ).format(layer_name, layer_index, dead_neurons, dead_neurons_share)
                print(str_warning)
