"""Adds a few custom callbacks. Currently write_logs() and
PatchedModelCheckpoint()
"""
import warnings
import logging
from logging.handlers import RotatingFileHandler
from time import sleep

import numpy as np
from keras import backend as K
from keras.callbacks import Callback


class WriteLog(Callback):
    """Logging callback writes relevant training info to separate log"""
    def __init__(self, hyperparams=None):
        """__init__

        :param params: model hyperparameters, defaults to None
        :type params: dict, optional
        """
        super(WriteLog, self).__init__()
        self.hyperparams = hyperparams or {}
        self.logger = logging.getLogger("Training Log")
        self.logger.setLevel(logging.INFO)
        self.handler = RotatingFileHandler(
            "traininglog.txt", maxBytes=20000, backupCount=5
        )
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.old_lr = 0

    def on_train_begin(self, logs=None):
        self.logger.info("Training started")
        self.old_lr = K.get_value(self.model.optimizer.lr)
        if self.params:
            self.logger.info("Current parameters: %s", self.hyperparams)

    def on_train_end(self, logs=None):
        self.logger.info("Training finished.\n\n")

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(
            "Epoch: %5d:\tloss: %7.4f, acc: %1.4f, val_loss: %7.4f, val_acc: %1.4f",
            epoch + 1, logs["loss"], logs["acc"], logs["val_loss"], logs["val_acc"]
        )
        new_lr = K.get_value(self.model.optimizer.lr)
        if self.old_lr > new_lr:
            self.logger.info(
                "Learning rate reduced from %1.7f to %1.7f",
                self.old_lr, new_lr
            )
            self.old_lr = new_lr


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

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(PatchedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.logger = logging.getLogger("Training Log")

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(
                'ModelCheckpoint mode %s is unknown, '
                'fallback to auto mode.' % (mode),
                RuntimeWarning
            )
            self.logger.warning(
                'ModelCheckpoint mode %s is unknown, fallback to auto mode.',
                mode
            )
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
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
                        'Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning
                    )
                    self.logger.warning(
                        "Can save best model only with %s available, skipping.",
                        self.monitor
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                '\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                ' saving model to %s'
                                % (epoch + 1, self.monitor, self.best,
                                    current, filepath)
                            )
                        self.logger.info(
                            "\nEpoch %5d: %s improved from %0.5f to %0.5f, saving model to %s",
                            epoch + 1, self.monitor, self.best, current, filepath
                        )
                        self.best = current

                        saved_correctly = False
                        while not saved_correctly:
                            try:
                                if self.save_weights_only:
                                    self.model.save_weights(filepath, overwrite=True)
                                else:
                                    self.model.save(filepath, overwrite=True)
                                saved_correctly = True
                            except OSError as error:
                                print('Error while trying to save the model: {}.\nTrying again...'.format(error))
                                sleep(5)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                saved_correctly = False
                while not saved_correctly:
                    try:
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        saved_correctly = True
                    except Exception as error:
                        print('Error while trying to save the model: {}.\nTrying again...'.format(error))
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
        return layer.get_config().get('activation', None) == 'relu'

    def get_relu_activations(self):
        model_input = self.model.input
        is_multi_input = isinstance(model_input, list)
        if not is_multi_input:
            model_input = [model_input]

        funcs = {}
        for index, layer in enumerate(self.model.layers):
            if not layer.get_weights():
                continue
            funcs[index] = K.function(model_input
                                      + [K.learning_phase()], [layer.output])

        if is_multi_input:
            list_inputs = []
            list_inputs.extend(self.x_train)
            list_inputs.append(1.)
        else:
            list_inputs = [self.x_train, 1.]

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
                if type(layer_weight) is not list:
                    raise ValueError("'Layer_weight' should be a list, "
                                     "but was {}".format(type(layer_weight)))

                # there are no weights for current layer; skip it
                # this is only legitimate if layer is "Activation"
                if len(layer_weight) == 0:
                    continue

                layer_weight_shape = np.shape(layer_weight[0])
                yield [layer_index,
                       layer_activations,
                       layer_name,
                       layer_weight_shape]


    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.x_train, "batch_size"):
            dim = self.x_train.dim
            batches = len(self.x_train)
            total = batches * self.x_train.batch_size
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
            if K.image_data_format() == 'channels_last':
                # features in last axis
                axis_filter = -1
            else:
                # features before the convolution axis, for weight_
                # len the input and output have to be subtracted
                axis_filter = -1 - (weight_len - 2)

            total_featuremaps = shape_act[axis_filter]

            axis = []
            for i in range(act_len):
                if (i != axis_filter) and (i != (len(shape_act) + axis_filter)):
                    axis.append(i)
            axis = tuple(axis)

            dead_neurons = np.sum(np.sum(activation_values, axis=axis) == 0)

            dead_neurons_share = float(dead_neurons) / float(total_featuremaps)
            if ((self.verbose and dead_neurons > 0)
                    or dead_neurons_share >= self.dead_neurons_share_threshold):
                str_warning = ('Layer {} (#{}) has {} '
                               'dead neurons ({:.2%})!').format(layer_name,
                                                                layer_index,
                                                                dead_neurons,
                                                                dead_neurons_share)
                print(str_warning)