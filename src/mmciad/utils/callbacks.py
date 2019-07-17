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
    def __init__(self, params=None):
        """__init__

        :param params: model hyperparameters, defaults to None
        :type params: dict, optional
        """
        super(WriteLog, self).__init__()
        self.params = params or {}
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
            self.logger.info("Current parameters: %s" % self.params)

    def on_train_end(self, logs=None):
        self.logger.info("Training finished.\n\n")

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(
            "Epoch: {:5d}:\t".format(epoch + 1),
            "loss: {:7.4f}, acc: {:1.4f}, val_loss: {:7.4f}, val_acc: {:1.4f}".format(
                logs["loss"], logs["acc"], logs["val_loss"], logs["val_acc"])
        )
        new_lr = K.get_value(self.model.optimizer.lr)
        if self.old_lr > new_lr:
            logger.info(
                "Learning rate reduced from {:1.7f} to {:1.7f}".format(
                    self.old_lr, new_lr
                )
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
                'ModelCheckpoint mode %s is unknown, '
                'fallback to auto mode.' % (mode)
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
                        'Can save best model only with %s available, '
                        'skipping.' % (self.monitor)
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
                            '\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                            ' saving model to %s'
                            % (epoch + 1, self.monitor, self.best, current, filepath)
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
                            except Exception as error:
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
