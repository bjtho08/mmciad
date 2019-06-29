import numpy as np
import pickle
import warnings
from time import sleep
from keras.callbacks import Callback

MODEL_SAVE_NAME = 0
MODEL_SAVE_WEIGHTS = 1

def save_weight_light(model, filename):
	layers = model.layers
	pickle_list = []
	for layerId in range(len(layers)):
		weigths = layers[layerId].get_weights()
		pickle_list.append([layers[layerId].name, weigths])

	with open(filename, 'wb') as f:
		pickle.dump(pickle_list, f, -1)

def load_weight_light(model, filename):
    layers = model.layers
    with open(filename, 'rb') as f:
        pickle_list = pickle.load(f)

    for layerId in range(len(layers)):
        if "gaussian_noise" in layers[layerId].name:
            layers[layerId].name = "GaussianNoise_preout"
        if "gaussian_noise" in pickle_list[layerId][MODEL_SAVE_NAME]:
            pickle_list[layerId][MODEL_SAVE_NAME] = "GaussianNoise_preout"
        assert(layers[layerId].name == pickle_list[layerId][MODEL_SAVE_NAME]),"{} not equal to {}".format(layers[layerId].name, pickle_list[layerId][MODEL_SAVE_NAME])
        layers[layerId].set_weights(pickle_list[layerId][MODEL_SAVE_WEIGHTS])
            

class ModelCheckpointLight(Callback):
	def __init__(self, filepath, monitor='val_loss', verbose=0,
				 save_best_only=False,
				 mode='auto', period=1):
		super(ModelCheckpointLight, self).__init__()
		self.monitor = monitor
		self.verbose = verbose
		self.filepath = filepath
		self.save_best_only = save_best_only
		self.period = period
		self.epochs_since_last_save = 0

		if mode not in ['auto', 'min', 'max']:
			warnings.warn('ModelCheckpointLight mode %s is unknown, '
						  'fallback to auto mode.' % (mode),
						  RuntimeWarning)
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
					warnings.warn('Can save best model only with %s available, '
								  'skipping.' % (self.monitor), RuntimeWarning)
				else:
					if self.monitor_op(current, self.best):
						if self.verbose > 0:
							print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
								  ' saving model to %s'
								  % (epoch + 1, self.monitor, self.best,
									 current, filepath))
						self.best = current
						save_weight_light(self.model, filepath)
					else:
						if self.verbose > 0:
							print('\nEpoch %05d: %s did not improve from %0.5f' %
								  (epoch + 1, self.monitor, self.best))
			else:
				if self.verbose > 0:
					print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
				save_weight_light(self.model, filepath)


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

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
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
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
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
