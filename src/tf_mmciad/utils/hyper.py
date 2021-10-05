"""Initialize a talos model object for hyper-parameter search

"""
import datetime
import os
import os.path as osp
from collections import OrderedDict
from pathlib import Path
from glob import glob
from typing import Any, Dict, List, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# from keras_radam import RAdam
from tensorflow.keras.activations import selu, sigmoid, tanh
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    LambdaCallback,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    ELU,
    LeakyReLU,
    PReLU,
    ReLU,
    Softmax,
    ThresholdedReLU,
)
from tensorflow.keras.losses import (
    binary_crossentropy,
    categorical_crossentropy,
    mae,
    mse,
)

# from keras_contrib.callbacks import DeadReluDetector
from tensorflow.keras.optimizers import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    Nadam,
    RMSprop,
)
from tensorflow.python import keras
from tf_mmciad.utils.callbacks import (  # DeadReluDetector
    LossAndAccTextingCallback,
    PatchedModelCheckpoint,
    TensorBoardWrapper,
)
from tf_mmciad.utils.custom_loss import (
    categorical_focal_loss,
    jaccard1_coef,
    jaccard1_loss,
    jaccard2_loss,
    tversky_loss,
    weighted_loss,
)
from tf_mmciad.utils.f_scores import F1Score
from tf_mmciad.utils.generator import DataGenerator
from tf_mmciad.utils.swish import Swish
from tf_mmciad.utils.u_net import u_net

# from keras_tqdm import TQDMNotebookCallback
# import talos

IMG_ROWS, IMG_COLS, IMG_CHANNELS = (None, None, 3)
# architecture params
NB_FILTERS_0 = 64

# ****  deep learning model
SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
BATCH_SIZE = 16
NB_EPOCH = 200
VERBOSE = 0

# **** DEBUGGING
CRASH_COUNT = 0

# ****  train


def get_string(name_or_str: Union[str, object]):
    """Return a string describing the input.
    If input is a string, then input is returned.
    Otherwise, get the object name, if this exists,
    or return the object string representation
    """
    if not isinstance(name_or_str, str):
        output = getattr(name_or_str, "__name__", str(name_or_str))
    else:
        output = name_or_str
    return output


def value_as_string(input_dict: dict):
    """Return a dictionary where all values are converted into strings
    """
    output_dict = dict()
    for key, val in input_dict.items():
        output_dict[key] = get_string(val)
    return output_dict


def get_loss_function(name: str, cls_wgts=None):
    """ Retrieve the requested loss function

    Available functions:
    --------------------
    mse, mae, binary_crossentropy, jaccard1_loss,
    jaccard2_loss, categorical_crossentropy,
    tversky_loss, categorical_focal_loss
    """
    func_dict = {
        "mse": mse,
        "mae": mae,
        "binary_CE": binary_crossentropy,
        "cat_CE": categorical_crossentropy,
        "cat_FL": categorical_focal_loss(),
        "tversky_loss": tversky_loss,
        "w_cat_CE": weighted_loss(categorical_crossentropy, cls_wgts),
        "w_cat_FL": weighted_loss(categorical_focal_loss, cls_wgts),
        "w_TL": weighted_loss(tversky_loss, cls_wgts),
        "jaccard1_loss": jaccard1_loss,
        "jaccard2_loss": jaccard2_loss,
    }
    loss_func = func_dict.get(name, None)
    if loss_func is None:
        try:
            loss_func = globals()[name]
        except KeyError as error:
            raise NameError(f"Wrong loss function name ({name})") from error

    loss_func.__name__ = name
    return loss_func


def get_act_function(name: str):
    """ Retrieve the requested activation function

    Available functions:
    --------------------
    swish, relu, elu, leaky_relu, prelu, softmax,
    thresholded_relu, tanh, selu, sigmoid
    """
    func_dict = {
        "swish": Swish,
        "relu": ReLU,
        "elu": ELU,
        "leaky_relu": LeakyReLU,
        "prelu": PReLU,
        "softmax": Softmax,
        "thresholded_relu": ThresholdedReLU,
        "tanh": tanh,
        "selu": selu,
        "sigmoid": sigmoid,
    }
    act_func = func_dict.get(name, None)
    if act_func is None:
        act_func = ReLU
        raise Warning("Uknown activation, falling back to ReLU")
    return act_func


def get_opt_function(name: str):
    """ Retrieve the requested optimizer

    Available functions:
    --------------------
    adam, radam, nadam, sgd, rmsprop, adadelta,
    adagrad, adamax
    """
    func_dict = {
        "adam": Adam,
        # "radam": RAdam,
        "nadam": Nadam,
        "sgd": SGD,
        "rmsprop": RMSprop,
        "adadelta": Adadelta,
        "adagrad": Adagrad,
        "adamax": Adamax,
    }
    opt_func = func_dict.get(name, None)
    if opt_func is None:
        opt_func = Adam
        raise Warning("Uknown optimizer, falling back to Adam")
    return opt_func


class LogSegmentationProgress(Callback):
    """Simple image writer class"""

    def __init__(self, file_writer_cm, tensorboard_params: dict):
        super().__init__()
        self.file_writer = file_writer_cm
        self.tb_params = tensorboard_params
        self.path = self.tb_params.pop("path")
        self.color_dict = self.tb_params.pop("color_dict")
        self.color_list = self.tb_params.pop("color_list")
        self.means = self.tb_params.pop("means")
        self.stds = self.tb_params.pop("stds")
        self.x_min = self.tb_params.pop("x_min")
        self.x_max = self.tb_params.pop("x_max")
        self.tb_args = [
            self.path,
            self.color_dict,
            self.means,
            self.stds,
            self.x_min,
            self.x_max,
        ]
        self.color_map = self.make_color_map(self.color_dict)

    def on_epoch_end(self, epoch, logs=None):
        _ = logs

        test_generator = DataGenerator(
            *self.tb_args, **self.tb_params, shuffle=False, augment=False,
        )
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(test_generator)
        test_pred = np.argmax(test_pred_raw, axis=-1)
        # Read the input and target images
        norm_input, targets = test_generator[0]
        # revert normalized input to raw RBG
        raw_input = (norm_input * self.stds) + self.means
        raw_input = np.round(raw_input).astype(np.uint8)
        # recreate color matrix
        # palette = np.array(list(self.color_dict.values()), dtype="uint8",)
        # convert one-hot encoded matrices to RBG
        cat_pred = self.color_list[test_pred].astype("uint8")
        cat_targets = self.color_list[np.argmax(targets, axis=-1)].astype("uint8")
        # cat_pred = palette[test_pred]
        #cat_targets = palette[np.argmax(targets, axis=-1)]
        # Log the image summaries.
        with self.file_writer.as_default():
            tf.summary.image("Raw input", raw_input, max_outputs=8, step=epoch)
            tf.summary.image("Ground truth", cat_targets, max_outputs=8, step=epoch)
            tf.summary.image("Prediction", cat_pred, max_outputs=8, step=epoch)

    def set_model(self, model):
        self.model = model

    def make_color_map(self, colors: Dict[int, List[int]]):
        """Create a new RGB map from a dictionary of color values.

        Args:
            colors (Dict[int, List[int]]): [description]

        Returns:
            [type]: [description]
        """
        color_map = np.ndarray(shape=(256 * 256 * 256), dtype="int32")
        color_map[:] = -1
        for idx, rgb_list in colors.items():
            rgb = rgb_list[0] * 65536 + rgb_list[1] * 256 + rgb_list[2]
            color_map[rgb] = idx
        return color_map


def single_run(
    weight_path,
    params,
    train_generator,
    val_generator,
    tensorboard_params=None,
    cls_wgts=None,
    notebook=True,
    debug=False,
    resume=False,
):
    """Run a single instance of training using the static parameters in the config file

    Args:
        weight_path (str): Path to the base weight folder
        params (Dict): Dictionary of parameters in the model
        train_generator (Class): Generator function for training data
        val_generator (Class): Generator function for validation data
        cls_wgts (None, or List of floats, optional): A list containing the weights
            applied to each class. Defaults to None.
        notebook (bool, optional): Whether function is executed inside a notebook.
            Defaults to True.
        debug (bool, optional): Enable debugging (partially implemented).
            Defaults to False.

    Returns:
        (tensorflow.keras.Model, Dict): compiled model and history object of the
            finished training
    """
    model_base_path = osp.join(weight_path, params["date"])

    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path, exist_ok=True)
    param_dict = OrderedDict()
    param_dict.update(params)
    loss_func = get_loss_function(param_dict["loss_func"])
    act_func = get_act_function(param_dict["act"])
    opt_func = get_opt_function(param_dict["opt"])

    if debug:
        param_dict["steps_per_epoch"] = 2
        param_dict["nb_epoch"] = 2
        global CRASH_COUNT  # pylint: disable=global-statement
        CRASH_COUNT += 1
        if CRASH_COUNT == 3:
            raise KeyboardInterrupt()

    if param_dict["class_weights"] is False and cls_wgts is None:
        class_weights = None
    elif param_dict["class_weights"] is False:
        class_weights = {i: 1 if k != 12 else 0 for i, k in enumerate(cls_wgts.keys())}
    else:
        class_weights = cls_wgts

    if str(param_dict["arch"]).lower() == "u-resnet":
        param_dict["maxpool"] = False
        # p["pretrain"] = 0
        # p["nb_filters_0"] = 32
        # p["depth"] = 3

    model_base_path = osp.join(weight_path, param_dict["date"])
    tensorboard_base_path = osp.join("logs", param_dict["date"])

    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path, exist_ok=True)
    basename = "model"

    suffix = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    model_handle = "_".join([basename, suffix])  # e.g. 'mylogfile_120508_171442'
    modelpath = osp.join(model_base_path, model_handle)
    tb_path = osp.join(tensorboard_base_path, model_handle)
    with open(file=modelpath + ".cfg", mode="w") as f:  # pylint: disable=invalid-name
        f.write("#" * 62 + "\n#" + " " * 60 + "#\n")
        f.write("#" + f"Model Paramters for {model_handle}_*.h5".center(60) + "#\n")
        f.write("#" + " " * 60 + "#\n" + "#" * 62 + "\n\n")
        f.write("Parameters:\n")
        for key, val in params.items():
            f.write(f"  {key} = {get_string(val)}\n")
        now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")
        f.write(f"\nTraining started: {now}\n")

    model_kwargs = {
        "shape": param_dict["shape"],
        "nb_filters": int(param_dict["nb_filters_0"]),
        "sigma_noise": param_dict["sigma_noise"],
        "depth": param_dict["depth"],
        "maxpool": param_dict["maxpool"],
        "initialization": param_dict["init"],
        "activation": act_func,
        "dropout": param_dict["dropout"],
        "output_channels": param_dict["num_cls"],
        "batchnorm": param_dict["batchnorm"],
        "pretrain": param_dict["pretrain"],
        "arch": param_dict["arch"],
    }
    csv_logger = CSVLogger(modelpath + ".fit.csv", append=True)

    tb_callback = TensorBoardWrapper(
        val_generator,
        log_dir=tb_path,
        histogram_freq=1,
        write_graph=True,
        embeddings_freq=0,
        update_freq="epoch",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0001, patience=6, verbose=0, mode="auto"
    )
    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=3, min_lr=1e-8, verbose=1
    )
    jaccard_model_checkpoint = PatchedModelCheckpoint(
        modelpath + "_epoch_{epoch}_val_jacc1_{val_jaccard1_coef:0.4f}.h5",
        verbose=0,
        monitor="val_jaccard1_coef",
        save_best_only=True,
    )
    loss_model_checkpoint = PatchedModelCheckpoint(
        modelpath + "_epoch_{epoch}_val_loss_{val_loss:0.4f}.h5",
        verbose=0,
        monitor="val_loss",
        save_best_only=True,
    )
    acc_model_checkpoint = PatchedModelCheckpoint(
        modelpath + "_epoch_{epoch}_acc_{acc:0.4f}.h5",
        verbose=0,
        monitor="acc",
        save_best_only=True,
    )

    f1_average = F1Score(num_classes=param_dict["num_cls"], average="weighted")
    f1_per_class = [
        F1Score(num_classes=param_dict["num_cls"], average="report", focus=i)
        for i in range(param_dict["num_cls"])
    ]

    try:
        latest_model = glob(modelpath + "_epoch_*_acc_*.h5")[0]
    except IndexError:
        resume = False
    if resume:
        cat_fl = categorical_focal_loss()
        custom_objs = {
            "Swish": Swish,
            "F1Score": F1Score,
            "cat_CE": categorical_crossentropy,
            "tversky_loss": tversky_loss,
            "categorical_focal_loss_fixed": cat_fl,
            "jaccard1_coef": jaccard1_coef,
            "cat_FL": cat_fl,
        }
        current_epoch = latest_model.rsplit("_")[4] + 1
        model = keras.models.load_model(latest_model, custom_objects=custom_objs)
    else:
        model = u_net(**model_kwargs)
        current_epoch = 0

        if param_dict.get("pretrain", 0):
            pretrain_model = (
                "imagenet_pretrain"
                if model_kwargs["nb_filters"] == 64
                else "imagenet32_pretrain"
            )
            pretrain_path = sorted(Path(weight_path, pretrain_model).glob("*.h5"))[-1]
            model.load_weights(pretrain_path, by_name=True, skip_mismatch=True)
            pretrain_layers = [
                "block{}_d_conv{}".format(block, layer)
                for block in range(1, param_dict["depth"] + 1)
                for layer in range(1, 3)
            ]
            bn_layers = [
                "block{}_d_bn{}".format(block, layer)
                for block in range(1, param_dict["depth"] + 1)
                for layer in range(1, 3)
            ]
            for n in pretrain_layers:
                model.get_layer(name=n).trainable = False
            for n in bn_layers:
                model.get_layer(name=n).trainable = False

        model.compile(
            loss=loss_func,
            optimizer=opt_func(param_dict["lr"]),
            metrics=["acc", f1_average, *f1_per_class, jaccard1_coef],
        )

    file_writer_seg = tf.summary.create_file_writer(tb_path + "/images")

    log_image_segmentation = LogSegmentationProgress(
        file_writer_seg, tensorboard_params
    )

    # tb_image_cb = LambdaCallback(on_epoch_end=log_image_segmentation)

    model_callbacks = [
        csv_logger,
        tb_callback,
        log_image_segmentation,
    ]  # , gamify_callback]
    if notebook:
        tqdm_progress = tfa.callbacks.TQDMProgressBar()
        model_callbacks.append(tqdm_progress)
    opti_callbacks = [
        early_stopping,
        reduce_lr_on_plateau,
        loss_model_checkpoint,
        jaccard_model_checkpoint,
        acc_model_checkpoint,
    ]

    history = model.fit(
        x=train_generator,
        epochs=param_dict["nb_epoch"],
        initial_epoch=current_epoch,
        validation_data=val_generator,
        workers=30,
        steps_per_epoch=param_dict.get("steps_per_epoch", None),
        validation_steps=param_dict.get("steps_per_epoch", None),
        class_weight=class_weights,
        verbose=param_dict["verbose"],
        callbacks=model_callbacks + opti_callbacks,
    )
    return model, history


class TalosModel:
    """Initialize a talos model object for hyper-parameter search

    :param weight_path: Path to the base weight folder
    :type weight_path: str
    :param cls_wgts: A list containing the weights applied to each class,
    or None
    :type cls_wgts: None, or List of floats
    :param params: Dictionary of fixed parameters in the model
    :type params: Dict
    :param train_generator: Generator function for training data
    :type train_generator: Class
    :param train_generator: Generator function for validation data
    :type train_generator: Class
    """

    grid_run = True

    def __init__(
        self,
        weight_path,
        static_params,
        train_generator,
        val_generator,
        tensorboard_params=None,
        cls_wgts=None,
        notebook=True,
        debug=False,
    ):

        self.static_params = static_params
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.tensorboard_params = tensorboard_params
        self.cls_wgts = cls_wgts
        self.notebook = notebook
        self.debug = debug
        self.model_count = 0
        self.crash_count = 0
        self.model_base_path = osp.join(weight_path, self.static_params["date"])
        self.tensorboard_base_path = osp.join("logs", self.static_params["date"])

        if not os.path.exists(self.model_base_path):
            os.makedirs(self.model_base_path, exist_ok=True)
        else:
            self.update_model_count()
        self.basename = "model"

        self.suffix = str(self.model_count)
        # Can be replaced with
        # datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
        self.model_handle = "_".join([self.basename, self.suffix])  # e.g. 'model_0'
        self.modelpath = osp.join(self.model_base_path, self.model_handle)

    @classmethod
    def set_grid_run(cls, flag: bool = True):
        """Change the class to expect running a grid search (currently the default)
        """
        cls.grid_run = flag

    def update_model_count(self):
        """ Update the model number for file naming purposes
        """
        cfg_count = len(
            [
                file
                for file in os.listdir(self.model_base_path)
                if str(file).endswith(".cfg")
            ]
        )
        self.model_count = max(cfg_count - 1, 0)

    def __call__(self, *args):
        """Talos model setup

        :param x: Dummy input needed for talos framework
        :type x: Array-like
        :param y: Dummy input needed for talos framework
        :type y: Array-like
        :param val_x: Dummy input needed for talos framework
        :type val_x: Array-like
        :param val_y: Dummy input needed for talos framework
        :type val_y: Array-like
        :param params: Hyperparameters supplied by talos
        :type params: Dict
        """
        # Dummy inputs
        *_, last = args
        param_dict = OrderedDict()
        param_dict.update(self.static_params)
        if isinstance(last, dict):
            talos_params = last
            self.write_config(talos_params)
            param_dict.update(talos_params)
        elif isinstance(last, int):
            self.write_config()
            # model_exp_name = last # What was I thinking?
        if not isinstance(last, dict) and self.grid_run:
            raise TypeError(f"Expected a dict of parameters, got {type(last)}")
        loss_func = get_loss_function(param_dict["loss_func"])
        act_func = get_act_function(param_dict["act"])
        opt_func = get_opt_function(param_dict["opt"])

        if self.debug:
            param_dict["steps_per_epoch"] = 2
            param_dict["nb_epoch"] = 2  # pylint: disable=global-statement
            self.crash_count += 1
            if self.crash_count == 3:
                raise KeyboardInterrupt()

        if param_dict["class_weights"] is False and self.cls_wgts is None:
            class_weights = None
        elif param_dict["class_weights"] is False:
            class_weights = [1 if k != 12 else 0 for k in self.cls_wgts.keys()]
        else:
            class_weights = list(self.cls_wgts.values())

        if str(param_dict["arch"]).lower() == "u-resnet":
            param_dict["maxpool"] = False
            # p["pretrain"] = 0
            # p["nb_filters_0"] = 32
            # p["depth"] = 3

        self.update_model_count()

        self.suffix = str(self.model_count)
        self.model_handle = "_".join([self.basename, self.suffix])  # e.g. 'model_0'
        self.modelpath = osp.join(self.model_base_path, self.model_handle)
        tb_path = osp.join(self.tensorboard_base_path, self.model_handle)

        model_kwargs = {
            "shape": param_dict["shape"],
            "nb_filters": int(param_dict["nb_filters_0"]),
            "sigma_noise": param_dict["sigma_noise"],
            "depth": param_dict["depth"],
            "maxpool": param_dict["maxpool"],
            "initialization": param_dict["init"],
            "activation": act_func,
            "dropout": param_dict["dropout"],
            "output_channels": param_dict["num_cls"],
            "batchnorm": param_dict["batchnorm"],
            "pretrain": param_dict["pretrain"],
            "arch": param_dict["arch"],
        }

        f1_per_class = [
            F1Score(num_classes=param_dict["num_cls"], average="report", focus=i)
            for i in range(param_dict["num_cls"])
        ]
        csv_logger = CSVLogger(self.modelpath + ".fit.csv", append=True)
        if self.notebook:
            tqdm_progress = tfa.callbacks.TQDMProgressBar()
        tb_callback = TensorBoardWrapper(
            self.val_generator,
            log_dir=tb_path,
            histogram_freq=1,
            write_graph=True,
            embeddings_freq=0,
            update_freq="epoch",
        )
        early_stopping = EarlyStopping(
            monitor="loss", min_delta=0.0001, patience=15, verbose=0, mode="auto"
        )
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor="loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1
        )
        jaccard_model_checkpoint = PatchedModelCheckpoint(
            self.modelpath + "_epoch_{epoch}_val_jacc1_{val_jaccard1_coef:0.4f}.h5",
            verbose=0,
            monitor="val_jaccard1_coef",
            save_best_only=True,
        )
        loss_model_checkpoint = PatchedModelCheckpoint(
            self.modelpath + "_epoch_{epoch}_val_loss_{val_loss:0.4f}.h5",
            verbose=0,
            monitor="val_loss",
            save_best_only=True,
        )

        acc_model_checkpoint = PatchedModelCheckpoint(
            self.modelpath + "_epoch_{epoch}_acc_{acc:0.4f}.h5",
            verbose=0,
            monitor="acc",
            save_best_only=True,
        )

        # talos_path = osp.join("talos", p["date"])
        # gamify_callback = talos.utils.ExperimentLogCallback(talos_path, talos_params)
        # dead_relus = DeadReluDetector(x_train=train_generator)

        model_callbacks = [
            csv_logger,
            tb_callback,
            LossAndAccTextingCallback(),
        ]  # , gamify_callback]
        if self.notebook:
            model_callbacks.append(tqdm_progress)
        opti_callbacks = [
            early_stopping,
            reduce_lr_on_plateau,
            loss_model_checkpoint,
            jaccard_model_checkpoint,
            acc_model_checkpoint,
        ]

        if param_dict["pretrain"] != 0:
            print(
                "starting with frozen layers\nclass weights: {}".format(class_weights)
            )
            model = u_net(**model_kwargs)

            file_writer_seg = tf.summary.create_file_writer(tb_path + "/images")

            log_image_segmentation = LogSegmentationProgress(
                file_writer_seg, self.tensorboard_params
            )

            tb_image_cb = LambdaCallback(on_epoch_end=log_image_segmentation)

            model_callbacks = [
                csv_logger,
                tb_callback,
                file_writer_seg,
                tb_image_cb,
            ]  # , gamify_callback]
            compile_kwargs = {
                "loss": loss_func,
                "optimizer": opt_func(param_dict["lr"]),
                "metrics": ["acc", f1_per_class, jaccard1_coef],
            }

            fit_kwargs = {
                "x": self.train_generator,
                "epochs": param_dict["nb_frozen"],
                "validation_data": self.val_generator,
                "use_multiprocessing": True,
                "workers": 30,
                "class_weight": class_weights,
                "verbose": param_dict["verbose"],
                "callbacks": model_callbacks,
            }

            model, history = self.fit(model, compile_kwargs, fit_kwargs)

            pretrain_layers = [
                "block{}_d_conv{}".format(block, layer)
                for block in range(1, param_dict["pretrain"] + 1)
                for layer in range(1, 3)
            ]
            for n in pretrain_layers:
                model.get_layer(name=n).trainable = True
            print("layers unfrozen\n")

            fit_kwargs.update(
                {
                    "epochs": param_dict["nb_epoch"],
                    "initial_epoch": param_dict["nb_frozen"],
                    "callbacks": model_callbacks + opti_callbacks,
                }
            )

            model, history = self.fit(model, compile_kwargs, fit_kwargs)
        else:
            # print(
            #   "No layers frozen at start\nclass weights: {}".format(class_weights)
            # )
            compile_kwargs = {
                "loss": loss_func,
                "optimizer": opt_func(param_dict["lr"]),
                "metrics": ["acc", jaccard1_coef],
            }

            fit_kwargs = {
                "x": self.train_generator,
                "epochs": param_dict["nb_epoch"],
                "validation_data": self.val_generator,
                "workers": 30,
                "steps_per_epoch": param_dict.get("steps_per_epoch", None),
                "validation_steps": param_dict.get("steps_per_epoch", None),
                "class_weight": class_weights,
                "verbose": param_dict["verbose"],
                "callbacks": model_callbacks + opti_callbacks,
            }
            model = u_net(**model_kwargs)

            file_writer_seg = tf.summary.create_file_writer(tb_path + "/images")

            log_image_segmentation = LogSegmentationProgress(
                file_writer_seg, self.tensorboard_params
            )

            tb_image_cb = LambdaCallback(on_epoch_end=log_image_segmentation)

            model_callbacks = [
                csv_logger,
                tb_callback,
                file_writer_seg,
                tb_image_cb,
            ]  # , gamify_callback]
            model, history = self.fit(model, compile_kwargs, fit_kwargs)
            self.model_count += 1
        return history, model

    def write_config(self, talos_params=None):
        """ Write a short file detailing the applied parameters
        """
        with open(
            file=self.modelpath + ".cfg", mode="w"
        ) as f:  # pylint: disable=invalid-name
            f.write("#" * 62 + "\n#" + " " * 60 + "#\n")
            f.write(
                "#" + f"Model Paramters for {self.model_handle}_*.h5".center(60) + "#\n"
            )
            f.write("#" + " " * 60 + "#\n" + "#" * 62 + "\n\n")
            if talos_params:
                f.write("Static Parameters:\n")
            else:
                f.write("Parameters:\n")
            for key, val in self.static_params.items():
                f.write(f"  {key} = {get_string(val)}\n")
            if talos_params:
                f.write("\nTalos Parameters:\n")
                for key, val in talos_params.items():
                    f.write(f"  {key} = {get_string(val)}\n")
            now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")
            f.write(f"\nTraining started: {now}\n")

    def fit(self, model, compile_kwargs: Dict[str, Any], fit_kwargs: Dict[str, Any]):
        """Fit model to data
        """
        model.compile(**compile_kwargs)
        history = model.fit(**fit_kwargs)
        return model, history
