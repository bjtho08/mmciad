"""Initialize a talos model object for hyper-parameter search

"""
import datetime
import os
import os.path as osp
from collections import OrderedDict
from typing import Union

import tensorflow_addons as tfa
from keras_radam import RAdam
from tensorflow.keras.activations import selu, sigmoid, tanh
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
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

from keras_contrib.layers.advanced_activations.swish import Swish

from .callbacks import PatchedModelCheckpoint  # , DeadReluDetector
from .custom_loss import (
    categorical_focal_loss,
    jaccard1_coef,
    jaccard1_loss,
    jaccard2_loss,
    tversky_loss,
    weighted_loss,
)
from .u_net import u_net

# from keras_tqdm import TQDMNotebookCallback
# import talos
# from sklearn.metrics.scorer import accuracy_score


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


def value_as_string(input_dict):
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
        except KeyError:
            raise NameError(f"Wrong loss function name ({name})")

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
        "radam": RAdam,
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


def prepare_for_talos(
    weight_path,
    cls_wgts,
    static_params,
    train_generator,
    val_generator,
    notebook=True,
    debug=False,
):
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

    def talos_model(x, y, val_x, val_y, talos_params):
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
        _ = x, y, val_x, val_y
        internal_params = OrderedDict()
        internal_params.update(static_params)
        internal_params.update(talos_params)
        loss_func = get_loss_function(internal_params["loss_func"])
        act_func = get_act_function(internal_params["act"])
        opt_func = get_opt_function(internal_params["opt"])

        if debug:
            internal_params["steps_per_epoch"] = 2
            internal_params["nb_epoch"] = 2
            global CRASH_COUNT  # pylint: disable=global-statement
            CRASH_COUNT += 1
            if CRASH_COUNT == 3:
                raise KeyboardInterrupt()

        if internal_params["class_weights"] is False:
            class_weights = [1 if k != 12 else 0 for k in cls_wgts.keys()]
        else:
            class_weights = list(cls_wgts.values())

        if str(internal_params["arch"]).lower() == "u-resnet":
            internal_params["maxpool"] = False
            # internal_params["pretrain"] = 0
            # internal_params["nb_filters_0"] = 32
            # internal_params["depth"] = 3

        model_base_path = osp.join(weight_path, internal_params["date"])

        if not os.path.exists(model_base_path):
            os.makedirs(model_base_path, exist_ok=True)
        basename = "model"

        suffix = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
        model_handle = "_".join([basename, suffix])  # e.g. 'mylogfile_120508_171442'
        modelpath = osp.join(model_base_path, model_handle)
        log_path = osp.join("./logs", static_params["date"], model_handle, "")
        with open(file=modelpath + ".cfg", mode="w") as f: # pylint: disable=invalid-name
            f.write("#" * 62 + "\n#" + " " * 60 + "#\n")
            f.write("#" + f"Model Paramters for {model_handle}_*.h5".center(60) + "#\n")
            f.write("#" + " " * 60 + "#\n" + "#" * 62 + "\n\n")
            f.write("Static Parameters:\n")
            for key, val in static_params.items():
                f.write(f"  {key} = {get_string(val)}\n")
            f.write("\nTalos Parameters:\n")
            for key, val in talos_params.items():
                f.write(f"  {key} = {get_string(val)}\n")
            now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")
            f.write(f"\nTraining started: {now}\n")

        model_kwargs = {
            "shape": internal_params["shape"],
            "nb_filters": int(internal_params["nb_filters_0"]),
            "sigma_noise": internal_params["sigma_noise"],
            "depth": internal_params["depth"],
            "maxpool": internal_params["maxpool"],
            "initialization": internal_params["init"],
            "activation": act_func,
            "dropout": internal_params["dropout"],
            "output_channels": internal_params["num_cls"],
            "batchnorm": internal_params["batchnorm"],
            "pretrain": internal_params["pretrain"],
            "arch": internal_params["arch"],
        }

        csv_logger = CSVLogger(modelpath + ".fit.csv", append=True)
        if notebook:
            tqdm_progress = tfa.callbacks.TQDMProgressBar()
        #            tqdm_progress = TQDMNotebookCallback(
        #                metric_format="{name}: {value:0.4f}", leave_inner=True, leave_outer=True
        #            )
        tb_callback = TensorBoard(
            log_dir=log_path,
            histogram_freq=0,
            write_graph=True,
            write_grads=False,
            write_images=False,
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

        # talos_path = osp.join("talos", internal_params["date"])
        # gamify_callback = talos.utils.ExperimentLogCallback(talos_path, talos_params)
        # dead_relus = DeadReluDetector(x_train=train_generator)

        model_callbacks = [csv_logger, tb_callback]  # , gamify_callback]
        if notebook:
            model_callbacks.append(tqdm_progress)
        opti_callbacks = [
            early_stopping,
            reduce_lr_on_plateau,
            loss_model_checkpoint,
            jaccard_model_checkpoint,
        ]

        if internal_params["pretrain"] != 0:
            print(
                "starting with frozen layers\nclass weights: {}".format(class_weights)
            )
            model = u_net(**model_kwargs)
            model.compile(
                loss=loss_func,
                optimizer=opt_func(internal_params["lr"]),
                metrics=["acc"],
                # weighted_metrics=["acc"],
            )

            history = model.fit_generator(
                generator=train_generator,
                epochs=internal_params["nb_frozen"],
                validation_data=val_generator,
                use_multiprocessing=True,
                workers=30,
                class_weight=class_weights,
                verbose=internal_params["verbose"],
                callbacks=model_callbacks,
            )

            pretrain_layers = [
                "block{}_d_conv{}".format(block, layer)
                for block in range(1, internal_params["pretrain"] + 1)
                for layer in range(1, 3)
            ]
            for n in pretrain_layers:
                model.get_layer(name=n).trainable = True
            print("layers unfrozen\n")
            model.compile(
                loss=loss_func,
                optimizer=opt_func(internal_params["lr"]),
                metrics=["acc"],
                # weighted_metrics=["acc"],
            )

            history = model.fit(
                x=train_generator,
                epochs=internal_params["nb_epoch"],
                initial_epoch=internal_params["nb_frozen"],
                validation_data=val_generator,
                use_multiprocessing=True,
                workers=30,
                class_weight=class_weights,
                verbose=internal_params["verbose"],
                callbacks=model_callbacks + opti_callbacks,
            )
        else:
            # print(
            #   "No layers frozen at start\nclass weights: {}".format(class_weights)
            # )
            model = u_net(**model_kwargs)

            model.compile(
                loss=loss_func,
                optimizer=opt_func(internal_params["lr"]),
                metrics=["acc", jaccard1_coef],
            )

            history = model.fit(
                x=train_generator,
                epochs=internal_params["nb_epoch"],
                validation_data=val_generator,
                workers=30,
                steps_per_epoch=internal_params.get("steps_per_epoch", None),
                validation_steps=internal_params.get("steps_per_epoch", None),
                # class_weight=class_weights,
                verbose=internal_params["verbose"],
                callbacks=model_callbacks + opti_callbacks,
            )
        return history, model

    return talos_model
