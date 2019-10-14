"""Initialize a talos model object for hyper-parameter search

"""
import os
import os.path as osp
from collections import OrderedDict

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.losses import categorical_crossentropy

# from keras_contrib.callbacks import DeadReluDetector
#from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop
from keras_tqdm import TQDMNotebookCallback
#from keras_radam import RAdam

# from sklearn.metrics.scorer import accuracy_score

from mmciad.utils.callbacks import PatchedModelCheckpoint#, DeadReluDetector

from .custom_loss import categorical_focal_loss, tversky_loss, weighted_loss
from .u_net import u_net

IMG_ROWS, IMG_COLS, IMG_CHANNELS = (None, None, 3)
# architecture params
NB_FILTERS_0 = 64

# ****  deep learning model
SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
BATCH_SIZE = 16
NB_EPOCH = 200
VERBOSE = 0

# ****  train


def value_as_string(input_dict):
    output_dict = {}
    for key, val in input_dict.items():
        if hasattr(val, "__name__"):
            output_dict[key] = val.__name__
        elif isinstance(val, str):
            output_dict[key] = val
        else:
            output_dict[key] = str(val)
    return output_dict


def talos_presets(weight_path, cls_wgts, static_params, train_generator, val_generator, notebook=True):
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
        path_elements = [
            '{}_{}'.format(key, val.__name__)
            if hasattr(val, '__name__')
            else '{}_{}'.format(key, val) for key, val in talos_params.items()
        ]
        path_elements.remove('{}_{}'.format("arch", talos_params["arch"]))
        if internal_params["loss_func"] == "cat_CE":
            loss_func = categorical_crossentropy
        elif internal_params["loss_func"] == "cat_FL":
            cat_focal_loss = categorical_focal_loss()
            loss_func = cat_focal_loss
        elif internal_params["loss_func"] in globals():
            loss_func = globals()[internal_params["loss_func"]]
        elif hasattr(internal_params["loss_func"], "__call__"):
            loss_func = internal_params["loss_func"]
        elif internal_params["loss_func"] == "w_cat_CE":
            loss_func = weighted_loss(categorical_crossentropy, cls_wgts)
            loss_func.__name__ = internal_params["loss_func"]
        elif internal_params["loss_func"] == "w_cat_FL":
            loss_func = weighted_loss(categorical_focal_loss, cls_wgts)
            loss_func.__name__ = internal_params["loss_func"]
        elif internal_params["loss_func"] == "w_TL":
            loss_func = weighted_loss(tversky_loss, cls_wgts)
            loss_func.__name__ = internal_params["loss_func"]
        else:
            raise NameError("Wrong loss function name")
        # mse, mae, binary_crossentropy, jaccard2_loss, categorical_crossentropy,
        # tversky_loss, categorical_focal_loss
        if internal_params["class_weights"] is False:
            class_weights = [1 if k != 12 else 0 for k in cls_wgts.keys()]
        else:
            class_weights = ([v for v in cls_wgts.values()],)

        if str(internal_params["arch"]).lower() == "u-resnet":
            internal_params["maxpool"] = False
            #internal_params["pretrain"] = 0
            #internal_params["nb_filters_0"] = 32
            #internal_params["depth"] = 3

        param_strings = value_as_string(internal_params)
        model_base_path = osp.join(
            weight_path,
            internal_params["today_str"],
            internal_params["arch"],
            "{} {}".format(param_strings["loss_func"], param_strings["opt"]),
        )

        if not os.path.exists(model_base_path):
            os.makedirs(model_base_path, exist_ok=True)

        modelpath = osp.join(
            model_base_path, "talos_U-net_model-" + '-'.join(path_elements) + ".h5"
        )
        log_path = osp.join(
            "./logs",
            static_params["today_str"],
            internal_params["arch"],
            "{} {}".format(param_strings["loss_func"], param_strings["opt"]),
            *path_elements, ''
        )
        model_kwargs = {
            "shape": internal_params["shape"],
            "nb_filters": int(internal_params["nb_filters_0"]),
            "sigma_noise": internal_params["sigma_noise"],
            "depth": internal_params["depth"],
            "maxpool": internal_params["maxpool"],
            "initialization": internal_params["init"],
            "activation": internal_params["act"],
            "dropout": internal_params["dropout"],
            "output_channels": internal_params["num_cls"],
            "batchnorm": internal_params["batchnorm"],
            "pretrain": internal_params["pretrain"],
            "arch": internal_params["arch"],
        }

        csv_logger = CSVLogger("csvlog.csv", append=True)
        if notebook:
            tqdm_progress = TQDMNotebookCallback(
                metric_format="{name}: {value:0.4f}", leave_inner=True, leave_outer=True
            )
        tb_callback = TensorBoard(
            log_dir=log_path,
            histogram_freq=0,
            batch_size=internal_params["batch_size"],
            write_graph=True,
            write_grads=False,
            write_images=True,
            embeddings_freq=0,
            update_freq="epoch",
        )
        early_stopping = EarlyStopping(
            monitor="loss", min_delta=0.0001, patience=15, verbose=0, mode="auto"
        )
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor="loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1
        )
        model_checkpoint = PatchedModelCheckpoint(
            modelpath, verbose=0, monitor="loss", save_best_only=True
        )
        # logger_callback = WriteLog(internal_params)
        # dead_relus = DeadReluDetector(x_train=train_generator)

        model_callbacks = [csv_logger, tb_callback]
        if notebook:
            model_callbacks.append(tqdm_progress)
        opti_callbacks = [early_stopping, reduce_lr_on_plateau, model_checkpoint]

        if internal_params["pretrain"] != 0:
            print(
                "starting with frozen layers\nclass weights: {}".format(class_weights)
            )
            model = u_net(**model_kwargs)
            model.compile(
                loss=loss_func,
                optimizer=internal_params["opt"](internal_params["lr"]),
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
            model.save()
            keras.models.load_model
            model.compile(
                loss=loss_func,
                optimizer=internal_params["opt"](internal_params["lr"]),
                metrics=["acc"],
                # weighted_metrics=["acc"],
            )

            history = model.fit_generator(
                generator=train_generator,
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
            print("No layers frozen at start\nclass weights: {}".format(class_weights))
            model = u_net(**model_kwargs)

            model.compile(
                loss=loss_func,
                optimizer=internal_params["opt"](internal_params["lr"]),
                metrics=["acc"],
                # weighted_metrics=["acc"],
            )

            history = model.fit_generator(
                generator=train_generator,
                epochs=internal_params["nb_epoch"],
                validation_data=val_generator,
                use_multiprocessing=True,
                workers=30,
                class_weight=class_weights,
                verbose=internal_params["verbose"],
                callbacks=model_callbacks + opti_callbacks,
            )
        return history, model

    return talos_model
