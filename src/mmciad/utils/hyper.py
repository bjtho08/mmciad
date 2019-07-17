"""Initialize a talos model object for hyper-parameter search

"""
import os
import os.path as osp
from collections import OrderedDict

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.losses import categorical_crossentropy
from keras.optimizers import (SGD, Adadelta, Adagrad, Adam, Adamax, Nadam,
                              RMSprop)
from keras_tqdm import TQDMNotebookCallback
from sklearn.metrics import accuracy_score

from mmciad.utils.callbacks import PatchedModelCheckpoint, WriteLog

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
        if hasattr(val, '__name__'):
            output_dict[key] = val.__name__
        elif isinstance(val, str):
            output_dict[key] = val
        else:
            output_dict[key] = str(val)
    return output_dict

def talos_presets(weight_path, cls_wgts, static_params, train_generator, val_generator):
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
        if internal_params["loss_func"] == "cat_CE":
            loss_func = categorical_crossentropy
        elif internal_params["loss_func"] == "cat_FL":
            cat_focal_loss = categorical_focal_loss()
            loss_func = cat_focal_loss
        elif internal_params["loss_func"] in globals():
            loss_func = globals()[internal_params["loss_func"]]
        elif hasattr(internal_params["loss_func"], '__call__'):
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
        
        depth = 4
        if str(internal_params["pretrain"]) in "enable resnet":
            internal_params["resnet"] = True
            internal_params["pretrain"] = 0
            internal_params["nb_filters_0"] = 32
            depth = 3

        param_strings = value_as_string(internal_params)
        model_base_path = osp.join(
            weight_path,
            internal_params["today_str"],
            "{} {}".format(
                param_strings["loss_func"],
                param_strings["opt"],
            ))

        if not os.path.exists(model_base_path):
            os.makedirs(model_base_path, exist_ok=True)

        modelpath = osp.join(
            model_base_path,
            "talos_U-net_model-"
            + "init_{}-act_{}-decay_{}-drop_{}-weights_{}-pretrain_{}-sigma_{}.h5".format(
                param_strings["init"],
                param_strings["act"],
                param_strings["decay"],
                param_strings["dropout"],
                param_strings["class_weights"],
                param_strings["pretrain"],
                param_strings["sigma_noise"],
            ),
        )
        log_path = (
            "./logs/"
            + "{}/{} {}/init_{}-act_{}-decay_{}-drop_{}-weights_{}-pretrain_{}-sigma_{}/".format(
                static_params["today_str"],
                param_strings["loss_func"],
                param_strings["opt"],
                param_strings["init"],
                param_strings["act"],
                param_strings["decay"],
                param_strings["dropout"],
                param_strings["class_weights"],
                param_strings["pretrain"],
                param_strings["sigma_noise"],
            )
        )

        if internal_params["pretrain"] != 0:
            print(
                "starting with frozen layers\nclass weights: {}".format(class_weights)
            )
            model = u_net(
                internal_params["shape"],
                int(internal_params["nb_filters_0"]),
                sigma_noise=internal_params["sigma_noise"],
                depth=depth,
                initialization=internal_params["init"],
                activation=internal_params["act"],
                dropout=internal_params["dropout"],
                output_channels=internal_params["num_cls"],
                batchnorm=internal_params["batchnorm"],
                pretrain=internal_params["pretrain"],
            )
            model.compile(
                loss=loss_func,
                optimizer=internal_params["opt"](internal_params["lr"]),
                metrics=["acc"],
                #weighted_metrics=["acc"],
            )

            history = model.fit_generator(
                generator=train_generator,
                epochs=10,
                validation_data=val_generator,
                use_multiprocessing=True,
                workers=30,
                class_weight=class_weights,
                verbose=internal_params["verbose"],
                callbacks=[
                    WriteLog(internal_params),
                    TQDMNotebookCallback(
                        metric_format="{name}: {value:0.4f}",
                        leave_inner=True,
                        leave_outer=True,
                    ),
                    TensorBoard(
                        log_dir=log_path,
                        histogram_freq=0,
                        batch_size=internal_params["batch_size"],
                        write_graph=True,
                        write_grads=False,
                        write_images=True,
                        embeddings_freq=0,
                        update_freq="epoch",
                    ),
                ],
            )

            pretrain_layers = [
                "block{}_conv{}".format(block, layer)
                for block in range(1, internal_params["pretrain"] + 1)
                for layer in range(1, 3)
            ]
            for n in pretrain_layers:
                model.get_layer(name=n).trainable = True
            print("layers unfrozen\n")

            model.compile(
                loss=loss_func,
                optimizer=internal_params["opt"](internal_params["lr"]),
                metrics=["acc"],
                #weighted_metrics=["acc"],
            )

            history = model.fit_generator(
                generator=train_generator,
                epochs=internal_params["nb_epoch"],
                initial_epoch=10,
                validation_data=val_generator,
                use_multiprocessing=True,
                workers=30,
                class_weight=class_weights,
                verbose=internal_params["verbose"],
                callbacks=[
                    WriteLog(internal_params),
                    TQDMNotebookCallback(
                        metric_format="{name}: {value:0.4f}",
                        leave_inner=True,
                        leave_outer=True,
                    ),
                    TensorBoard(
                        log_dir=log_path,
                        histogram_freq=0,
                        batch_size=internal_params["batch_size"],
                        write_graph=True,
                        write_grads=False,
                        write_images=True,
                        embeddings_freq=0,
                        update_freq="epoch",
                    ),
                    EarlyStopping(
                        monitor="loss",
                        min_delta=0.0001,
                        patience=10,
                        verbose=0,
                        mode="auto",
                    ),
                    ReduceLROnPlateau(
                        monitor="loss", factor=0.1, patience=3, min_lr=1e-7, verbose=1
                    ),
                    PatchedModelCheckpoint(
                        modelpath, verbose=0, monitor="loss", save_best_only=True
                    ),
                ],
            )
        else:
            print("No layers frozen at start\nclass weights: {}".format(class_weights))
            model = u_net(
                internal_params["shape"],
                int(internal_params["nb_filters_0"]),
                sigma_noise=internal_params["sigma_noise"],
                depth=depth,
                initialization=internal_params["init"],
                activation=internal_params["act"],
                dropout=internal_params["dropout"],
                output_channels=internal_params["num_cls"],
                batchnorm=internal_params["batchnorm"],
                pretrain=internal_params["pretrain"],
                resnet=internal_params["resnet"],
            )

            model.compile(
                loss=loss_func,
                optimizer=internal_params["opt"](internal_params["lr"]),
                metrics=["acc"],
                #weighted_metrics=["acc"],
            )

            history = model.fit_generator(
                generator=train_generator,
                epochs=internal_params["nb_epoch"],
                validation_data=val_generator,
                use_multiprocessing=True,
                workers=30,
                class_weight=class_weights,
                verbose=internal_params["verbose"],
                callbacks=[
                    WriteLog(internal_params),
                    TQDMNotebookCallback(
                        metric_format="{name}: {value:0.4f}",
                        leave_inner=True,
                        leave_outer=True,
                    ),
                    TensorBoard(
                        log_dir=log_path,
                        histogram_freq=0,
                        batch_size=internal_params["batch_size"],
                        write_graph=True,
                        write_grads=False,
                        write_images=True,
                        embeddings_freq=0,
                        update_freq="epoch",
                    ),
                    EarlyStopping(
                        monitor="loss",
                        min_delta=0.0001,
                        patience=10,
                        verbose=0,
                        mode="auto",
                    ),
                    ReduceLROnPlateau(
                        monitor="loss", factor=0.1, patience=3, min_lr=1e-7, verbose=1
                    ),
                    PatchedModelCheckpoint(
                        modelpath, verbose=0, monitor="loss", save_best_only=True
                    ),

                ],
            )
        return history, model
    return talos_model
