import os
import os.path as osp
from glob import glob
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
from keras_contrib.layers.advanced_activations.swish import Swish
from sklearn.metrics import jaccard_score
from skimage.io import imsave

from .io import load_slides_as_dict
from .preprocessing import calculate_stats
from .postprocessing import auto_contrast
from .custom_loss import (
    categorical_focal_loss,
    jaccard2_loss,
    tversky_loss,
)


data_path = "./data/"
weight_path = "./weights/"
results_path = "./results/"
train_date = "2019-11-02"
weights = sorted(
    glob(osp.join(weight_path, train_date, "*", "*", "*.h5")), key=str.lower
)
Size = namedtuple("Size", ["x", "y"])
# weights = [modelpath]
INV_DPI = 1 / 100
SCALE = 1 / 8


def batch(iterable, n=1):
    """Create small batches from iterable
    
    Parameters
    ----------
    iterable : iterable
        input iterable to be subdivided
    n : int, optional
        batch size, by default 1
    """
    length = len(iterable)
    counter = 0
    for ndx in range(0, length, n):
        counter += 1
        yield counter, iterable[ndx : min(ndx + n, length)]


def sliding_window(image, step_size: int, window_size: int):
    """Create sliding window over image matrix
    
    Parameters
    ----------
    image : array-like
        Input image
    step_size : int
        Distance between each window upper left corner
    window_size : int
        Size of each window
    """
    # slide a window across the image
    for y in range(0, image.shape[1] - window_size[0] + step_size, step_size):
        for x in range(0, image.shape[0] - window_size[1] + step_size, step_size):
            # yield the current window
            res_img = image[x : x + window_size[0], y : y + window_size[1]]
            change = False
            if res_img.shape[1] != window_size[1]:
                y = image.shape[1] - window_size[1]
                change = True
            if res_img.shape[0] != window_size[0]:
                x = image.shape[0] - window_size[0]
                change = True
            if change:
                res_img = image[x : x + window_size[0], y : y + window_size[1]]
            yield (x, y, x + window_size[0], y + window_size[1], res_img)


def predict_window(model, img, step_size=1000, wsize=1024, num_class=11):
    size = Size(img.shape[0], img.shape[1])
    dtype = img.dtype
    if (size.x > wsize) & (size.y > wsize):
        output_img = np.zeros(shape=(size.x, size.y, num_class))
        output_img[:] = np.nan
        for (x, y, dx, dy, I) in sliding_window(img, step_size, (wsize, wsize)):
            window_prediction = model.predict(np.expand_dims(I, axis=0))
            output_img[x:dx, y:dy] = np.nanmean(
                np.stack((output_img[x:dx, y:dy], np.squeeze(window_prediction)), axis=0), axis=0
            ).astype(dtype)
        return output_img
    output_img = model.predict(img)
    return output_img


def evaluate_window(model, img, target, step_size=1000, wsize=1024, batch_size=None):
    size = Size(img.shape[0], img.shape[1])
    if (size.x > wsize) & (size.y > wsize):
        input_tiles = []
        target_tiles = []
        for (x, y, dx, dy, I) in sliding_window(img, step_size, (wsize, wsize)):
            input_tiles.append(I)
            target_tiles.append(target[x:dx, y:dy])
        input_tiles = np.asarray(input_tiles)
        target_tiles = np.asarray(target_tiles)
        if batch_size is None:
            batch_size = input_tiles.shape[0]
        print(target_tiles.shape)
        return model.evaluate(x=input_tiles, y=target_tiles, batch_size=batch_size)
    return model.evaluate(x=img, y=target, batch_size=1)


def concat_windows(slides, targets, step_size=1000, wsize=1024):
    input_tiles = []
    target_tiles = []
    for img, target in zip(slides, targets):
        size = Size(img.shape[0], img.shape[1])
        if (size.x > wsize) & (size.y > wsize):
            for (x, y, dx, dy, I) in sliding_window(slide, step_size, (wsize, wsize)):
                input_tiles.append(I)
                target_tiles.append(target[x:dx, y:dy])
    input_tiles = np.asarray(input_tiles)
    target_tiles = np.asarray(target_tiles)
    return input_tiles, target_tiles


class_map = {
    "Background": 0,
    "Other": 1,
    "Epithelium": 2,
    "Glandular tissue": 3,
    "Necrosis": 4,
    "Stroma": 5,
    "Muscular tissue": 6,
    "Inflammation_lymphatic tissue": 7,
    "Ulcer": 8,
    "Dysplasia": 9,
    "Keratin pearl": 10,
    "Cancer": 11,
    "IGNORE": 12,
}

class_colors = {
    0: [0, 0, 0],
    1: [128, 128, 128],
    2: [75, 195, 0],
    3: [0, 26, 128],
    4: [94, 243, 255],
    5: [255, 179, 128],
    6: [217, 30, 242],
    7: [204, 102, 51],
    8: [153, 51, 0],
    9: [245, 223, 37],
    10: [179, 230, 179],
    11: [179, 26, 26],
    12: [255, 255, 255],
}
active_labels = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12]
active_classes = [sorted(class_map, key=class_map.get)[i] for i in active_labels]
colorvec = np.asarray([class_colors[i] for i in active_labels])
num_cls = len(active_labels)
ignore_cls = 12

train_m, train_s, x_min, x_max = calculate_stats(path=data_path, local=False)
test_slides, test_targets = load_slides_as_dict(
    data_path, "test", train_m, train_s, [x_min, x_max], True, num_cls, colorvec
)

subslides = {}
for name in test_targets.keys():  # ["N9a-1", "T4a"]:#"N10a", "T4b-1", "T4b-2"]:
    subslides[name] = test_slides[name]
print(subslides.keys())
prediction_byte = {}
jaccard = {}
jaccard_weighted = {}
jaccard_weighted_nobg = {}

for count, b in batch(weights, 8):
    row_titles = []
    col_titles = active_classes.copy()
    col_titles.extend(["argmax", "target", "input"])

    pad = 5  # in points
    # test_name = "T4a"
    fig, axes = {}, {}
    for name, slide in subslides.items():
        slide = np.squeeze(slide)
        input_width = slide.shape[1]
        input_height = slide.shape[0]
        num_columns = num_cls + 3
        fig_width = input_width * INV_DPI * SCALE
        fig_height = input_height * INV_DPI * SCALE
        fig[name], axes[name] = plt.subplots(
            nrows=1,
            ncols=num_columns,
            sharex=True,
            sharey=True,
            figsize=(fig_width * num_columns, fig_height),
            gridspec_kw={"hspace": 0, "wspace": 0},
        )
    for i, w in enumerate(b):
        # sess = get_session()
        # clear_session()
        # sess.close()
        # sess = get_session()

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # K.tensorflow_backend.set_session(tf.Session(config=config))
        with tf.device("/cpu:*"):  # Remember to re-indent model + compile below
            test_model = load_model(
                weights[0],
                custom_objects={
                    "Swish": Swish,
                    "tversky_loss": tversky_loss,
                    "categorical_focal_loss": categorical_focal_loss,
                    "jaccard2_loss": jaccard2_loss,
                },
            )

        for test_name, slide in subslides.items():
            slide = np.squeeze(slide)
            loss_dir = str(osp.split(w)[0]).split("/")[-2]
            arch_dir = str(osp.split(w)[0]).split("/")[-1]
            base_name = osp.split(w)[-1]
            class_wgt_dir = "weights_{}".format("True" in str(base_name))
            act_dir = "{}".format(
                base_name[base_name.rfind("activation_") : base_name.find("-dropout_")]
            )
            init_dir = "{}".format(
                base_name[
                    base_name.rfind("initialization_") : base_name.find("-activation_")
                ]
            )
            # pre_dir = '{}'.format(base_name[base_name.rfind('pretrain_'):base_name.find('-sigma_')])
            res_path = osp.join(
                results_path,
                train_date,
                test_name,
                loss_dir,
                arch_dir,
                " ".join([class_wgt_dir, init_dir, act_dir]),
            )
            row_titles.append(
                loss_dir + "\n" + class_wgt_dir + "\n" + act_dir + "\n" + init_dir
            )
            if not osp.exists(res_path):
                os.makedirs(res_path)
            raw_input = auto_contrast(slide)
            # test_model.load_weights(w)
            print(f"slide: {slide.shape!s:>10}")
            output = predict_window(test_model, slide)
            prediction = colorvec[np.argmax(output, axis=-1)]
            prediction_byte[test_name] = np.argmax(output, axis=-1)
            target_byte = np.argmax(test_targets[test_name], axis=-1)
            print(f"prediction_byte: {prediction_byte[test_name].shape!s:>10}")
            print(f"test_target: {test_targets[test_name].shape!s:>10}")
            print(f"test_target_byte: {target_byte.shape!s:>10}")
            jaccard[test_name] = jaccard_score(
                np.ndarray.flatten(prediction_byte[test_name]),
                np.ndarray.flatten(target_byte),
                average=None,
            )
            jaccard_weighted[test_name] = jaccard_score(
                np.ndarray.flatten(prediction_byte[test_name]),
                np.ndarray.flatten(target_byte),
                average="weighted",
            )
            jaccard_weighted_nobg[test_name] = jaccard_score(
                np.ndarray.flatten(prediction_byte[test_name]),
                np.ndarray.flatten(target_byte),
                labels=active_labels[1:],
                average="weighted",
            )
            if len(b) == 1:
                for j in range(num_cls):
                    axes[test_name][j].imshow(output[:, :, j], cmap="jet")
                    plt.imsave(
                        res_path + "/result-{}.png".format(active_classes[j]),
                        output[:, :, j],
                        cmap="jet",
                    )
                axes[test_name][num_cls].imshow(prediction)
                axes[test_name][num_cls + 1].imshow(colorvec[target_byte])
                axes[test_name][num_cls + 2].imshow(raw_input)
            else:
                for j in range(num_cls):
                    axes[test_name][i, j].imshow(output[:, :, j], cmap="jet")
                    plt.imsave(
                        res_path + "/result-{}.png".format(active_classes[j]),
                        output[:, :, j],
                        cmap="jet",
                    )
                axes[test_name][i, num_cls].imshow(prediction)
                axes[test_name][num_cls + 1].imshow(colorvec[target_byte])
                axes[test_name][i, num_cls + 2].imshow(raw_input)
            imsave(
                res_path + "/result_argmax.png",
                prediction.astype(np.uint8),
                check_contrast=False,
            )
            imsave(
                res_path + "/result_input.png",
                np.array(raw_input * 256.0, dtype=np.uint8),
                check_contrast=False,
            )

    for name in subslides:
        if len(b) == 1:
            for ax, col in zip(axes[name][:], col_titles):
                ax.set_title(col)
            for row in row_titles:
                axes[name][0].annotate(
                    row,
                    xy=(0, 0.5),
                    xytext=(-axes[name][0].yaxis.labelpad - pad, 0),
                    xycoords=axes[name][0].yaxis.label,
                    textcoords="offset points",
                    size="large",
                    ha="right",
                    va="center",
                    wrap=True,
                )
        else:
            for ax, col in zip(axes[name][0], col_titles):
                ax.set_title(col)
            for ax, row in zip(axes[name][:, 0], row_titles):
                ax.annotate(
                    row,
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    size="large",
                    ha="right",
                    va="center",
                    wrap=True,
                )
        fig[name].tight_layout(pad=1.5)
        fig[name].subplots_adjust(left=0.15, top=0.95)
        plt.show()
        fig[name].savefig(
            osp.join("./results", train_date, name + "-" + str(count) + "-overview.png")
        )
        fig[name].clear()
        plt.close()
