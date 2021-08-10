"""Sliding window operations
"""
from typing import Tuple, Union
from collections import namedtuple
from tempfile import TemporaryFile
import numpy as np
from tqdm.auto import tqdm
from tensorflow.keras.models import Model

Size = namedtuple("Size", ["x", "y"])


def sliding_window(
    image: np.ndarray,
    step_size: Union[int, Tuple[int, int]],
    window_size: Union[int, Tuple[int, int]],
):
    """Generate a sliding window of a given image size

    Args:
        image ([type]): [description]
        step_size ([type]): [description]
        window_size ([type]): [description]

    Yields:
        [type]: [description]
    """
    # slide a window across the image
    width, height = image.shape[:2]

    if isinstance(window_size, int):
        window_w = window_h = window_size
    else:
        window_w, window_h = window_size

    if isinstance(step_size, int):
        step_w = step_h = step_size
    else:
        step_w, step_h = step_size

    for y in range(0, height - window_h + step_h, step_h):
        for x in range(0, width - window_w + step_w, step_w):
            # yield the current window
            if y > height - window_h:
                y = height - window_h
            if x > width - window_w:
                x = width - window_w
            dx = x + window_w
            dy = y + window_h
            res_img = image[x:dx, y:dy]
            yield (x, y, dx, dy, res_img)


def predict_window(
    model: Model, img: np.ndarray, step_size=1000, window_size=1024, num_class=11
):
    """Predict a window of a given image using the specified machine learning model.

    Args:
        model ([type]): [description]
        img ([type]): [description]
        step_size (int, optional): [description]. Defaults to 1000.
        window_size (int, optional): [description]. Defaults to 1024.
        num_class (int, optional): [description]. Defaults to 11.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    width, height = img.shape[:2]
    dtype = img.dtype
    if (width > window_size) & (height > window_size):
        with TemporaryFile(prefix="window_out_", dir="data/tmp/") as out_file:
            output_img = np.memmap(
                out_file, dtype=dtype, mode="w+", shape=(width, height, num_class),
            )
        # output_img = np.zeros(shape=(width, height, num_class))
        output_img[:] = np.nan
        x_total = 1 + (width - window_size + step_size) // step_size
        y_total = 1 + (height - window_size + step_size) // step_size
        pbar = tqdm(total=x_total * y_total, desc="Predicting window")
        for (x, y, dx, dy, window) in sliding_window(img, step_size, window_size):
            window_prediction = model.predict(np.expand_dims(window, axis=0))
            pred_shape = np.squeeze(window_prediction).shape
            out_shape = output_img[x:dx, y:dy].shape
            if out_shape != pred_shape:
                raise ValueError(
                    f"incoming tile shape ({pred_shape}) is "
                    + f"different from existing shape ({out_shape})"
                )
            output_img[x:dx, y:dy] = np.nanmean(
                np.stack(
                    (output_img[x:dx, y:dy], np.squeeze(window_prediction)), axis=0
                ),
                axis=0,
            ).astype(dtype)
            pbar.update(1)
        pbar.close()
        return output_img
    output_img = model.predict(img)
    return output_img


def concat_windows(slides, targets, step_size=1000, window_size=1024):
    """Concatenate two sliding windows into two n-D arrays.

    Args:
        slides ([type]): [description]
        targets ([type]): [description]
        step_size (int, optional): [description]. Defaults to 1000.
        window_size (int, optional): [description]. Defaults to 1024.

    Returns:
        [ndarray]: [description]
    """
    input_tiles = []
    target_tiles = []
    for slide, target in zip(slides, targets):
        width, height, _ = slide.shape
        if (width > window_size) & (height > window_size):
            for (x, y, dx, dy, window) in sliding_window(slide, step_size, window_size):
                input_tiles.append(window)
                target_tiles.append(target[x:dx, y:dy])
    input_tiles = np.asarray(input_tiles)
    target_tiles = np.asarray(target_tiles)
    return input_tiles, target_tiles
