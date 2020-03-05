"""Custom loss functions

"""
# from keras.losses import categorical_crossentropy
from itertools import product
from tensorflow.keras import backend as K
from tensorflow import math as M
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
import numpy as np

# import pdb

SMOOTH = 1.0

#  dice_coef and dice_coef_loss have been borrowed from:
#  https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py


def dice1_coef(y_true, y_pred, smooth=SMOOTH):
    """Dice coefficient

    :param y_true: true label
    :type y_true: int
    :param y_pred: predicted label
    :type y_pred: int or float
    :param smooth: smoothing parameter, defaults to SMOOTH
    :type smooth: float, optional
    :return: Dice coefficient
    :rtype: float
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice1_loss(y_true, y_pred, smooth=SMOOTH):
    """Dice Loss function

    :param y_true: true label
    :type y_true: int
    :param y_pred: predicted label
    :type y_pred: int or float
    :param smooth: smoothing parameter, defaults to SMOOTH
    :type smooth: float, optional
    :return: Dice loss
    :rtype: float
    """
    return 1 - dice1_coef(y_true, y_pred, smooth)


def dice2_coef(y_true, y_pred, smooth=SMOOTH):
    """Dice-squared coefficient.

    :param y_true: true label
    :type y_true: int
    :param y_pred: predicted label
    :type y_pred: int or float
    :param smooth: smoothing parameter, defaults to SMOOTH
    :type smooth: float, optional
    :return: Dice-squared coefficient
    :rtype: float
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth
    )


def dice2_loss(y_true, y_pred, smooth=SMOOTH):
    """Dice-squared loss.

    :param y_true: true label
    :type y_true: int
    :param y_pred: predicted label
    :type y_pred: int or float
    :param smooth: smoothing parameter, defaults to SMOOTH
    :type smooth: float, optional
    :return: Dice-squared loss
    :rtype: float
    """
    return 1 - dice2_coef(y_true, y_pred, smooth)


def jaccard2_coef(y_true, y_pred, smooth=SMOOTH):
    """Jaccard squared index coefficient

    :param y_true: true label
    :type y_true: int
    :param y_pred: predicted label
    :type y_pred: int or float
    :param smooth: smoothing parameter, defaults to SMOOTH
    :type smooth: float, optional
    :return: Jaccard coefficient
    :rtype: float
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard2_loss(y_true, y_pred, smooth=SMOOTH):
    """Jaccard squared loss

    :param y_true: true label
    :type y_true: int
    :param y_pred: predicted label
    :type y_pred: int or float
    :param smooth: smoothing parameter, defaults to SMOOTH
    :type smooth: float, optional
    :return: Jaccard loss
    :rtype: float
    """
    return 1 - jaccard2_coef(y_true, y_pred, smooth)


def jaccard1_coef(y_true, y_pred, smooth=SMOOTH):
    """Jaccard index coefficient

    :param y_true: true label
    :type y_true: int
    :param y_pred: predicted label
    :type y_pred: int or float
    :param smooth: smoothing parameter, defaults to SMOOTH
    :type smooth: float, optional
    :return: Jaccard coefficient
    :rtype: float
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard1_loss(y_true, y_pred, smooth=SMOOTH):
    """Jaccard loss

    :param y_true: true label
    :type y_true: int
    :param y_pred: predicted label
    :type y_pred: int or float
    :param smooth: smoothing parameter, defaults to SMOOTH
    :type smooth: float, optional
    :return: Jaccard loss
    :rtype: float
    """
    return 1 - jaccard1_coef(y_true, y_pred, smooth=smooth)


# Ref: salehi17, "Tversky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18


def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum(
        (1 - y_pred) * y_true
    )
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return 1 - answer


def weighted_loss(original_loss_func, weights_list):
    """create weighted loss function from unweighted.

    :param original_loss_func: unweighted loss function
    :type original_loss_func: function
    :param weights_list: list of class weights
    :type weights_list: list
    :return: weighted loss function
    :rtype: function
    """

    def loss_func(y_true, y_pred):
        class_selectors = y_true

        weights = [
            class_selectors[idx] * np.asarray([v for v in weights_list.values()])
            for idx in range(len(y_true))
        ]
        for idx, w in weights_list.items():
            for i in range(len(weights)):
                weights[i][:] = 1.0
                if w == 0:
                    weights[i][idx] = 2.0
                    weights[idx][:] = 0.0
                    weights[idx][idx] = 1.0
        y_pred_weighted = y_pred.__mul__(weights)

        loss = original_loss_func(y_true, y_pred_weighted)
        return loss

    return loss_func


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label
      is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)],
        metrics=["accuracy"], optimizer=adam
     )
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1.0 - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1.0 - epsilon)

        return -K.sum(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0)
        )

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)],
                   metrics=["accuracy"], optimizer=adam)
    """

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


def get_weighted_categorical_crossentropy(weights):
    r"""L = - \sum_i weights[i] y_true[i] \log(y_pred[i])

    :param weights: a list of weights for each class.
    :return: loss function.
    """
    weights = K.variable(weights, dtype=K.floatx())

    def w_cat_CE(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return w_cat_CE


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[..., 0])
    y_pred_max = K.max(y_pred, axis=-1, keepdims=True)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_t, c_p in product(range(nb_cl), range(nb_cl)):
        final_mask += (
            K.cast(weights[c_t, c_p], tf.float32)
            * K.cast(y_pred_max_mat[..., c_p], tf.float32)
            * K.cast(y_true[..., c_t], tf.float32)
        )
    return categorical_crossentropy(y_true, y_pred) * final_mask


def feedback_weight_map(flat_probs, flat_labels, beta, op):
    """
    return the feedback weight map in 1-D tensor
    :param flat_probs: prediction tensor in shape [-1, n_class]
    :param flat_labels: ground truth tensor in shape [-1, n_class]
    """
    probs = K.reduce_sum(flat_probs * flat_labels, axis=-1)
    weight_map = K.exp(-K.pow(probs, beta) * K.log(K.constant(op, "float")))
    return weight_map


def tversky_ce(y_true, y_pred):
    """Combination of region-based and pixel-based loss function

    Parameters
    ----------
    y_true : Array-like tensor
        A tensor of the same shape as `y_pred`

    y_pred : Array-like tensor
        A tensor resulting from a softmax

    Returns
    -------
    Scalar tensor
        output tensor
    """
    return tversky_loss(
        y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10
    ) + categorical_crossentropy(y_true, y_pred)


def wasserstein_disagreement_map(prediction, ground_truth, M):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened pred_proba and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.
    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    n_classes = K.int_shape(prediction)[-1]
    # unstack_labels = tf.unstack(ground_truth, axis=-1)
    ground_truth = tf.cast(ground_truth, dtype=tf.float64)
    # unstack_pred = tf.unstack(prediction, axis=-1)
    prediction = tf.cast(prediction, dtype=tf.float64)
    # print("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(prediction[:, i], ground_truth[:, j])
            )
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map


def generalised_wasserstein_dice_loss(y_true, y_predicted, weight_map):
    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in
    Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
    Multi-class Segmentation using Holistic Convolutional Networks.
    MICCAI 2017 (BrainLes)
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    # apply softmax to pred scores
    n_classes = K.int_shape(y_predicted)[-1]

    ground_truth = tf.cast(tf.reshape(y_true, (-1, n_classes)), dtype=tf.int64)
    pred_proba = tf.cast(tf.reshape(y_predicted, (-1, n_classes)), dtype=tf.float64)

    M = tf.cast(weight_map, dtype=tf.float64)
    # compute disagreement map (delta)
    # print("M shape is ", M.shape, pred_proba, one_hot)
    delta = wasserstein_disagreement_map(pred_proba, ground_truth, M)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(ground_truth, dtype=tf.float64)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot), axis=1
    )
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1.0 - delta), axis=0)
    WGDL = 1.0 - (2.0 * true_pos) / (2.0 * true_pos + all_error)

    return tf.cast(WGDL, dtype=tf.float32)
