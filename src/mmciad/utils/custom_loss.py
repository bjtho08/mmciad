import numpy as np
from keras.losses import categorical_crossentropy
from keras import backend as K
import tensorflow as tf

# import pdb

SMOOTH = 1.0

#  dice_coef and dice_coef_loss have been borrowed from:
#  https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py


def dice1_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice1_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - dice1_coef(y_true, y_pred, smooth)


def dice2_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth
    )


def dice2_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - dice2_coef(y_true, y_pred, smooth)


def jaccard2_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard2_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - jaccard2_coef(y_true, y_pred, smooth)


def jaccard1_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard1_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - jaccard1_coef(y_true, y_pred, smooth=smooth)


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
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
    return 1-answer


# def tversky_loss(y_true, y_pred):
#     """Tversky loss function for image segmentation

#     :param y_true: [description]
#     :type y_true: [type]
#     :param y_pred: [description]
#     :type y_pred: [type]
#     :return: [description]
#     :rtype: [type]
#     """
#     alpha = 0.3
#     beta = 0.7

#     ones = K.ones_like(y_true)
#     p0 = y_pred      # proba that voxels are class i
#     p1 = ones-y_pred # proba that voxels are not class i
#     g0 = y_true
#     g1 = ones-y_true

#     num = K.sum(p0*g0, (0, 1, 2))
#     den = num + alpha*K.sum(p0*g1, (0, 1, 2)) + beta*K.sum(p1*g0, (0, 1, 2))

#     T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

#     Ncl = K.cast(K.shape(y_true)[-1], 'float32')
#     return Ncl-T


def weighted_loss(original_loss_func, weights_list):
    def loss_func(y_true, y_pred):
        axis = -1  # if channels last
        # axis=  1 #if channels first

        # argmax returns the index of the element with the greatest value
        # done in the class axis, it returns the class index
        class_selectors = K.argmax(y_true, axis=axis)

        # considering weights are ordered by class, for each class
        # true(1) if the class index is equal to the weight index
        class_selectors = [
            K.equal(K.cast(i, "int64"), K.cast(class_selectors, "int64"))
            for i in range(len(weights_list))
        ]

        # casting boolean to float for calculations
        # each tensor in the list contains 1 for ground true class is equal to its index
        # if you sum all these, you will get a tensor full of ones.
        class_selectors = [K.cast(x, K.floatx()) for x in class_selectors]

        # for each of the selections above, multiply their respective weight
        weights = [
            sel * w
            for sel, w in zip(
                class_selectors, sorted(value for value in weights_list.values())
            )
        ]

        # sums all the selections
        # result is a tensor with the respective weight for each element in predictions
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]

        # make sure your original_loss_func only collapses the class axis
        # you need the other axes intact to multiply the weights tensor
        loss = original_loss_func(y_true, y_pred)
        loss = loss * weight_multiplier

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
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
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
    """L = - \sum_i weights[i] y_true[i] \log(y_pred[i])
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
