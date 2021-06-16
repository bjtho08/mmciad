"""U-Net+ResNet model implementation with keras"""

import keras
from keras import backend as K
from keras.models import Model
from keras.activations import relu
from keras.layers.merge import concatenate, add
from keras.layers import (
    Input,
    Add,
    Activation,
    Concatenate,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    GaussianNoise,
    Dropout,
    BatchNormalization,
)

def _shortcut(input_, residual, resize=None):
    input_shape = K.int_shape(input_)
    residual_shape = K.int_shape(residual)
    #stride_width = input_shape[1] / residual_shape[1]
    #stride_height = input_shape[2] / residual_shape[2]
    equal_channels = input_shape[3] == residual_shape[3]
    shortcut = input_
    if resize is not None or not equal_channels:
        if resize < 1:
            shortcut = UpSampling2D(
                size=(int(round(1/resize)),int(round(1/resize)))
            )(shortcut)
            resize = max(1, resize)
            resize = max(1, resize)
        shortcut = Conv2D(
            filters=residual_shape[3],
            kernel_size=1,
            strides=(int(round(resize)), int(round(resize))),
            padding="same",
            kernel_initializer="he_normal")(shortcut)
    return add([shortcut, residual])


def bottleneck(m, nb_filters, init, layer, level, drop, down, up):
    n = BatchNormalization(name="layer{}_block{}_bn1".format(layer, level))(m)
    n = Activation("relu")(n)
    n = Conv2D(
        filters=nb_filters,
        kernel_size=1,
        strides=down,
        padding="same",
        kernel_initializer=init,
        name="layer{}_block{}_conv1".format(layer, level))(n)
    n = BatchNormalization(name="layer{}_block{}_bn2".format(layer, level))(n)
    n = Activation("relu")(n)
    n = Conv2D(
        filters=nb_filters,
        kernel_size=3,
        padding="same",
        kernel_initializer=init,
        name="layer{}_block{}_conv2".format(layer, level))(n)
    n = BatchNormalization(name="layer{}_block{}_bn3".format(layer, level))(n)
    n = Activation("relu")(n)
    n = UpSampling2D(up)(n) if up > 1 else n
    n = Conv2D(
        filters=nb_filters*4,
        kernel_size=1,
        padding="same",
        kernel_initializer=init,
        name="layer{}_block{}_conv3".format(layer, level))(n)
    n = Dropout(drop, name="layer{}_block{}_drop".format(layer, level)) if drop else n
    resize = down/up
    return _shortcut(m, n, resize)

def basic(m, nb_filters, init, level, drop, down, up):
    n = BatchNormalization(name="basic_block{}_bn1".format(level))(m)
    n = Activation("relu")(n)
    n = Conv2D(
        filters=nb_filters,
        kernel_size=3,
        strides=down,
        padding="same",
        kernel_initializer=init,
        name="basic_block{}_conv1".format(level))(n)
    n = Dropout(drop, name="basic_block{}_drop".format(level)) if drop else n
    n = BatchNormalization(name="basic_block{}_bn2".format(level))(n)
    n = Activation("relu")(n)
    n = UpSampling2D(up)(n) if up > 1 else n
    n = Conv2D(
        filters=nb_filters,
        kernel_size=3,
        padding="same",
        kernel_initializer=init,
        name="basic_block{}_conv2".format(level))(n)
    return n

def simple_block(m, nb_filters, init, level, drop, down, up):
    n = BatchNormalization(name="simple_block{}_bn1".format(level))(m)
    n = Activation("relu")(n)
    n = MaxPooling2D(
        pool_size=(down, down),
        name="simple_block{}_MaxPool".format(level)
    )(n) if down > 1 else n
    n = Conv2D(
        filters=nb_filters,
        kernel_size=3,
        padding="same",
        kernel_initializer=init,
        name="simple_block{}_conv1".format(level))(n)
    n = UpSampling2D(up)(n) if up > 1 else n
    n = Dropout(drop, name="simple_block{}_drop".format(level)) if drop else n
    resize = down/up
    return _shortcut(m, n, resize)

def level_block(m, nb_filters, init, layer, level, drop, down, up, num_levels):
    if num_levels > 0:
        n = bottleneck(
            m,
            nb_filters,
            init=init,
            layer=layer,
            level=level,
            drop=drop,
            down=down,
            up=up)
        level = level + 1 # FIX improper recursion, resets on each call
        m = level_block(
            n,
            nb_filters,
            init,
            layer,
            level,
            drop,
            down=1,
            up=1,
            num_levels=num_levels-1)
    return m

def u_resnet(
    shape,
    nb_filters=64,
    #conv_size=3,
    depth=5,
    initialization="he_normal",
    inc_rate=2.0,
    dropout=0,
    output_channels=5,
    #batchnorm=False,
    #maxpool=True,
    #upconv=True,
    #pretrain=0,
    #sigma_noise=0,
):
    i = Input(shape, name="input_layer")
    down_1 = Conv2D(nb_filters, 3, padding="same", name="down_1_block1_conv1")(i)
    down_2 = simple_block(down_1, nb_filters, initialization, level="down_2", drop=dropout, down=2, up=1)
    down_3 = level_block(down_2, int(nb_filters*1), initialization, layer="down_3", level=1, drop=dropout, down=2, up=1, num_levels=3)
    down_4 = level_block(down_3, int(nb_filters*2), initialization, layer="down_4", level=1, drop=dropout, down=2, up=1, num_levels=8)
    #down_5 = level_block(down_4, int(nb_filters*4), initialization, layer="down_5", level=1, drop=dropout, down=2, up=1, num_levels=10)
    across_1 = bottleneck(down_4, int(nb_filters*4), initialization, layer="across_1", level=1, drop=dropout, down=2, up=1)
    across_2 = bottleneck(across_1, int(nb_filters*4), initialization, layer="across_2", level=1, drop=dropout, down=1, up=1)
    across_3 = bottleneck(across_2, int(nb_filters*4), initialization, layer="across_3", level=1, drop=dropout, down=1, up=2)
    across_3 = Concatenate(name="up_1_Concatenate")([down_4, across_3])
    #up_1 = level_block(across_3, int(nb_filters*4), initialization, layer="up_1", level=1, drop=dropout, down=1, up=2, num_levels=10)
    #up_1 = Concatenate(name="up_2_Concatenate")([down_4, up_1])
    up_2 = level_block(across_3, int(nb_filters*2), initialization, layer="up_2", level=1, drop=dropout, down=1, up=2, num_levels=8)
    up_2 = Concatenate(name="up_3_Concatenate")([down_3, up_2])
    up_3 = level_block(up_2, int(nb_filters*1), initialization, layer="up_3", level=1, drop=dropout, down=1, up=2, num_levels=3)
    up_3 = Concatenate(name="up_4_Concatenate")([down_2, up_3])
    up_4 = simple_block(up_3, nb_filters, initialization, level="up_4", drop=dropout, down=1, up=2)
    up_4 = Concatenate(name="up_5_Concatenate")([down_1, up_4])
    up_5 = Conv2D(nb_filters, 3, padding="same", name="up_5_block1_conv1")(up_4)
    classifier = BatchNormalization(name="classifier_bn1")(up_5)
    classifier = Activation("relu")(classifier)
    classifier = Conv2D(nb_filters, 3, padding="same", name="classifier_conv1")(classifier)
    o = Conv2D(output_channels, 1, activation="softmax", name="conv_out")(classifier)
    return Model(inputs=i, outputs=o)

