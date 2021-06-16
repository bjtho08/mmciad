"""U-Net model implementation with keras"""

from tensorflow.keras import Model

# from keras.layers.advanced_activations import LeakyReLU
# from keras.activations import relu

# from keras_contrib.layers.advanced_activations import swish
from tensorflow.keras.layers import (
    add,
    Layer,
    Input,
    ReLU,
    # Activation,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    GaussianNoise,
    Dropout,
    BatchNormalization,
)
from tensorflow.python.keras.layers.core import Dense, Flatten
from tf_mmciad.model.crfrnnlayer import CrfRnnLayer
from tf_mmciad.model.layers import MyConv2D


def _shortcut(input_: Layer, residual: Layer):
    # input_shape = K.int_shape(input_)
    residual_shape = residual.shape
    # stride_width = int(round(input_shape[1] / residual_shape[1]))
    # stride_height = int(round(input_shape[2] / residual_shape[2]))
    # equal_channels = input_shape[3] == residual_shape[3]
    shortcut = input_
    sc_base_name = "_".join(residual.name.split("_")[:2])
    # if stride_width > 1 or stride_height > 1 or not equal_channels:
    if "_d" in residual.name or "_bottom" in residual.name:
        shortcut = Conv2D(
            filters=residual_shape[3],
            kernel_size=(1, 1),
            strides=(2, 2),
            padding="same",
            kernel_initializer="he_normal",
            name=f"shortcut_{sc_base_name}",
        )(shortcut)
    if "_u" in residual.name:
        shortcut = Conv2D(
            filters=residual_shape[3],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            name=f"shortcut_{sc_base_name}",
        )(shortcut)
    return add([residual, shortcut], name=f"add_{sc_base_name}")


def batchnorm_activate(m, bn, level, acti, iter_):
    n = BatchNormalization(name=f"block{level}_bn{iter_}")(m) if bn else m
    try:
        n = acti(
            name=f"block{level}_{acti.__name__}{iter_}", trainable=True
        )(n)
    except TypeError:
        n = acti(name=f"block{level}_{acti.__name__}{iter_}")(n)
    return n


def bottleneck(m, nb_filters, conv_size, init, acti, bn, level, strides=1, do=0):
    n = batchnorm_activate(m, bn, level, acti, 1)
    n = Conv2D(
        filters=nb_filters,
        kernel_size=1,
        strides=strides,
        padding="same",
        kernel_initializer=init,
        name=f"block{level}_conv1",
    )(n)
    n = batchnorm_activate(n, bn, level, acti, 2)
    n = Conv2D(
        filters=nb_filters,
        kernel_size=conv_size,
        padding="same",
        kernel_initializer=init,
        name=f"block{level}_conv2",
    )(n)
    n = batchnorm_activate(n, bn, level, acti, 3)
    n = Conv2D(
        filters=nb_filters * 4,
        kernel_size=1,
        padding="same",
        kernel_initializer=init,
        name=f"block{level}_conv3",
    )(n)
    n = Dropout(do, name=f"block{level}_drop") if do else n
    return _shortcut(m, n)


def conv_block(m, nb_filters, conv_size, init, acti, bn, level, strides=None, do=0):
    _ = strides
    n = MyConv2D(
        nb_filters,
        conv_size,
        padding=(1, 1),
        kernel_initializer=init,
        name=f"block{level}_conv1",
    )(m)
    n = batchnorm_activate(n, bn, level, acti, 1)
    n = Dropout(do, name=f"block{level}_drop1")(n) if do else n
    n = MyConv2D(
        nb_filters,
        conv_size,
        padding=(1, 1),
        kernel_initializer=init,
        name=f"block{level}_conv2",
    )(n)
    n = batchnorm_activate(n, bn, level, acti, 2)
    return n


def level_block(
    m, nb_filters, conv_size, init, depth, inc, acti, do, bn, mp, up, level=1, res=False, enc_only=False,
):
    if res:
        block = bottleneck
    else:
        block = conv_block
    if depth > 0:
        n = block(
            m, nb_filters, conv_size, init, acti, bn, str(level) + "_d", strides=2
        )
        m = (
            MaxPooling2D(pool_size=(2, 2), name=f"block{level}_d_MaxPool")(n)
            if mp
            else MyConv2D(
                nb_filters,
                conv_size,
                strides=2,
                padding=(1, 1),
                kernel_initializer=init,
                name=f"block{level}_d_DownSamplingConv2D",
            )(n)
            if not res
            else n
        )
        m = Dropout(do, name=f"block{level}_d_drop2")(m) if do else m
        m = level_block(
            m,
            int(inc * nb_filters),
            conv_size,
            init,
            depth - 1,
            inc,
            acti,
            do,
            bn,
            mp,
            up,
            level + 1,
            res,
            enc_only,
        )
        if not enc_only:
            if up:
                m = UpSampling2D(size=(2, 2), name=f"block{level}_u_upsampling")(m)
            else:
                m = Conv2DTranspose(
                    nb_filters, 3, strides=2, padding="same", kernel_initializer=init
                )(m)
                m = acti(name=f"block{level}_{acti.__name__}1")(m)
            n = Concatenate(name=f"Concatenate_{depth}")([n, m])
            m = block(n, nb_filters, conv_size, init, acti, bn, str(level) + "_u")
    else:
        m = block(
            m, nb_filters, conv_size, init, acti, bn, str(level) + "_bottom", 2, do
        )
    return m


def u_net(
    shape,
    nb_filters=64,
    conv_size=3,
    initialization="glorot_uniform",
    depth=4,
    inc_rate=2.0,
    activation=ReLU,
    dropout=0,
    output_channels=5,
    batchnorm=True,
    maxpool=True,
    upconv=True,
    sigma_noise=0,
    arch="U-Net",
    crf=False,
    encode_only=False,
    **kwargs,
):
    """U-Net model.

    Standard U-Net model, plus optional gaussian noise, dropout,
    batchnorm and residual blocks.
    Note that the dimensions of the input images should be
    multiples of 16.

    Arguments:
    shape: image shape, in the format (nb_channels, x_size, y_size).
    nb_filters : initial number of filters in the convolutional layer.
    depth : The depth of the U-net, i.e. the number of contracting steps
            before expansion begins
    inc_rate : the multiplier for number of filters per layer
    conv_size : size of convolution.
    initialization: initialization of the convolutional layers.
    activation: activation of the convolutional layers.
    sigma_noise: standard deviation of the gaussian noise layer.
                 If equal to zero, this layer is deactivated.
    output_channels: number of output channels.
    drop: dropout rate

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015

    Credits:
    The starting point for the code of this function comes from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    by Marko Jocic
    """
    _ = kwargs
    resnet = False
    if arch.lower() not in ["u-resnet", "u-net"]:
        raise ValueError("Wrong architecture ")
    if arch.lower() == "u-resnet":
        resnet = True
    i = Input(shape, name="input_layer")
    m = (
        Conv2D(
            filters=nb_filters,
            kernel_size=conv_size,
            padding="same",
            kernel_initializer=initialization,
            name="pre_conv",
        )(i)
        if resnet
        else i
    )
    o = level_block(
        m,
        nb_filters,
        conv_size,
        initialization,
        depth,
        inc_rate,
        activation,
        dropout,
        batchnorm,
        maxpool,
        upconv,
        res=resnet,
        enc_only=encode_only,
    )
    if not encode_only:
        o = UpSampling2D(size=(2, 2), name="post_upsampling")(o) if resnet else o
        o = (
            Conv2D(
                filters=nb_filters,
                kernel_size=conv_size,
                padding="same",
                kernel_initializer=initialization,
                name="post_conv",
            )(o)
            if resnet
            else o
        )
        o = Concatenate(name="Concatenate_out")([o, m]) if resnet else o
        o = batchnorm_activate(o, batchnorm, "out", activation, 1) if resnet else o
        if sigma_noise > 0:
            o = GaussianNoise(sigma_noise, name="GaussianNoise_preout")(o)
        o = Conv2D(output_channels, 1, activation="softmax", name="conv_out")(o)
        o = (
            CrfRnnLayer(
                image_dims=shape[:2],
                num_classes=21,
                theta_alpha=160.0,
                theta_beta=3.0,
                theta_gamma=3.0,
                num_iterations=10,
                name="crfrnn",
            )([o, i])
            if crf
            else o
        )
    else:
        o = MaxPooling2D(pool_size=(2, 2))(o)
        o = Flatten()(o)
        o = Dense(4096, activation="relu", name="dense_1")(o)
        o = Dense(4096, activation="relu", name="dense_2")(o)
        o = Dense(output_channels, activation="softmax", name="dense_out")(o)
    modelname = (
        "BS" + arch
        if batchnorm and activation.__name__ == "Swish"
        else "B" + arch
        if batchnorm
        else arch
    )
    if encode_only:
        modelname += "encoding_path"
    return Model(inputs=i, outputs=o, name=modelname)
