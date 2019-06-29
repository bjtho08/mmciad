"""U-Net model implementation with keras"""

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, concatenate, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, GaussianNoise, Dropout, BatchNormalization

def conv_block(m, nb_filters, conv_size, init, acti, bn, level, do=0):
    n = Conv2D(nb_filters, conv_size, activation=acti, padding='same', kernel_initializer=init, name="block{}_conv1".format(level))(m)
    n = BatchNormalization(name="block{}_bn1".format(level))(n) if bn else n
    n = Dropout(do, name="block{}_drop1".format(level))(n) if do else n
    n = Conv2D(nb_filters, conv_size, activation=acti, padding='same', kernel_initializer=init, name="block{}_conv2".format(level))(n)
    n = BatchNormalization(name="block{}_bn2".format(level))(n) if bn else n
    return n

def level_block(m, nb_filters, conv_size, init, depth, inc, acti, do, bn, mp, up, level=1):
    if depth > 0:
        n = conv_block(m, nb_filters, conv_size, init, acti, bn, level)
        m = MaxPooling2D(pool_size=(2, 2), name="block{}_MaxPool".format(level))(n) if mp else Conv2D(nb_filters, conv_size, strides=2, padding='same', kernel_initializer=init)(n)
        m = Dropout(do, name="block{}_drop2".format(level))(m) if do else m
        m = level_block(m, int(inc*nb_filters), conv_size, init, depth-1, inc, acti, do, bn, mp, up, level+1)
        if up:
            m = UpSampling2D(size=(2, 2), name="block{}_upsampling".format(level))(m)
        else:
            m = Conv2DTranspose(nb_filters, 3, strides=2, activation=acti, padding='same', kernel_initializer=init)(m)
        n = Concatenate(name="Concatenate_{}".format(depth))([n, m])
        m = conv_block(n, nb_filters, conv_size, init, acti, bn, level+depth*2)
    else:
        m = conv_block(m, nb_filters, conv_size, init, acti, bn, level, do)
    return m


def u_net(shape, nb_filters=64, conv_size=3, initialization="glorot_uniform", depth=4, inc_rate=2., activation='relu',
            dropout=0, output_channels=5, batchnorm=False, maxpool=True, upconv=True, pretrain=0, sigma_noise=0):
    """U-Net model.

    Standard U-Net model, plus optional gaussian noise.
    Note that the dimensions of the input images should be
    multiples of 16.

    Arguments:
    shape: image shape, in the format (nb_channels, x_size, y_size).
    nb_filters : initial number of filters in the convolutional layer.
    depth : The depth of the U-net, i.e. the number of contracting steps before expansion begins
    inc_rate : the multiplier for number of filters per layer
    conv_size : size of convolution.
    initialization: initialization of the convolutional layers.
    activation: activation of the convolutional layers.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
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
    i = Input(shape, name='input_layer')
    o = level_block(i, nb_filters, conv_size, initialization, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv)
    if sigma_noise > 0:
        o = GaussianNoise(sigma_noise, name="GaussianNoise_preout")(o)
    o = Conv2D(output_channels, 1, activation='softmax', name="conv_out")(o)
    if pretrain > 0:
        pretrained_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=shape, pooling='max')
        w = []
        pretrain_layers = ['block{}_conv{}'.format(block, layer) for block in range(1,pretrain+1) for layer in range(1,3)]
        for n in pretrain_layers:
            w.append(pretrained_model.get_layer(name=n).get_weights())
        del pretrained_model
        new_model = Model(inputs=i, outputs=o)
        for i, n in enumerate(pretrain_layers):
            new_model.get_layer(name=n).set_weights(w[i])
        return new_model
    return Model(inputs=i, outputs=o)