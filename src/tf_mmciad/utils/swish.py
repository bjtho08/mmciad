import tensorflow as tf
from tensorflow.keras.layers import Layer

class Swish(Layer):
    """Creates a Swish layer.

    Args:
        Layer ([type]): [description]
    """
    def __init__(self, beta=1.0, trainable=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    # pylint: disable=attribute-defined-outside-init
    def build(self, input_shape):
        self.beta_factor = tf.Variable(self.beta,
                                      dtype=tf.float32,
                                      name='beta_factor',
                                      trainable=self.trainable)
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Swish, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * tf.sigmoid(self.beta_factor * inputs)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
