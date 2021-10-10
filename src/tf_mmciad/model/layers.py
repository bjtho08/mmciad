from tensorflow.python.ops.gen_math_ops import floor as tffloor
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow import pad, TensorShape
from tensorflow.keras.layers import Layer, Conv2D
from typing import Tuple


class ReflectionPadding2D(Layer):
    """
    2D Reflection Padding
    Attributes:
        - padding: (padding_width, padding_height) tuple
    """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3],
        )

    def call(self, inputs, **kwargs):
        padding_width, padding_height = self.padding
        return pad(
            inputs,
            [
                [0, 0],
                [padding_height, padding_height],
                [padding_width, padding_width],
                [0, 0],
            ],
            "REFLECT",
        )
    
    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'dilation_rate':
                self.dilation_rate,
            'groups':
                self.groups,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MyConv2D(Conv2D):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding: Tuple[int, int]=(1, 1),
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            filters,
            kernel_size,
            strides=strides,
            padding="valid",
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        if not all([isinstance(p, int) for p in padding]):
            self.internal_padding = (1, 1)
        else:
            self.internal_padding = padding

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            self._conv_output_length(
                length,
                self.kernel_size[i],
                self.internal_padding[i],
                self.strides[i],
                dilation=self.dilation_rate[i],
            )
            for i, length in enumerate(spatial_input_shape)
        ]

    @staticmethod
    def _conv_output_length(input_length, filter_size, padding, stride, dilation=1):
        """Determines output length of a convolution given input length.
        Arguments:
            input_length: integer.
            filter_size: integer.
            padding: integer.
            stride: integer.
            dilation: dilation rate, integer.
        Returns:
            The output length (integer).
        """
        if input_length is None:
            return None
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
        output_length = input_length - dilated_filter_size + 2 * padding
        return tffloor(output_length / stride) + 1

    def compute_output_shape(self, input_shape):
        input_shape = TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == "channels_last":
            return TensorShape(
                input_shape[:batch_rank]
                + self._spatial_output_shape(input_shape[batch_rank:-1])
                + [self.filters]
            )
        else:
            return TensorShape(
                input_shape[:batch_rank]
                + [self.filters]
                + self._spatial_output_shape(input_shape[batch_rank + 1 :])
            )

    def call(self, inputs):
        padding_width, padding_height = self.internal_padding
        padded_tensor = pad(
            inputs,
            [
                [0, 0],
                [padding_height, padding_height],
                [padding_width, padding_width],
                [0, 0],
            ],
            "REFLECT",
        )
        return super().call(padded_tensor)

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'dilation_rate':
                self.dilation_rate,
            'groups':
                self.groups,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
