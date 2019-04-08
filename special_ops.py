import tensorflow as tf
# from tensorflow.python.ops.nn_ops import *
import math

def conv1d_transpose_special(
        value,
        out_channels,
        out_width,
        kernel_size,
        stride,
        padding="SAME",
        kernel_initializer=None,
        name=None):
    """The transpose of `conv1d`.
    This operation is sometimes called "deconvolution" after [Deconvolutional
    Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf), but is
    actually the transpose (gradient) of `conv1d` rather than an actual
    deconvolution.
    Args:
      value: A 3-D `Tensor` of type `float` and shape
        `[batch, in_width, in_channels]`
      stride: An `integer`.  The number of entries by which
        the filter is moved right at each step.
      padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
        See the "returns" section of `tf.nn.convolution` for details.
      name: Optional name for the returned tensor.
    Returns:
      A `Tensor` with the same type as `value`.
    Raises:
      ValueError: If input/output depth does not match `filter`'s shape, or if
        padding is other than `'VALID'` or `'SAME'`.
    """
    with tf.name_scope(name, "conv1d_transpose", [value]) as name:
        with tf.variable_scope(name, "conv1d_transpose", [value]):
            # output_shape_ = ops.convert_to_tensor(
            #     output_shape, name="output_shape")
            # if not output_shape_.get_shape().is_compatible_with(tensor_shape.vector(3)):
            #     raise ValueError("output_shape must have shape (3,), got {}".format(
            #         output_shape_.get_shape()))

            # The format could be either NWC or NCW, map to NHWC or NCHW
            data_format_2d = "NHWC"
            axis = 2

            filter = tf.Variable(
                initial_value=kernel_initializer([kernel_size, value.get_shape().dims[axis]], dtype=tf.float32)
            )

            if not value.get_shape().dims[axis].is_compatible_with(
                    filter.get_shape()[-1]):
                raise ValueError("input channels does not match filter's input channels, "
                                 "{} != {}".format(value.get_shape()[axis],
                                                   filter.get_shape()[-1]))

            # if isinstance(output_shape, (list, np.ndarray)):
            #     # output_shape's shape should be == [3] if reached this point.
            #     if not filter.get_shape().dims[1].is_compatible_with(
            #             output_shape[axis]):
            #         raise ValueError(
            #             "output_shape does not match filter's output channels, "
            #             "{} != {}".format(output_shape[axis],
            #                               filter.get_shape()[1]))

            if padding != "VALID" and padding != "SAME":
                raise ValueError("padding must be either VALID or SAME:"
                                 " {}".format(padding))

            in_width = value.shape[1]
            if padding == "valid":
                in_width_c = math.ceil((out_width - kernel_size + 1)/stride)
                if (in_width != in_width_c):
                    raise ValueError("out_width: {}, kernel_size: {} and in_width: {} are not compatible with computed in_width_c: {}".format(
                        out_width, kernel_size, in_width, in_width_c
                    ))
            elif padding == "same":
                in_width_c = math.ceil(out_width/stride)
                if (in_width != in_width_c):
                    raise ValueError("out_width: {}, kernel_size: {} and in_width: {} are not compatible with computed in_width_c: {}".format(
                        out_width, kernel_size, in_width, in_width_c
                    ))

            batch_size = tf.shape(value)[0]

            output_shape = tf.concat(
                [[batch_size], [1], [out_width, out_channels]], axis=0)
            spatial_start_dim = 1
            strides = [1, 1, stride, 1]
            
            value = tf.expand_dims(value, spatial_start_dim)
            filter = tf.expand_dims(filter, 0)  # pylint: disable=redefined-builtin

            result = tf.nn.conv2d_backprop_input(
                input_sizes=output_shape,
                filter=filter,
                out_backprop=value,
                strides=strides,
                padding=padding,
                data_format=data_format_2d,
                name=name)
            return tf.squeeze(result, [spatial_start_dim])
