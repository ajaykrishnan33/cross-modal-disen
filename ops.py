import tensorflow as tf
import config

def batchnorm(inputs, axis=3):
    return tf.layers.batch_normalization(
        inputs, axis=axis, epsilon=1e-5, momentum=0.1, training=True, 
        gamma_initializer=tf.random_normal_initializer(1.0, 0.02)
    )

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def gen_conv2d(batch_input, out_channels, kernel_size=4, strides=(2,2)):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(
        batch_input, out_channels, kernel_size=kernel_size, strides=strides, padding="same", 
        kernel_initializer=initializer
    )

def gen_fc(batch_input, out_channels=8):
    # With no initializer argument, it uses glorot
   return tf.layers.dense(batch_input, out_channels)

def gen_deconv2d(batch_input, out_channels, kernel_size=4, strides=(2,2)):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(
        batch_input, out_channels, kernel_size=kernel_size, strides=strides, padding="same", 
        kernel_initializer=initializer
    )
