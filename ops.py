import tensorflow as tf
import config
from special_ops import conv1d_transpose_special

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

@tf.RegisterGradient("ReverseGrad")
def _reverse_grad(unused_op, grad):
    return -1.0*grad

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

def gen_conv1d(batch_input, out_channels, kernel_size=4, stride=2, padding="same"):
    # [batch, in_width, in_channels] => [batch, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv1d(
        batch_input, out_channels, kernel_size=kernel_size, strides=stride, padding=padding, 
        kernel_initializer=initializer
    )

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

def gen_deconv1d(batch_input, out_channels, out_width, kernel_size=4, stride=2, padding="same"):
    # [batch, in_width, in_channels] => [batch, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)

    return conv1d_transpose_special(
        batch_input, out_channels, out_width=out_width, kernel_size=kernel_size,
        stride=stride, padding=padding, kernel_initializer=initializer
    )

def gen_deconv2d(batch_input, out_channels, kernel_size=4, strides=(2,2)):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(
        batch_input, out_channels, kernel_size=kernel_size, strides=strides, padding="same", 
        kernel_initializer=initializer
    )

def discrim_conv2d(batch_input, out_channels, kernel_size=4, strides=(2,2)):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(
        batch_input, out_channels, kernel_size=kernel_size, strides=strides, padding="same", 
        kernel_initializer=initializer
    )

def discrim_conv1d(batch_input, out_channels, kernel_size=4, stride=2, padding="same"):
    # [batch, in_width, in_channels] => [batch, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv1d(
        batch_input, out_channels, kernel_size=kernel_size, strides=stride, padding=padding, 
        kernel_initializer=initializer
    )

def discrim_fc(batch_input, out_channels=1):
    # With no initializer argument, it uses glorot
    return tf.layers.dense(batch_input, out_channels)

def gen_embedder(batch_input, vocab_size, out_channels, max_length):
    embedder = tf.keras.layers.Embedding(
        vocab_size, out_channels, input_length=max_length,
        embeddings_initializer=tf.random_normal_initializer(0, 0.02)
    )

    return embedder(batch_input)

def create_text_embedder(text):
    # text: [batch, max_length, vocab_size]

    # encoder_embedding: [batch, max_length, vocab_size] => [batch, max_length, wrl]
    with tf.name_scope("text_embedding", values=[text]):
        z = tf.reshape(text, [-1, config.vocab_size])
        encoded_text = gen_embedder(z, config.vocab_size, config.wrl, config.max_length)
        encoded_text = tf.reshape(encoded_text, [-1, config.max_length, config.wrl])

    return encoded_text

def create_text_deembedder(text):
    # text_deembedding: [batch, max_length, wrl] => [batch, max_length, vocab_size]
    with tf.name_scope("text_deembedding", values=[text]):
        z = tf.reshape(text, [-1, config.wrl])
        decoded_text = gen_fc(z, config.vocab_size)
        decoded_text = tf.reshape(decoded_text, [-1, config.max_length, config.vocab_size])

    # no softmax here

    return decoded_text
