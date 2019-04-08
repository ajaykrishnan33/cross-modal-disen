import tensorflow as tf
import config

from ops import *

def create_text_encoder(embedded_text):

    # encoded_text: [batch, max_length, config.wrl]

    layers = []
    # encoder_1: [batch, max_length, config.wrl] => [batch, max_length/2, config.wrl*2]
    with tf.variable_scope("encoder_conv1"):
       output = gen_conv1d(embedded_text, config.wrl*2)
       layers.append(output)

    # max_length = 32
    # config.wrl = 256
    conv_layer_specs = [
        (config.wrl * 4, 4, 2, "same"), # encoder_2: [batch, max_length/2, config.wrl*2] => [batch, max_length/4, config.wrl*4]
        (config.wrl * 8, 4, 2, "same"), # encoder_3: [batch, max_length/4, config.wrl*4] => [batch, max_length/8, config.wrl*8]
        (config.wrl * 16, 4, 2, "same"), # encoder_4: [batch, max_length/8, config.wrl*8] => [batch, max_length/16, config.wrl*16]
        (config.wrl * 32, 2, 1, "valid"), # encoder_5: [batch, max_length/16, config.wrl*16] => [batch, max_length/32, config.wrl*32]
    ]

    for out_channels, kernel_size, stride, padding in conv_layer_specs:
        with tf.variable_scope("encoder_conv%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv1d(rectified, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            output = batchnorm(convolved, axis=2)
            layers.append(output)

    # Shared part of the representation
    with tf.variable_scope("encoder_shared_fc1"):
        rectified = lrelu(layers[-1], 0.2)
        fc1_input = tf.reshape(rectified, [-1, config.wrl*32])
        fc1_output = gen_fc(fc1_input, out_channels=1024) #4096
        fc1_bn = batchnorm(fc1_output, axis=1)

    # with tf.variable_scope("encoder_shared_fc2"):
    #     rectified = lrelu(fc1_bn, 0.2)
    #     fc2_output = gen_fc(rectified, out_channels=1024)
    #     fc2_bn = batchnorm(fc2_output, axis=1)

    # Exclusive part of the representation
    with tf.variable_scope("encoder_exclusive_fc1"):
        rectified = lrelu(layers[-1], 0.2)
        efc1_input = tf.reshape(rectified, [-1, config.wrl*32])
        efc1_output = gen_fc(efc1_input, out_channels=256) #4096
        efc1_bn = batchnorm(efc1_output, axis=1)

    # with tf.variable_scope("encoder_exclusive_fc2"):
    #     rectified = lrelu(efc1_bn, 0.2)
    #     efc2_output = gen_fc(rectified, out_channels=256)
    #     efc2_bn = batchnorm(efc2_output, axis=1)


    sR = fc1_bn #fc2_bn
    eR = efc1_bn #efc2_bn


    return sR, eR

