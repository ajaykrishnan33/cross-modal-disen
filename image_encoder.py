import tensorflow as tf
import config

from ops import *

def create_image_encoder(image):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngfI]
    with tf.variable_scope("encoder_conv1"):
       output = gen_conv2d(image, config.ngfI)
       layers.append(output)

    conv_layer_specs = [
        (config.ngfI * 2, 4, (2,2)), # encoder_2: [batch, 128, 128, ngfI] => [batch, 64, 64, ngfI*2]
        (config.ngfI * 4, 4, (2,2)), # encoder_3: [batch, 64, 64, ngfI*2] => [batch, 32, 32, ngfI*4]
        (config.ngfI * 8, 4, (2,2)), # encoder_4: [batch, 32, 32, ngfI*4] => [batch, 16, 16, ngfI*8]
        (config.ngfI * 8, 4, (2,2)), # encoder_5: [batch, 16, 16, ngfI*8] => [batch, 8, 8, ngfI*8]
        (config.ngfI * 8, 4, (2,2)), # encoder_6: [batch, 8, 8, ngfI*8] => [batch, 4, 4, ngfI*8]
    ]

    for out_channels, kernel_size, strides in conv_layer_specs:
        with tf.variable_scope("encoder_conv%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv2d(rectified, out_channels, kernel_size, strides)
            output = batchnorm(convolved)
            layers.append(output)

    # Shared part of the representation
    with tf.variable_scope("encoder_shared_fc1"):
        rectified = lrelu(layers[-1], 0.2)
        fc1_input = tf.reshape(rectified, [-1, 4*4*8*config.ngfI])
        fc1_output = gen_fc(fc1_input, out_channels=1024) #4096
        fc1_bn = batchnorm(fc1_output, axis=1)

    # with tf.variable_scope("encoder_shared_fc2"):
    #     rectified = lrelu(fc1_bn, 0.2)
    #     fc2_output = gen_fc(rectified, out_channels=1024)
    #     fc2_bn = batchnorm(fc2_output, axis=1)

    # Exclusive part of the representation
    with tf.variable_scope("encoder_exclusive_fc1"):
        rectified = lrelu(layers[-1], 0.2)
        efc1_input = tf.reshape(rectified, [-1, 4*4*8*config.ngfI])
        efc1_output = gen_fc(efc1_input, out_channels=256) #4096
        efc1_bn = batchnorm(efc1_output, axis=1)

    # with tf.variable_scope("encoder_exclusive_fc2"):
    #     rectified = lrelu(efc1_bn, 0.2)
    #     efc2_output = gen_fc(rectified, out_channels=256)
    #     efc2_bn = batchnorm(efc2_output, axis=1)


    sR = fc1_bn #fc2_bn
    eR = efc1_bn #efc2_bn


    return sR, eR

