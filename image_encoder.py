import tensorflow as tf
import config

from ops import *

def create_image_encoder(image):

    # Shared part of the representation
    with tf.variable_scope("encoder_shared_fc1"):
        fc1_input = image
        fc1_output = gen_fc(fc1_input, out_channels=4096) #4096
        fc1_bn = batchnorm(fc1_output, axis=1)

    with tf.variable_scope("encoder_shared_fc2"):
        fc2_input = fc1_bn
        fc2_output = gen_fc(fc2_input, out_channels=300)
        fc2_bn = batchnorm(fc2_output, axis=1)

    # Exclusive part of the representation
    with tf.variable_scope("encoder_exclusive_fc1"):
        efc1_input = image
        efc1_output = gen_fc(efc1_input, out_channels=4096) #4096
        efc1_bn = batchnorm(efc1_output, axis=1)

    with tf.variable_scope("encoder_exclusive_fc2"):
        efc2_input = image
        efc2_output = gen_fc(efc2_input, out_channels=100) #4096
        efc2_bn = batchnorm(efc2_output, axis=1)

    sR = fc2_bn #fc2_bn
    eR = efc2_bn #efc2_bn

    return sR, eR
