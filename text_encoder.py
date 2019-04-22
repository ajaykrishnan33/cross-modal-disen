import tensorflow as tf
import config

from ops import *

def create_text_encoder(embedded_text):

    # encoded_text: [batch, max_length, config.wrl]

    with tf.variable_scope("encoder_gru"):
        gru = tf.keras.layers.GRU(300, return_state=True)
        _, encoded_text = gru(embedded_text)

    # Shared part of the representation
    with tf.variable_scope("encoder_shared_fc1"):
        fc1_input = encoded_text
        fc1_output = gen_fc(fc1_input, out_channels=300) #4096
        fc1_bn = batchnorm(fc1_output, axis=1)

    # Exclusive part of the representation
    with tf.variable_scope("encoder_exclusive_fc1"):
        efc1_input = encoded_text
        efc1_output = gen_fc(efc1_input, out_channels=100) #4096
        efc1_bn = batchnorm(efc1_output, axis=1)


    sR = fc1_bn #fc2_bn
    eR = efc1_bn #efc2_bn

    return sR, eR
