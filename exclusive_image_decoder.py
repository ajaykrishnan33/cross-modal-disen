import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

import config
from ops import *

def create_exclusive_image_decoder(eR, generator_outputs_channels):

    initial_input = eR

    with tf.variable_scope("decoder_fc1"):
        # Add GRL right after the input
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": "ReverseGrad"}):
            initial_input = tf.identity(initial_input)
        fc1_output = gen_fc(initial_input, out_channels=8192) #4096
        rectified = lrelu(fc1_output, 0.2)
        fc1_bn = batchnorm(rectified, axis=1)

    # with tf.variable_scope("decoder_fc2"):
    #     fc2_output = gen_fc(fc1_bin, out_channels=8192)
    #     rectified = lrelu(fc2_output, 0.2)
    #     fc2_bn = batchnorm(rectified, axis=1)

    z = tf.reshape(fc1_bn, [-1, 4, 4, config.ngfI*8]) # fc2_bn

    layer_specs = [
        (config.ngfI * 8, 0.5, 4, (2,2)),   # decoder_conv6: [batch, 4, 4, ngf * 8] => [batch, 8, 8, ngf * 8]
        (config.ngfI * 8, 0.5, 4, (2,2)),   # decoder_conv5: [batch, 8, 8, ngf * 8] => [batch, 16, 16, ngf * 8]
        (config.ngfI * 4, 0.5, 4, (2,2)),   # decoder_conv4: [batch, 16, 16, ngf * 8] => [batch, 32, 32, ngf * 4]
        (config.ngfI * 2, 0.0, 4, (2,2)),   # decoder_conv3: [batch, 32, 32, ngf * 4] => [batch, 64, 64, ngf * 2]
        (config.ngfI, 0.0, 4, (2,2)),       # decoder_conv2: [batch, 64, 64, ngf * 2] => [batch, 128, 128, ngf]
    ]

    num_encoder_layers = 6
    layers = []

    for decoder_layer, (out_channels, dropout, kernel_size, strides) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_conv%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                input = z
            else:
                input = layers[-1]

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv2d(rectified, out_channels, kernel_size=kernel_size, strides=strides)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_conv1"):
        input = layers[-1]
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels, a)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

