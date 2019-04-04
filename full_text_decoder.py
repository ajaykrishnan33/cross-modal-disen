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

from ops import *
import config

def create_full_text_decoder(sR, eR, generator_outputs_channels, noise = True):

    batch_size = sR.shape[0]

    ex_size = eR.shape[1]
    sh_size = sR.shape[1]

    initial_input = tf.concat([sR, eR], axis=1) # 1024 + 256 = 1280

    if config.mode == "train":
        if noise:
            inoise = tf.random_normal(initial_input.shape, mean=0.0, stddev=a.noise)
            initial_input += inoise


    with tf.variable_scope("decoder_fc1"):
        fc1_output = gen_fc(initial_input, output_channels=8192) #4096
        rectified = tf.nn.relu(fc1_output, 0.2)
        fc1_bn = batchnorm(rectified, axis=1)

    # with tf.variable_scope("decoder_fc2"):
    #     fc2_output = gen_fc(fc1_bin, output_channels=8192)
    #     rectified = tf.nn.relu(fc2_output, 0.2)
    #     fc2_bn = batchnorm(rectified, axis=1)

    z = tf.reshape(fc1_bn, [-1, 1, config.wrl*32]) # fc2_bn

    # Add noise only at train time
    if config.mode == "train":
        layer_specs = [
            (config.wrl * 16, 0.5, 2, 1, "valid", 2),   # decoder_conv5: [batch, 1, wrl * 32] => [batch, 2, wrl * 16]
            (config.wrl * 8, 0.5, 4, 2, "same", 4),   # decoder_conv4: [batch, 2, wrl * 16] => [batch, 4, wrl * 8]
            (config.wrl * 4, 0.5, 4, 2, "same", 8),   # decoder_conv3: [batch, 4, wrl * 8] => [batch, 8, wrl * 4]
            (config.wrl * 2, 0.5, 4, 2, "same", 16),   # decoder_conv2: [batch, 8, wrl * 4] => [batch, 16, wrl * 2]
            # (config.wrl, 0.0, 4, 2, "same", 32),       # decoder_conv1: [batch, 16, wrl * 2] => [batch, 32, wrl]
        ]
    else:
        layer_specs = [
            (config.wrl * 16, 0.0, 2, 1, "valid", 2),   # decoder_conv5: [batch, 1, wrl * 32] => [batch, 2, wrl * 16]
            (config.wrl * 8, 0.0, 4, 2, "same", 4),   # decoder_conv4: [batch, 2, wrl * 16] => [batch, 4, wrl * 8]
            (config.wrl * 4, 0.0, 4, 2, "same", 8),   # decoder_conv3: [batch, 4, wrl * 8] => [batch, 8, wrl * 4]
            (config.wrl * 2, 0.0, 4, 2, "same", 16),   # decoder_conv2: [batch, 8, wrl * 4] => [batch, 16, wrl * 2]
            # (config.wrl, 0.0, 4, 2, "same", 32),       # decoder_conv1: [batch, 16, wrl * 2] => [batch, 32, wrl]
        ]

    num_encoder_layers = 5
    layers =[]

    for decoder_layer, (out_channels, dropout, kernel_size, stride, padding, out_width) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_conv%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                # Use here combination of shared and exclusive
                input = z
            else:
                input = layers[-1]

            rectified = tf.nn.relu(input)

            output = gen_deconv1d(
                rectified, out_channels, out_width, kernel_size=kernel_size, 
                strides=stride, padding=padding
            )
            output = batchnorm(output, axis=2)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 16, wrl*2] => [batch, 32, wrl]
    with tf.variable_scope("decoder_conv1"):
        # No skip connections
        input = layers[-1]
        rectified = tf.nn.relu(input)
        output = gen_deconv1d(
            rectified, generator_outputs_channels, out_width=32
        )
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]
