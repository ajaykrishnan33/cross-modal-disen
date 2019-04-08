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

def create_full_image_decoder(sR, eR, generator_outputs_channels, noise = True):

    batch_size = sR.shape[0]

    ex_size = eR.shape[1]
    sh_size = sR.shape[1]

    initial_input = tf.concat([sR, eR], axis=1) # 1024 + 256 = 1280

    if config.mode == "train":
        if noise:
            inoise = tf.random_normal(tf.shape(initial_input), mean=0.0, stddev=config.noise)
            initial_input += inoise


    with tf.variable_scope("decoder_fc1"):
        fc1_output = gen_fc(initial_input, output_channels=8192) #4096
        rectified = lrelu(fc1_output, 0.2)
        fc1_bn = batchnorm(rectified, axis=1)

    # with tf.variable_scope("decoder_fc2"):
    #     fc2_output = gen_fc(fc1_bin, output_channels=8192)
    #     rectified = lrelu(fc2_output, 0.2)
    #     fc2_bn = batchnorm(rectified, axis=1)

    z = tf.reshape(fc1_bn, [-1, 4, 4, config.ngfI*8]) # fc2_bn

    # Add noise only at train time
    if config.mode == "train":
        layer_specs = [
            (config.ngf * 8, 0.5, 4, (2,2)),   # decoder_conv6: [batch, 4, 4, ngf * 8] => [batch, 8, 8, ngf * 8]
            (config.ngf * 8, 0.5, 4, (2,2)),   # decoder_conv5: [batch, 8, 8, ngf * 8] => [batch, 16, 16, ngf * 8]
            (config.ngf * 4, 0.5, 4, (2,2)),   # decoder_conv4: [batch, 16, 16, ngf * 8] => [batch, 32, 32, ngf * 4]
            (config.ngf * 2, 0.0, 4, (2,2)),   # decoder_conv3: [batch, 32, 32, ngf * 4] => [batch, 64, 64, ngf * 2]
            (config.ngf, 0.0, 4, (2,2)),       # decoder_conv2: [batch, 64, 64, ngf * 2] => [batch, 128, 128, ngf]
        ]
    else:
        layer_specs = [
            (config.ngf * 8, 0.0, 4, (2,2)),   # decoder_conv6: [batch, 4, 4, ngf * 8] => [batch, 8, 8, ngf * 8]
            (config.ngf * 8, 0.0, 4, (2,2)),   # decoder_conv5: [batch, 8, 8, ngf * 8] => [batch, 16, 16, ngf * 8]
            (config.ngf * 4, 0.0, 4, (2,2)),   # decoder_conv4: [batch, 16, 16, ngf * 8] => [batch, 32, 32, ngf * 4]
            (config.ngf * 2, 0.0, 4, (2,2)),   # decoder_conv3: [batch, 32, 32, ngf * 4] => [batch, 64, 64, ngf * 2]
            (config.ngf, 0.0, 4, (2,2)),       # decoder_conv2: [batch, 64, 64, ngf * 2] => [batch, 128, 128, ngf]
        ]

    num_encoder_layers = 6
    layers =[]

    for decoder_layer, (out_channels, dropout, kernel_size, strides) in enumerate(layer_specs):
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
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv2d(rectified, out_channels, kernel_size=kernel_size, strides=strides)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_conv1"):
        # No skip connections
        input = layers[-1]
        rectified = tf.nn.relu(input)
        output = gen_deconv2d(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

