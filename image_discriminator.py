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

def create_image_discriminator(discrim_targets):
    layers = []

    # layer_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_conv1"):
        convolved = discrim_conv2d(discrim_targets, config.ndf, strides=(2,2))
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # discrim_img_num_conv_layers = 6
    for i in range(config.discrim_img_num_conv_layers-1):
        with tf.variable_scope("layer_conv%d" % (len(layers) + 1)):
            out_channels = config.ndf * min(2**(i+1), 8)
            convolved = discrim_conv2d(layers[-1], out_channels, strides=(2,2))
            # No BatchNorm in WGAN-GP critic
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

    # layer_5: fully connected [batch, 4, 4, ndf * 8] => [batch, 1,1,1]
    with tf.variable_scope("layer_fc%d" % (len(layers) + 1)):
        rinput = tf.reshape(rectified, [-1, 4*4*8*config.ndf])
        output = discrim_fc(rinput, out_channels=1)
        # there is no non-linearity
        layers.append(output)

    return tf.reshape(output,[-1])
