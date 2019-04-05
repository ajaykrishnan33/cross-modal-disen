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

def create_text_discriminator(embeddded_text):
    layers = []

    # layer_1: [batch, max_length, wrl] => [batch, max_length/2, wrl*2]
    with tf.variable_scope("layer_conv1"):
        convolved = discrim_conv1d(embedded_text, config.wrl*2, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)


    # discrim_txt_num_conv_layers = 5
    for i in range(config.discrim_txt_num_conv_layers-1):
        with tf.variable_scope("layer_conv%d" % (len(layers) + 1)):
            out_channels = config.wrl * min(2**(i+2), 32)
            convolved = discrim_conv1d(layers[-1], out_channels, stride=2)
            # No BatchNorm in WGAN-GP critic
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

    # layer_4: fully connected [batch, 1, wrl * 32] => [batch, 1,1]
    with tf.variable_scope("layer_fc%d" % (len(layers) + 1)):
        rinput = tf.reshape(rectified, [-1, 32*config.wrl])
        output = discrim_fc(rinput, out_channels=1)
        # there is no non-linearity
        layers.append(output)

    return tf.reshape(output,[-1])
