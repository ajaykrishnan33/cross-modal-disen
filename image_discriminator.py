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

def create_image_discriminator(discrim_targets, n_layers=3):
        layers = []

        # layer_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv2d(discrim_targets, config.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = config.ndf * min(2**(i+1), 8)
                stride = 2
                convolved = discrim_conv2d(layers[-1], out_channels, stride=stride)
                # No BatchNorm in WGAN-GP critic
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)

        # layer_4: fully connected [batch, 4, 4, ndf * 4] => [batch, 1,1,1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            rinput = tf.reshape(rectified, [-1, 4*4*4*config.ndf])
            output = discrim_fc(rinput, out_channels=1)
            # there is no non-linearity
            layers.append(output)

        return tf.reshape(output,[-1])
