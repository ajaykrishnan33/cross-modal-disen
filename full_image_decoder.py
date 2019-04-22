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

    initial_input = tf.concat([sR, eR], axis=1) # 300 + 100 = 400

    if config.mode == "train":
        if noise:
            inoise = tf.random_normal(tf.shape(initial_input), mean=0.0, stddev=config.noise)
            initial_input += inoise


    with tf.variable_scope("decoder_fc1"):
        fc1_output = gen_fc(initial_input, out_channels=4096) #4096
        rectified = lrelu(fc1_output, 0.2)
        fc1_bn = batchnorm(rectified, axis=1)

    with tf.variable_scope("decoder_fc2"):
        fc2_output = gen_fc(fc1_bn, out_channels=4096) #4096
        rectified = lrelu(fc2_output, 0.2)
        fc2_bn = batchnorm(rectified, axis=1)

    return fc2_bn
