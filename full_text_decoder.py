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

def create_full_text_decoder(embedded_text, sR, eR, noise = True):

    batch_size = sR.shape[0]

    ex_size = eR.shape[1]
    sh_size = sR.shape[1]

    initial_input = tf.concat([sR, eR], axis=1) # 300 + 100 = 400

    if config.mode == "train":
        if noise:
            inoise = tf.random_normal(tf.shape(initial_input), mean=0.0, stddev=config.noise)
            initial_input += inoise


    with tf.variable_scope("decoder_fc1"):
        fc1_output = gen_fc(initial_input, out_channels=300) #4096
        rectified = tf.nn.relu(fc1_output)
        fc1_bn = batchnorm(rectified, axis=1)

    with tf.variable_scope("decoder_gru"):
        gru = tf.keras.layers.GRU(300, return_sequences=True, unroll=True)
        outputs = gru(embedded_text, initial_state=fc1_bn)

    return outputs
