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

from data_ops import vocabulary

def create_exclusive_text_decoder(embedded_text, eR):

    initial_input = eR

    with tf.variable_scope("decoder_fc1"):
        # Add GRL right after the input
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": "ReverseGrad"}):
            initial_input = tf.identity(initial_input)
        fc1_output = gen_fc(initial_input, out_channels=300) #4096
        rectified = tf.nn.relu(fc1_output)
        fc1_bn = batchnorm(rectified, axis=1)

    with tf.variable_scope("decoder_gru"):
        gru = tf.keras.layers.GRU(300, return_sequences=True, unroll=True)
        outputs = gru(embedded_text, initial_state=fc1_bn)

    result = tf.concat((embedded_text[:,0:1,:], outputs[:,:-1,:]), axis=1)

    return result