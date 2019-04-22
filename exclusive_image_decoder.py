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
        fc1_output = gen_fc(initial_input, out_channels=4096) #4096
        rectified = lrelu(fc1_output, 0.2)
        fc1_bn = batchnorm(rectified, axis=1)

    with tf.variable_scope("decoder_fc2"):
        fc2_output = gen_fc(fc1_bn, out_channels=4096) #4096
        rectified = lrelu(fc2_output, 0.2)
        fc2_bn = batchnorm(rectified, axis=1)

    return fc2_bn