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

def create_text_discriminator(embedded_text):

    with tf.variable_scope("layer_gru"):
        gru = tf.keras.layers.GRU(300, return_state=True)
        _, encoded_text = gru(embedded_text)

    # layer_4: fully connected [batch, 300] => [batch, 1]
    with tf.variable_scope("layer_fc1"):
        output = discrim_fc(rinput, out_channels=1)

    return tf.reshape(output,[-1])
