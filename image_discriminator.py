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
    # # layer_1: fully connected [batch, 4096] => [batch, 1024]
    with tf.variable_scope("img_discrim_fc1"):
        fc1_output = discrim_fc(discrim_targets, out_channels=1024)

    with tf.variable_scope("img_discrim_fc2"):
        output = discrim_fc(fc1_output, out_channels=1)

    return tf.reshape(output,[-1])
