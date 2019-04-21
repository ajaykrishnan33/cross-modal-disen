import tensorflow as tf
import numpy as np
import os
import json
import glob
import random
import collections
import math
import time

from ops import *
from model import create_model

import os

import config

from data_ops import MSCOCODataset, TestDataset, Vocabulary
from tqdm import tqdm

CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "inputsI, inputsT")

def load_dataset():
    if config.input_dir is None or not os.path.exists(config.input_dir):
        raise Exception("input_dir does not exist")

    test_dataset = TestDataset(
        json_file="../mscoco/annotations/cross_modal_retrieval.json",
        inputs="image",
        input_dir="../mscoco/val2014/",
        batch_size=config.batch_size
    )

    inputsI = tf.placeholder(dtype=tf.float32, shape=(None, config.image_size, config.image_size, 3))
    inputsT = tf.placeholder(dtype=tf.int32, shape=(None, config.max_length))

    return Examples(
        inputsI = inputsI,
        inputsT = inputsT
    ), test_dataset


def main():
    if config.seed is None:
        config.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if config.checkpoint is None:
        raise Exception("checkpoint required for test mode")

    # load some options from the checkpoint
    options = {"ngfI", "ndfI"}
    with open(os.path.join(config.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(config.a, key, val)

    for k, v in config._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(config.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(config.a), sort_keys=True, indent=4))

    examples, dataset = load_dataset()

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputsI, examples.inputsT)
    
    sR_I2T = model.sR_I2T
    sR_T2I = model.sR_T2I
    eR_I2T = model.eR_I2T
    eR_T2I = model.eR_T2I

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = config.output_dir if (config.trace_freq > 0 or config.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with sv.managed_session(config=sess_config) as sess:
        print("parameter_count =", sess.run(parameter_count))
        
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(config.checkpoint)
        saver.restore(sess, checkpoint)


        # testing
        # at most, process the test data once
        start = time.time()
        correct_answers = 0
        total_qs = 0
        for k in tqdm(range(dataset.max_steps())):
            batch = dataset.next_batch()

            inputs = []
            choices = []

            total_qs += len(batch)

            for data_item in batch:
                inputs.append(data_item["processed_input"])
                choices.extend(data_item["processed_choice_list"])

            inputs = np.vstack((*inputs,))
            choices = np.vstack((*choices,))

            if config.which_direction == "AtoB":
                input_results = sess.run({
                    "shared": sR_I2T,
                    "exclusive": eR_I2T
                }, feed_dict={
                    examples.inputsI: inputs
                })
                
                choice_results = sess.run({
                    "shared": sR_T2I,
                    "exclusive": eR_T2I
                }, feed_dict={
                    examples.inputsT: choices
                })
            else:
                input_results = sess.run({
                    "shared": sR_T2I,
                    "exclusive": eR_T2I
                }, feed_dict={
                    examples.inputsT: inputs
                })
                
                choice_results = sess.run({
                    "shared": sR_I2T,
                    "exclusive": eR_I2T
                }, feed_dict={
                    examples.inputsI: choices
                })

            for i, q in enumerate(input_results["shared"]):
                min_dist = 2.0**32
                min_dist_id = -1
                for j, c in enumerate(choice_results["shared"][i*config.max_choices:(i+1)*config.max_choices]):
                    dist = np.linalg.norm(q-c)
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_id = j
                if min_dist_id == batch[i]["answer_id"]:
                    correct_answers += 1

    print("Correctly answered/Total Questions:{}/{}".format(correct_answers, total_qs))
    print("Percentage:{}".format(correct_answers*1.0/total_qs*100.0))


main()
