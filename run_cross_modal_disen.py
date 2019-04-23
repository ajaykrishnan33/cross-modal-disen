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

from data_ops import MSCOCODataset, vocabulary
from tqdm import tqdm

Examples = collections.namedtuple("Examples", "inputsI, inputsT, masks, count, steps_per_epoch, dataset")

def load_examples():
    if config.input_dir is None or not os.path.exists(config.input_dir):
        raise Exception("input_dir does not exist")

    train_dataset = MSCOCODataset("train")
    inputsI, inputsT, masks = train_dataset.next_batch()
    count = train_dataset.total_size
    steps_per_epoch = int(math.ceil(train_dataset.total_size/config.batch_size))
    # inputsI_val, inputsT_val = MSCOCODataset("val").next_batch()

    return Examples(
        inputsI=inputsI,
        inputsT=inputsT,
        masks=masks,
        count=count,
        steps_per_epoch=steps_per_epoch,
        dataset=train_dataset
    )


def main():
    if config.seed is None:
        config.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    for k, v in config._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(config.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(config.a), sort_keys=True, indent=4))

    examples = load_examples()

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputsI, examples.inputsT, examples.masks)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = config.output_dir if (config.trace_freq > 0 or config.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with sv.managed_session(config=sess_config) as sess:
        print("parameter_count =", sess.run(parameter_count))
        print("Started session")
        if config.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(config.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if config.max_epochs is not None:
            max_steps = examples.steps_per_epoch * config.max_epochs
        if config.max_steps is not None:
            max_steps = config.max_steps

        print("Max steps: {}".format(max_steps))
        
        # training
        print("Starting training")
        start = time.time()

        sess.run(examples.dataset.iterator.initializer, feed_dict={
            examples.dataset.images_placeholder: examples.dataset.images,
            examples.dataset.captions_placeholder: examples.dataset.captions,
            examples.dataset.masks_placeholder: examples.dataset.masks
        })

        for step in tqdm(range(max_steps)):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            if should(config.trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()


            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }

            if should(config.progress_freq):
                fetches["discrimI2T_loss"] = model.discrimI2T_loss
                fetches["discrimT2I_loss"] = model.discrimT2I_loss
                fetches["genI2T_loss"] = model.genI2T_loss
                fetches["genT2I_loss"] = model.genT2I_loss
                fetches["autoencoderI_loss"] = model.autoencoderI_loss
                fetches["autoencoderT_loss"] = model.autoencoderT_loss
                fetches["code_recon_loss"] = model.code_recon_loss
                fetches["feat_recon_loss"] = model.feat_recon_loss

            # if should(config.summary_freq):
            #     fetches["summary"] = sv.summary_op

            results = sess.run(fetches, options=options, run_metadata=run_metadata)

            # if should(config.summary_freq):
            #     print("recording summary")
            #     sv.summary_writer.add_summary(results["summary"], results["global_step"])

            if should(config.trace_freq):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(config.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * config.batch_size / (time.time() - start)
                remaining = (max_steps - step) * config.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                print("discrimI2T_loss", results["discrimI2T_loss"])
                print("discrimT2I_loss", results["discrimT2I_loss"])
                print("genI2T_loss", results["genI2T_loss"])
                print("genT2I_loss", results["genT2I_loss"])
                print("autoencoderI_loss", results["autoencoderI_loss"])
                print("autoencoderT_loss", results["autoencoderT_loss"])
                print("code_recon_loss", results["code_recon_loss"])
                print("feat_recon_loss", results["feat_recon_loss"])

            if should(config.save_freq):
                print("saving model")
                saver.save(sess, os.path.join(config.output_dir, "model"), global_step=sv.global_step)

            if sv.should_stop():
                break


main()
