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

from data_ops import MSCOCODataset, Vocabulary

CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "ids, inputsI, inputsT, ids_val, inputsI_val, inputsT_val, count, steps_per_epoch")

vocabulary = Vocabulary()

def load_examples():
    if config.input_dir is None or not os.path.exists(config.input_dir):
        raise Exception("input_dir does not exist")

    if config.mode == "train":
        train_dataset = MSCOCODataset("val")
        ids, inputsI, inputsT = train_dataset.next_batch()
        count = train_dataset.total_size
        steps_per_epoch = int(math.ceil(train_dataset.total_size/config.batch_size))
        ids_val, inputsI_val, inputsT_val = MSCOCODataset("val").next_batch()
    elif config.mode == "test":
        test_dataset = MSCOCODataset("test")
        ids, inputsI, inputsT = test_dataset.next_batch()
        count = test_dataset.total_size
        steps_per_epoch = int(math.ceil(test_dataset.total_size/config.batch_size))
        ids_val, inputsI_val, inputsT_val = None, None, None

    return Examples(
        ids=ids,
        inputsI=inputsI,
        inputsT=inputsT,
        ids_val=ids_val,
        inputsI_val=inputsI_val,
        inputsT_val=inputsT_val,
        count=count,
        steps_per_epoch=steps_per_epoch,
    )

def save_results(fetches, step=None):
    results_dir = os.path.join(config.output_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    filesets = []
    for i, sample_id in enumerate(fetches["ids"]):
        # name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        name = str(sample_id.decode('utf-8'))
        fileset = {"name": name, "step": step}
        img_kinds = ["inputsI", "auto_outputsI", "outputsT2I", "outputsT2Ip"]
        txt_kinds = ["outputsI2T", "outputsI2Tp", "inputsT", "auto_outputsT"]

        for kind in img_kinds:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(results_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)

        for kind in txt_kinds:
            filename = name + "-" + kind + ".txt"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(results_dir, filename)
            contents = fetches[kind][i]

            string = " ".join(contents)

            with open(out_path, "wb") as f:
                f.write(string)

        filesets.append(fileset)
    return filesets


# def save_features(fetches, step=None):
#     image_dir = os.path.join(config.output_dir, "features")
#     if not os.path.exists(image_dir):
#         os.makedirs(image_dir)

#     filesets = []
#     for i, in_path in enumerate(fetches["paths"]):
#         name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
#         fileset = {"name": name, "step": step}
#         filename = name + ".mat"
#         out_path = os.path.join(image_dir, filename)
#         sio.savemat(out_path,{'inX':fetches["inputsI"][i],
#                              'inY':fetches["inputsT"][i],
#                              'sR_I2T':fetches["sR_I2T"][i],
#                              'sR_T2I':fetches["sR_T2I"][i],
#                              'eR_I2T':fetches["eR_I2T"][i],
#                              'eR_T2I':fetches["eR_T2I"][i]})

#     return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(config.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>inputsI</th><th>out(1)</th><th>out(2)</th><th>auto</th><th>inputsT</th><th>out(1)</th><th>out(2)</th><th>auto</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])
        all_kinds = {
            "inputsI":"img", 
            "outputsI2T":"txt", 
            "outputsI2Tp":"txt",
            "auto_outputsI":"img",
            "inputsT":"txt",
            "outputsT2I":"img", 
            "outputsT2Ip":"img",
            "auto_outputsT":"txt"
        }

        for kind in all_kinds:
            if all_kinds[kind]=="img":
                index.write("<td><img src='results/%s'></td>" % fileset[kind])
            else:
                with open("results/"+fileset[kind], "r") as f:
                    caption = f.read()
                    index.write("<td>{}</td>".format(caption))

        index.write("</tr>")
    return index_path


def main():
    if config.seed is None:
        config.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if config.mode == "test" or config.mode == "features":
        if config.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngfI", "ndfI"}
        with open(os.path.join(config.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(config.a, key, val)

    for k, v in config._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(config.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(config.a), sort_keys=True, indent=4))

    examples = load_examples()

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputsI, examples.inputsT)

    # undo colorization splitting on images that we use for display/output
    inputsI = deprocess(examples.inputsI)
    inputsT = examples.inputsT
    outputsI2T = model.outputsI2T
    outputsT2I = deprocess(model.outputsT2I)
    outputsI2Tp = model.outputsI2Tp
    outputsT2Ip = deprocess(model.outputsT2Ip)
    outputs_exclusiveI2T = model.outputs_exclusiveI2T
    outputs_exclusiveT2I = deprocess(model.outputs_exclusiveT2I)
    auto_outputI = deprocess(model.auto_outputI)
    auto_outputT = model.auto_outputT
    sR_I2T = model.sR_I2T
    sR_T2I = model.sR_T2I
    eR_I2T = model.eR_I2T
    eR_T2I = model.eR_T2I

    def convert_img(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    def convert_index_txt(index_vectors):

        def sample_stringifier(index_vector):
            def word_stringifier(index):
                return vocabulary.get_word(index)
            return tf.map_fn(word_stringifier, index_vector, dtype=tf.string)

        return tf.map_fn(sample_stringifier, index_vectors, dtype=tf.string)

    def convert_txt(one_hot_vectors):
        index_vectors = tf.argmax(one_hot_vectors, axis=2)
        return convert_index_txt(index_vectors)

    # reverse any processing on images and text so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputsI"):
        converted_inputsI = convert_img(inputsI)

    with tf.name_scope("convert_inputsT"):
        converted_inputsT = convert_index_txt(inputsT)

    with tf.name_scope("convert_outputsI2T"):
        converted_outputsI2T = convert_txt(outputsI2T)

    with tf.name_scope("convert_outputsT2I"):
        converted_outputsT2I = convert_img(outputsT2I)

    with tf.name_scope("convert_outputsI2Tp"):
        converted_outputsI2Tp = convert_txt(outputsI2Tp)

    with tf.name_scope("convert_outputsT2Ip"):
        converted_outputsT2Ip = convert_img(outputsT2Ip)

    with tf.name_scope("convert_outputs_exclusiveI2T"):
        converted_outputs_exclusiveI2T = convert_txt(outputs_exclusiveI2T)

    with tf.name_scope("convert_outputs_exclusiveT2I"):
        converted_outputs_exclusiveT2I = convert_img(outputs_exclusiveT2I)

    with tf.name_scope("convert_auto_outputsI"):
        converted_auto_outputI = convert_img(auto_outputI)

    with tf.name_scope("convert_auto_outputsT"):
        converted_auto_outputT = convert_txt(auto_outputT)

    with tf.name_scope("encode_data"):
        display_fetches = {
            "ids": examples.ids,
            "inputsI": tf.map_fn(tf.image.encode_png, converted_inputsI, dtype=tf.string, name="inputI_pngs"),
            "inputsT": converted_inputsT,
            "outputsI2T": converted_outputsI2T,
            "outputsT2I": tf.map_fn(tf.image.encode_png, converted_outputsT2I, dtype=tf.string, name="outputT2I_pngs"),
            "outputsI2Tp": converted_outputsI2Tp,
            "outputsT2Ip": tf.map_fn(tf.image.encode_png, converted_outputsT2Ip, dtype=tf.string, name="outputT2Ip_pngs"),
            "outputs_exclusiveI2T": converted_outputs_exclusiveI2T,
            "outputs_exclusiveT2I": tf.map_fn(tf.image.encode_png, converted_outputs_exclusiveT2I, dtype=tf.string, name="output_exclusiveT2I_pngs"),
            "auto_outputsI": tf.map_fn(tf.image.encode_png, converted_auto_outputI, dtype=tf.string, name="auto_outputI_pngs"),
            "auto_outputsT": converted_auto_outputT
        }
    # with tf.name_scope("extract_features"):
    #     features_fetches = {
    #         "ids": examples.ids,
    #         "inputsI": converted_inputsI,
    #         "sR_I2T": sR_I2T,
    #         "eR_I2T": eR_I2T,
    #         "inputsT": converted_inputsT,
    #         "sR_T2I": sR_T2I,
    #         "eR_T2I": eR_T2I,
    #     }

    # summaries
    with tf.name_scope("I1_input_summary"):
        tf.summary.image("inputsI", converted_inputsI,max_outputs=3)

    with tf.name_scope("T1_input_summary"):
        tf.summary.text("inputsT", converted_inputsT,max_outputs=3)

    with tf.name_scope("I2T_output_summary"):
        tf.summary.text("outputsI2T", converted_outputsI2T,max_outputs=3)

    with tf.name_scope("T2I_output_summary"):
        tf.summary.image("outputsT2I", converted_outputsT2I,max_outputs=3)

    with tf.name_scope("I_autoencoder_summary"):
        tf.summary.image("auto_outputI", converted_auto_outputI,max_outputs=3)

    with tf.name_scope("T_autoencoder_summary"):
        tf.summary.text("auto_outputT", converted_auto_outputT,max_outputs=3)

    with tf.name_scope("otherNoise_output_summary"):
        tf.summary.text("outputsI2Tp", converted_outputsI2Tp,max_outputs=3)
        tf.summary.image("outputsT2Ip", converted_outputsT2Ip,max_outputs=3)

    with tf.name_scope("zzexclusive_I2T_summary"):
        tf.summary.text("outputsI2T", converted_outputs_exclusiveI2T,max_outputs=3)

    with tf.name_scope("zzexclusive_T2I_summary"):
        tf.summary.image("outputsT2I", converted_outputs_exclusiveT2I,max_outputs=3)

    tf.summary.scalar("discriminatorI2T_loss", model.discrimI2T_loss)
    tf.summary.scalar("discriminatorT2I_loss", model.discrimT2I_loss)
    tf.summary.scalar("generatorI2T_loss", model.genI2T_loss)
    tf.summary.scalar("generatorT2I_loss", model.genT2I_loss)
    tf.summary.scalar("generator_exclusiveI2T_loss", model.gen_exclusiveI2T_loss)
    tf.summary.scalar("discriminator_exclusiveI2T_loss", model.discrim_exclusiveI2T_loss)
    tf.summary.scalar("generator_exclusiveT2I_loss", model.gen_exclusiveT2I_loss)
    tf.summary.scalar("discriminator_exclusiveT2I_loss", model.discrim_exclusiveT2I_loss)
    tf.summary.scalar("autoencoderI_loss", model.autoencoderI_loss)
    tf.summary.scalar("autoencoderT_loss", model.autoencoderT_loss)
    tf.summary.scalar("feat_recon_loss", model.feat_recon_loss)
    tf.summary.scalar("code_sR_I2T_recon_loss", model.code_sR_I2T_recon_loss)
    tf.summary.scalar("code_sR_T2I_recon_loss", model.code_sR_T2I_recon_loss)
    tf.summary.scalar("code_eR_I2T_recon_loss", model.code_eR_I2T_recon_loss)
    tf.summary.scalar("code_eR_T2I_recon_loss", model.code_eR_T2I_recon_loss)
    tf.summary.scalar("code_recon_loss", model.code_recon_loss)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = config.output_dir if (config.trace_freq > 0 or config.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if config.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(config.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if config.max_epochs is not None:
            max_steps = examples.steps_per_epoch * config.max_epochs
        if config.max_steps is not None:
            max_steps = config.max_steps

        if config.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_results(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)

        # elif config.mode == "features":
        #     max_steps = min(examples.steps_per_epoch, max_steps)
        #     for step in range(max_steps):
        #         results = sess.run(features_fetches)
        #         save_features(results)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
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

                if should(config.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(config.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(config.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(config.display_freq):
                    print("saving display images")
                    filesets = save_results(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

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
