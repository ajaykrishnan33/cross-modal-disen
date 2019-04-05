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

from data_ops import MSCOCODataset

CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputsX, inputsY, count, steps_per_epoch")


def load_examples():
    if config.input_dir is None or not os.path.exists(config.input_dir):
        raise Exception("input_dir does not exist")

    dataset = MSCOCODataset(config.mode)

    
    # No longer in terms of input/target, but bidirectionally on domains X and Y
    inputsX, inputsY = [a_images, b_images]

    return Examples(
        paths=paths_batch,
        inputsX=inputsX_batch,
        inputsY=inputsY_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(config.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        all_kinds = ["inputsX", "outputsX2Y", "outputsX2Yp",
                         "auto_outputsX","im_swapped_X", "sel_auto_X","inputsY",
                         "outputsY2X", "outputsY2Xp","auto_outputsY" ,"im_swapped_Y", "sel_auto_Y"]

        for kind in all_kinds:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def save_features(fetches, step=None):
    image_dir = os.path.join(config.output_dir, "features")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        filename = name + ".mat"
        out_path = os.path.join(image_dir, filename)
        sio.savemat(out_path,{'inX':fetches["inputsX"][i],
                             'inY':fetches["inputsY"][i],
                             'sR_X2Y':fetches["sR_X2Y"][i],
                             'sR_Y2X':fetches["sR_Y2X"][i],
                             'eR_X2Y':fetches["eR_X2Y"][i],
                             'eR_Y2X':fetches["eR_Y2X"][i]})


    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(config.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>inX</th><th>out(1)</th><th>out(2)</th><th>auto</th><th>swap</th><th>randomimage</th><th>inY</th><th>out(1)</th><th>out(2)</th><th>auto</th><th>swap</th><th>rnd</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])
        all_kinds = ["inputsX", "outputsX2Y", "outputsX2Yp",
                     "auto_outputsX","im_swapped_X", "sel_auto_X","inputsY",
                     "outputsY2X", "outputsY2Xp","auto_outputsY" ,"im_swapped_Y", "sel_auto_Y"]

        for kind in all_kinds:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

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
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(config.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        config.scale_size = CROP_SIZE
        config.flip = False

    for k, v in config.config._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(config.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputsX, examples.inputsY)

    # undo colorization splitting on images that we use for display/output
    inputsX = deprocess(examples.inputsX)
    inputsY = deprocess(examples.inputsY)
    outputsX2Y = deprocess(model.outputsX2Y)
    outputsY2X = deprocess(model.outputsY2X)
    outputsX2Yp = deprocess(model.outputsX2Yp)
    outputsY2Xp = deprocess(model.outputsY2Xp)
    outputs_exclusiveX2Y = deprocess(model.outputs_exclusiveX2Y)
    outputs_exclusiveY2X = deprocess(model.outputs_exclusiveY2X)
    auto_outputX = deprocess(model.auto_outputX)
    auto_outputY = deprocess(model.auto_outputY)
    im_swapped_X = deprocess(model.im_swapped_X)
    im_swapped_Y = deprocess(model.im_swapped_Y)
    sel_auto_X = deprocess(model.sel_auto_X)
    sel_auto_Y = deprocess(model.sel_auto_Y)
    sR_X2Y = model.sR_X2Y
    sR_Y2X = model.sR_Y2X
    eR_X2Y = model.eR_X2Y
    eR_Y2X = model.eR_Y2X

    def convert(image):
        if config.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * config.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputsX"):
        converted_inputsX = convert(inputsX)

    with tf.name_scope("convert_inputsY"):
        converted_inputsY = convert(inputsY)

    with tf.name_scope("convert_outputsX2Y"):
        converted_outputsX2Y = convert(outputsX2Y)

    with tf.name_scope("convert_outputsY2X"):
        converted_outputsY2X = convert(outputsY2X)

    with tf.name_scope("convert_outputsX2Yp"):
        converted_outputsX2Yp = convert(outputsX2Yp)

    with tf.name_scope("convert_outputsY2Xp"):
        converted_outputsY2Xp = convert(outputsY2Xp)

    with tf.name_scope("convert_outputs_exclusiveX2Y"):
        converted_outputs_exclusiveX2Y = convert(outputs_exclusiveX2Y)

    with tf.name_scope("convert_outputs_exclusiveY2X"):
        converted_outputs_exclusiveY2X = convert(outputs_exclusiveY2X)

    with tf.name_scope("convert_auto_outputsX"):
        converted_auto_outputX = convert(auto_outputX)

    with tf.name_scope("convert_auto_outputsY"):
        converted_auto_outputY = convert(auto_outputY)

    with tf.name_scope("convert_im_swapped_Y"):
        converted_im_swapped_Y = convert(im_swapped_Y)

    with tf.name_scope("convert_sel_auto_Y"):
        converted_sel_auto_Y= convert(sel_auto_Y)

    with tf.name_scope("convert_im_swapped_X"):
        converted_im_swapped_X = convert(im_swapped_X)

    with tf.name_scope("convert_sel_auto_X"):
        converted_sel_auto_X= convert(sel_auto_X)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputsX": tf.map_fn(tf.image.encode_png, converted_inputsX, dtype=tf.string, name="inputX_pngs"),
            "inputsY": tf.map_fn(tf.image.encode_png, converted_inputsY, dtype=tf.string, name="inputY_pngs"),
            "outputsX2Y": tf.map_fn(tf.image.encode_png, converted_outputsX2Y, dtype=tf.string, name="outputX2Y_pngs"),
            "outputsY2X": tf.map_fn(tf.image.encode_png, converted_outputsY2X, dtype=tf.string, name="outputY2X_pngs"),
            "outputsX2Yp": tf.map_fn(tf.image.encode_png, converted_outputsX2Yp, dtype=tf.string, name="outputX2Yp_pngs"),
            "outputsY2Xp": tf.map_fn(tf.image.encode_png, converted_outputsY2Xp, dtype=tf.string, name="outputY2Xp_pngs"),
            "outputs_exclusiveX2Y": tf.map_fn(tf.image.encode_png, converted_outputs_exclusiveX2Y, dtype=tf.string, name="output_exclusiveX2Y_pngs"),
            "outputs_exclusiveY2X": tf.map_fn(tf.image.encode_png, converted_outputs_exclusiveY2X, dtype=tf.string, name="output_exclusiveY2X_pngs"),
            "auto_outputsX": tf.map_fn(tf.image.encode_png, converted_auto_outputX, dtype=tf.string, name="auto_outputX_pngs"),
            "auto_outputsY": tf.map_fn(tf.image.encode_png, converted_auto_outputY, dtype=tf.string, name="auto_outputY_pngs"),
            "im_swapped_Y": tf.map_fn(tf.image.encode_png, converted_im_swapped_Y, dtype=tf.string, name="im_swapped_Y_pngs"),
            "sel_auto_Y": tf.map_fn(tf.image.encode_png, converted_sel_auto_Y, dtype=tf.string, name="sel_auto_Y_pngs"),
            "im_swapped_X": tf.map_fn(tf.image.encode_png, converted_im_swapped_X, dtype=tf.string, name="im_swapped_X_pngs"),
            "sel_auto_X": tf.map_fn(tf.image.encode_png, converted_sel_auto_X, dtype=tf.string, name="sel_auto_X_pngs"),
        }
    with tf.name_scope("extract_features"):
        features_fetches = {
            "paths": examples.paths,
            "inputsX": converted_inputsX,
            "sR_X2Y": sR_X2Y,
            "eR_X2Y": eR_X2Y,
            "inputsY": converted_inputsY,
            "sR_Y2X": sR_Y2X,
            "eR_Y2X": eR_Y2X,
        }

    # summaries
    with tf.name_scope("X1_input_summary"):
        tf.summary.image("inputsX", converted_inputsX,max_outputs=3)

    with tf.name_scope("Y1_input_summary"):
        tf.summary.image("inputsY", converted_inputsY,max_outputs=3)

    with tf.name_scope("X2Y_output_summary"):
        tf.summary.image("outputsX2Y", converted_outputsX2Y,max_outputs=3)

    with tf.name_scope("Y2X_outpu2_summary"):
        tf.summary.image("outputsY2X", converted_outputsY2X,max_outputs=3)

    with tf.name_scope("X_autoencoder_summary"):
        tf.summary.image("auto_outputX", converted_auto_outputX,max_outputs=3)

    with tf.name_scope("Y_autoencoder_summary"):
        tf.summary.image("auto_outputY", converted_auto_outputY,max_outputs=3)

    with tf.name_scope("swapped_1Y_summary"):
        tf.summary.image("im_swapped_Y", converted_im_swapped_Y,max_outputs=3)
        tf.summary.image("sel_auto_Y", converted_sel_auto_Y,max_outputs=3)

    with tf.name_scope("swapped_2X_summary"):
        tf.summary.image("im_swapped_X", converted_im_swapped_X,max_outputs=3)
        tf.summary.image("sel_auto_X", converted_sel_auto_X,max_outputs=3)

    with tf.name_scope("otherNoise_output_summary"):
        tf.summary.image("outputsX2Yp", converted_outputsX2Yp,max_outputs=3)
        tf.summary.image("outputsY2Xp", converted_outputsY2Xp,max_outputs=3)

    with tf.name_scope("zzexclusive_X2Y_summary"):
        tf.summary.image("outputsX2Y", converted_outputs_exclusiveX2Y,max_outputs=3)

    with tf.name_scope("zzexclusive_Y2X_summary"):
        tf.summary.image("outputsY2X", converted_outputs_exclusiveY2X,max_outputs=3)

    tf.summary.scalar("discriminatorX2Y_loss", model.discrimX2Y_loss)
    tf.summary.scalar("discriminatorY2X_loss", model.discrimY2X_loss)
    tf.summary.scalar("generatorX2Y_loss", model.genX2Y_loss)
    tf.summary.scalar("generatorY2X_loss", model.genY2X_loss)
    tf.summary.scalar("generator_exclusiveX2Y_loss", model.gen_exclusiveX2Y_loss)
    tf.summary.scalar("discriminator_exclusiveX2Y_loss", model.discrim_exclusiveX2Y_loss)
    tf.summary.scalar("generator_exclusiveY2X_loss", model.gen_exclusiveY2X_loss)
    tf.summary.scalar("discriminator_exclusiveY2X_loss", model.discrim_exclusiveY2X_loss)
    tf.summary.scalar("autoencoderX_loss", model.autoencoderX_loss)
    tf.summary.scalar("autoencoderY_loss", model.autoencoderY_loss)
    tf.summary.scalar("feat_recon_loss", model.feat_recon_loss)
    tf.summary.scalar("code_sR_X2Y_recon_loss", model.code_sR_X2Y_recon_loss)
    tf.summary.scalar("code_sR_Y2X_recon_loss", model.code_sR_Y2X_recon_loss)
    tf.summary.scalar("code_eR_X2Y_recon_loss", model.code_eR_X2Y_recon_loss)
    tf.summary.scalar("code_eR_Y2X_recon_loss", model.code_eR_Y2X_recon_loss)
    tf.summary.scalar("code_recon_loss", model.code_recon_loss)

    #for var in tf.trainable_variables():
        #tf.summary.histogram(var.op.name + "/values", var)

    #for grad, var in model.discrimX2Y_grads_and_vars + model.genX2Y_grads_and_vars:
        #tf.summary.histogram(var.op.name + "/gradientsX2Y", grad)

    #for grad, var in model.discrimY2X_grads_and_vars + model.genY2X_grads_and_vars:
        #tf.summary.histogram(var.op.name + "/gradientsY2X", grad)

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
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)

        elif config.mode == "features":
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(features_fetches)
                save_features(results)
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
                    fetches["discrimX2Y_loss"] = model.discrimX2Y_loss
                    fetches["discrimY2X_loss"] = model.discrimY2X_loss
                    fetches["genX2Y_loss"] = model.genX2Y_loss
                    fetches["genY2X_loss"] = model.genY2X_loss
                    fetches["autoencoderX_loss"] = model.autoencoderX_loss
                    fetches["autoencoderY_loss"] = model.autoencoderY_loss
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
                    filesets = save_images(results["display"], step=results["global_step"])
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
                    print("discrimX2Y_loss", results["discrimX2Y_loss"])
                    print("discrimY2X_loss", results["discrimY2X_loss"])
                    print("genX2Y_loss", results["genX2Y_loss"])
                    print("genY2X_loss", results["genY2X_loss"])
                    print("autoencoderX_loss", results["autoencoderX_loss"])
                    print("autoencoderY_loss", results["autoencoderY_loss"])
                    print("code_recon_loss", results["code_recon_loss"])
                    print("feat_recon_loss", results["feat_recon_loss"])

                if should(config.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(config.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
