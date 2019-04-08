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
from image_encoder import *
from text_encoder import *
from full_image_decoder import *
from full_text_decoder import *
from exclusive_image_decoder import *
from exclusive_text_decoder import *
from image_discriminator import *
from text_discriminator import *

import config

LAMBDA = 10

Model = collections.namedtuple("Model", "outputsI2T, outputsT2I,\
                               outputsI2Tp, outputsT2Ip,\
                               outputs_exclusiveI2T,outputs_exclusiveT2I,\
                               discrim_exclusiveI2T_loss,discrim_exclusiveT2I_loss,\
                               auto_outputI, auto_outputT\
                               predict_realI2T, predict_realT2I,\
                               predict_fakeI2T, predict_fakeT2I,\
                               sR_I2T,sR_T2I,\
                               eR_I2T,eR_T2I,\
                               discrimI2T_loss, discrimT2I_loss,\
                               genI2T_loss, genT2I_loss,\
                               gen_exclusiveI2T_loss,gen_exclusiveT2I_loss\
                               autoencoderI_loss, autoencoderT_loss,\
                               feat_recon_loss,code_recon_loss,\
                               code_sR_I2T_recon_loss,code_sR_T2I_recon_loss,\
                               code_eR_I2T_recon_loss,code_eR_T2I_recon_loss,\
                               train")

def create_model(inputsI, inputsT):    

    ######### IMAGE_TRANSLATORS
    with tf.variable_scope("generatorI2T_encoder"):
        sR_I2T, eR_I2T = create_image_encoder(inputsI)

    with tf.variable_scope("text_embedding"):
        inputsT_embedded = create_text_embedder(inputsT)

    with tf.variable_scope("generatorT2I_encoder"):
        sR_T2I, eR_T2I = create_text_encoder(inputsT_embedded)

    # Generate random noise to substitute exclusive rep
    z = tf.random_normal(eR_I2T.shape)
    z2 = tf.random_normal(eR_I2T.shape)

    # One copy of the decoder for the noise input, the second copy for the correct the cross-domain autoencoder
    with tf.name_scope("generatorI2T_decoder_noise"):
        with tf.variable_scope("generatorI2T_decoder"):
            outputsI2T_embedded = create_full_text_decoder(sR_I2T, z)
        
        with tf.variable_scope("text_deembedder"):
            outputsI2T = create_text_deembedder(outputsI2T_embedded)

        with tf.variable_scope("generatorI2T_decoder", reuse=True):
            outputsI2Tp_embedded = create_full_text_decoder(sR_I2T, z2)

        with tf.variable_scope("text_deembedder", reuse=True):
            outputsI2Tp = create_text_deembedder(outputsI2Tp_embedded)

    with tf.name_scope("generatorI2T_reconstructor"):
        with tf.variable_scope("generatorT2I_encoder", reuse=True):
            sR_I2T_recon, eR_I2T_recon = create_text_encoder(outputsI2T_embedded)


    with tf.name_scope("generatorT2I_decoder_noise"):
        with tf.variable_scope("generatorT2I_decoder"):
            out_channels = int(inputsI.get_shape()[-1])
            outputsT2I = create_full_image_decoder(sR_T2I, z, out_channels)

        with tf.variable_scope("generatorT2I_decoder",reuse=True):
            outputsT2Ip = create_full_image_decoder(sR_T2I, z2, out_channels)

    with tf.name_scope("generatorT2I_reconstructor"):
        with tf.variable_scope("generatorI2T_encoder", reuse=True):
            sR_T2I_recon, eR_T2I_recon = create_image_encoder(outputsT2I)

    ######### CROSS-DOMAIN AUTOENCODERS
    with tf.name_scope("autoencoderI"):
        # Use here decoder T2I but with shared input from I2T encoder
        with tf.variable_scope("generatorT2I_decoder", reuse=True):
            out_channels = int(inputsI.get_shape()[-1])
            auto_outputI = create_full_image_decoder(sR_T2I, eR_I2T, out_channels)

    with tf.name_scope("autoencoderT"):
        # Use here decoder I2T but with input from T2I encoder
        with tf.variable_scope("generatorI2T_decoder", reuse=True):
            auto_outputT_embedded = create_full_text_decoder(sR_I2T, eR_T2I)

        with tf.variable_scope("text_deembedder", reuse=True):
            auto_outputT = create_text_deembedder(auto_outputT_embedded)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables

    # We will now have 2 different discriminators, one per direction, and two
    # copies of each for real/fake pairs

    with tf.name_scope("real_discriminatorI2T"):
        with tf.variable_scope("discriminatorI2T"):
            predict_realI2T = create_text_discriminator(inputsT_embedded)

    with tf.name_scope("real_discriminatorT2I"):
        with tf.variable_scope("discriminatorT2I"):
            predict_realT2I = create_image_discriminator(inputsI)

    with tf.name_scope("fake_discriminatorI2T"):
        with tf.variable_scope("discriminatorI2T", reuse=True):
            predict_fakeI2T = create_text_discriminator(outputsI2T_embedded)

    with tf.name_scope("fake_discriminatorT2I"):
        with tf.variable_scope("discriminatorT2I", reuse=True):
            predict_fakeT2I = create_image_discriminator(outputsT2I)


    ######### EXCLUSIVE REPRESENTATION
    # Create generators/discriminators for exclusive representation
    with tf.variable_scope("generator_exclusiveI2T_decoder"):
        outputs_exclusiveI2T_embedded = create_exclusive_text_decoder(eR_I2T)
    
    with tf.variable_scope("text_deembedder", reuse=True):
        outputs_exclusiveI2T = create_text_deembedder(outputs_exclusiveI2T_embedded)

    with tf.name_scope("real_discriminator_exclusiveI2T"):
        with tf.variable_scope("discriminator_exclusiveI2T"):
            predict_real_exclusiveI2T = create_text_discriminator(inputsT_embedded)

    with tf.name_scope("fake_discriminator_exclusiveI2T"):
        with tf.variable_scope("discriminator_exclusiveI2T", reuse=True):
            predict_fake_exclusiveI2T = create_text_discriminator(outputs_exclusiveI2T_embedded)


    with tf.variable_scope("generator_exclusiveT2I_decoder"):
        outputs_exclusiveT2I = create_exclusive_image_decoder(eR_T2I, out_channels)

    with tf.name_scope("real_discriminator_exclusiveT2I"):
        with tf.variable_scope("discriminator_exclusiveT2I"):
            predict_real_exclusiveT2I = create_image_discriminator(inputsI)

    with tf.name_scope("fake_discriminator_exclusiveT2I"):
        with tf.variable_scope("discriminator_exclusiveT2I", reuse=True):
            predict_fake_exclusiveT2I = create_image_discriminator(outputs_exclusiveT2I)


    ######### LOSSES

    with tf.name_scope("generatorI2T_loss"):
        genI2T_loss_GAN = -tf.reduce_mean(predict_fakeI2T)
        genI2T_loss = genI2T_loss_GAN * config.gan_weight

    with tf.name_scope("discriminatorI2T_loss"):
        discrimI2T_loss = tf.reduce_mean(predict_fakeI2T) - tf.reduce_mean(predict_realI2T)
        alpha = tf.random_uniform(shape=[tf.shape(outputsI2T_embedded)[0],1], minval=0., maxval=1.)
        differences = tf.reshape(outputsI2T_embedded,[-1, config.txt_output_dim])-tf.reshape(inputsT_embedded,[-1,config.txt_output_dim])
        interpolates = tf.reshape(inputsT_embedded, [-1,config.txt_output_dim]) + (alpha*differences)
        with tf.variable_scope("discriminatorI2T", reuse=True):
            gradients = tf.gradients(create_text_discriminator(tf.reshape(interpolates,[-1,config.text_size[0],config.text_size[1]])),
                         [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)

        tf.summary.histogram("I2T/fake_score", predict_fakeI2T)
        tf.summary.histogram("I2T/real_score", predict_realI2T)
        tf.summary.histogram("I2T/disc_loss", discrimI2T_loss )
        tf.summary.histogram("I2T/gradient_penalty", gradient_penalty)
        discrimI2T_loss += LAMBDA*gradient_penalty

    with tf.name_scope("generatorT2I_loss"):
        genT2I_loss_GAN = -tf.reduce_mean(predict_fakeT2I)
        genT2I_loss = genT2I_loss_GAN * config.gan_weight

    with tf.name_scope("discriminatorT2I_loss"):
        discrimT2I_loss = tf.reduce_mean(predict_fakeT2I) - tf.reduce_mean(predict_realT2I)
        alpha = tf.random_uniform(shape=[tf.shape(outputsT2I)[0],1], minval=0., maxval=1.)
        differences = tf.reshape(outputsT2I,[-1,config.img_output_dim])-tf.reshape(inputsI,[-1,config.img_output_dim])
        interpolates = tf.reshape(inputsI,[-1,config.img_output_dim]) + (alpha*differences)
        with tf.variable_scope("discriminatorT2I", reuse=True):
            gradients = tf.gradients(create_image_discriminator(tf.reshape(interpolates,[-1,config.image_size,config.image_size,3])),
                         [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        discrimT2I_loss += LAMBDA*gradient_penalty

    with tf.name_scope("generator_exclusiveI2T_loss"):
        gen_exclusiveI2T_loss_GAN = -tf.reduce_mean(predict_fake_exclusiveI2T)
        gen_exclusiveI2T_loss = gen_exclusiveI2T_loss_GAN * config.gan_exclusive_weight

    with tf.name_scope("discriminator_exclusiveI2T_loss"):
        discrim_exclusiveI2T_loss = tf.reduce_mean(predict_fake_exclusiveI2T) - tf.reduce_mean(predict_real_exclusiveI2T)
        alpha = tf.random_uniform(shape=[tf.shape(outputs_exclusiveI2T_embedded)[0],1], minval=0., maxval=1.)
        differences = tf.reshape(outputs_exclusiveI2T_embedded,[-1,config.txt_output_dim])-tf.reshape(inputsT_embedded,[-1,config.txt_output_dim])
        interpolates = tf.reshape(inputsT_embedded,[-1,config.txt_output_dim]) + (alpha*differences)
        with tf.variable_scope("discriminator_exclusiveI2T", reuse=True):
            gradients = tf.gradients(create_image_discriminator(tf.reshape(interpolates,[-1,config.text_size[0],config.text_size[1]])),
                             [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        discrim_exclusiveI2T_loss += LAMBDA*gradient_penalty

    with tf.name_scope("generator_exclusiveT2I_loss"):
        gen_exclusiveT2I_loss_GAN = -tf.reduce_mean(predict_fake_exclusiveT2I)
        gen_exclusiveT2I_loss = gen_exclusiveT2I_loss_GAN * config.gan_exclusive_weight


    with tf.name_scope("discriminator_exclusiveT2I_loss"):
        discrim_exclusiveT2I_loss = tf.reduce_mean(predict_fake_exclusiveT2I) - tf.reduce_mean(predict_real_exclusiveT2I)
        alpha = tf.random_uniform(shape=[tf.shape(outputs_exclusiveT2I)[0],1], minval=0., maxval=1.)
        differences = tf.reshape(outputs_exclusiveT2I,[-1,config.img_output_dim])-tf.reshape(inputsI,[-1,config.img_output_dim])
        interpolates = tf.reshape(inputsI,[-1,config.img_output_dim]) + (alpha*differences)
        with tf.variable_scope("discriminator_exclusiveT2I", reuse=True):
            gradients = tf.gradients(create_text_discriminator(tf.reshape(interpolates,[-1,config.image_size,config.image_size,3])),
                             [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                       reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        discrim_exclusiveT2I_loss += LAMBDA*gradient_penalty

    with tf.name_scope("autoencoderI_loss"):
        autoencoderI_loss = config.l1_weight * tf.reduce_mean(tf.abs(auto_outputI-inputsI))

    # with tf.name_scope("autoencoderT_loss"):
    #     autoencoderT_loss = config.l1_weight*tf.reduce_mean(tf.abs(auto_outputT-inputsT))
    with tf.name_scope("autoencoderT_loss"):
        autoencoderT_loss = config.l1_weight * tf.contrib.seq2seq.sequence_loss(
            auto_outputT, inputsT, tf.ones(shape=(config.vocab_size,))
        )

    with tf.name_scope("feat_recon_loss"):
        feat_recon_loss = config.l1_weight*tf.reduce_mean(tf.abs(sR_I2T-sR_T2I))

    with tf.name_scope("code_recon_loss"):
        code_sR_I2T_recon_loss = tf.reduce_mean(tf.abs(sR_I2T_recon-sR_I2T))
        code_sR_T2I_recon_loss = tf.reduce_mean(tf.abs(sR_T2I_recon-sR_T2I))
        code_eR_I2T_recon_loss = tf.reduce_mean(tf.abs(eR_I2T_recon-z))
        code_eR_T2I_recon_loss = tf.reduce_mean(tf.abs(eR_T2I_recon-z))
        code_recon_loss = config.l1_weight*(code_sR_I2T_recon_loss + code_sR_T2I_recon_loss
                                    +code_eR_I2T_recon_loss + code_eR_T2I_recon_loss)

    ######### OPTIMIZERS

    with tf.name_scope("discriminatorI2T_train"):
        discrimI2T_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminatorI2T")]
        discrimI2T_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        discrimI2T_grads_and_vars = discrimI2T_optim.compute_gradients(discrimI2T_loss, var_list=discrimI2T_tvars)
        discrimI2T_train = discrimI2T_optim.apply_gradients(discrimI2T_grads_and_vars)

    with tf.name_scope("discriminatorT2I_train"):
        discrimT2I_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminatorT2I")]
        discrimT2I_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        discrimT2I_grads_and_vars = discrimT2I_optim.compute_gradients(discrimT2I_loss, var_list=discrimT2I_tvars)
        discrimT2I_train = discrimT2I_optim.apply_gradients(discrimT2I_grads_and_vars)

    with tf.name_scope("generatorI2T_train"):
        with tf.control_dependencies([discrimI2T_train]):
            genI2T_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generatorI2T")]
            genI2T_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
            genI2T_grads_and_vars = genI2T_optim.compute_gradients(genI2T_loss, var_list=genI2T_tvars)
            genI2T_train = genI2T_optim.apply_gradients(genI2T_grads_and_vars)

    with tf.name_scope("generatorT2I_train"):
        with tf.control_dependencies([discrimT2I_train]):
            genT2I_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generatorT2I")]
            genT2I_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
            genT2I_grads_and_vars = genT2I_optim.compute_gradients(genT2I_loss, var_list=genT2I_tvars)
            genT2I_train = genT2I_optim.apply_gradients(genT2I_grads_and_vars)

    with tf.name_scope("discriminator_exclusiveI2T_train"):
        discrim_exclusiveI2T_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator_exclusiveI2T")]
        discrim_exclusiveI2T_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        discrim_exclusiveI2T_grads_and_vars = discrim_exclusiveI2T_optim.compute_gradients(discrim_exclusiveI2T_loss, var_list=discrim_exclusiveI2T_tvars)
        discrim_exclusiveI2T_train = discrim_exclusiveI2T_optim.apply_gradients(discrim_exclusiveI2T_grads_and_vars)

    with tf.name_scope("generator_exclusiveI2T_train"):
        with tf.control_dependencies([discrim_exclusiveI2T_train]):
            gen_exclusiveI2T_tvars = [var for var in tf.trainable_variables()
                                      if var.name.startswith("generator_exclusiveI2T")
                                        or var.name.startswith("generatorI2T_encoder")]
            gen_exclusiveI2T_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
            gen_exclusiveI2T_grads_and_vars = gen_exclusiveI2T_optim.compute_gradients(gen_exclusiveI2T_loss, var_list=gen_exclusiveI2T_tvars)
            gen_exclusiveI2T_train = gen_exclusiveI2T_optim.apply_gradients(gen_exclusiveI2T_grads_and_vars)

    with tf.name_scope("discriminator_exclusiveT2I_train"):
        discrim_exclusiveT2I_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator_exclusiveT2I")]
        discrim_exclusiveT2I_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        discrim_exclusiveT2I_grads_and_vars = discrim_exclusiveT2I_optim.compute_gradients(discrim_exclusiveT2I_loss, var_list=discrim_exclusiveT2I_tvars)
        discrim_exclusiveT2I_train = discrim_exclusiveT2I_optim.apply_gradients(discrim_exclusiveT2I_grads_and_vars)

    with tf.name_scope("generator_exclusiveT2I_train"):
        with tf.control_dependencies([discrim_exclusiveT2I_train]):
            gen_exclusiveT2I_tvars = [var for var in tf.trainable_variables()
                                      if var.name.startswith("generator_exclusiveT2I")
                                        or var.name.startswith("generatorT2I_encoder")]
            gen_exclusiveT2I_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
            gen_exclusiveT2I_grads_and_vars = gen_exclusiveT2I_optim.compute_gradients(gen_exclusiveT2I_loss, var_list=gen_exclusiveT2I_tvars)
            gen_exclusiveT2I_train = gen_exclusiveT2I_optim.apply_gradients(gen_exclusiveT2I_grads_and_vars)

    with tf.name_scope("autoencoderI_train"):
        autoencoderI_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorI2T_encoder")
                              or var.name.startswith("generatorT2I_encoder")
                              or var.name.startswith("generatorT2I_decoder")]
        autoencoderI_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        autoencoderI_grads_and_vars = autoencoderI_optim.compute_gradients(autoencoderI_loss, var_list=autoencoderI_tvars)
        autoencoderI_train = autoencoderI_optim.apply_gradients(autoencoderI_grads_and_vars)

    with tf.name_scope("autoencoderT_train"):
        autoencoderT_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorT2I_encoder") or
                              var.name.startswith("generatorI2T_encoder") or
                              var.name.startswith("generatorI2T_decoder")]
        autoencoderT_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        autoencoderT_grads_and_vars = autoencoderT_optim.compute_gradients(autoencoderT_loss, var_list=autoencoderT_tvars)
        autoencoderT_train = autoencoderT_optim.apply_gradients(autoencoderT_grads_and_vars)


    with tf.name_scope("feat_recon_train"):
        feat_recon_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorI2T_encoder") or
                              var.name.startswith("generatorT2I_encoder")]
        feat_recon_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        feat_recon_grads_and_vars = feat_recon_optim.compute_gradients(feat_recon_loss, var_list=feat_recon_tvars)
        feat_recon_train = feat_recon_optim.apply_gradients(feat_recon_grads_and_vars)

    with tf.name_scope("code_recon_train"):
        code_recon_tvars = [var for var in tf.trainable_variables() if
                              var.name.startswith("generatorI2T") or
                              var.name.startswith("generatorT2I")]
        code_recon_optim = tf.train.AdamOptimizer(config.lr, config.beta1)
        code_recon_grads_and_vars = code_recon_optim.compute_gradients(code_recon_loss, var_list=code_recon_tvars)
        code_recon_train = code_recon_optim.apply_gradients(code_recon_grads_and_vars)



    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrimI2T_loss, discrimT2I_loss,
                               genI2T_loss, genT2I_loss,
                               autoencoderI_loss, autoencoderT_loss,
                               feat_recon_loss,code_recon_loss,
                               code_sR_I2T_recon_loss, code_sR_T2I_recon_loss,
                               code_eR_I2T_recon_loss, code_eR_T2I_recon_loss,
                               discrim_exclusiveI2T_loss, discrim_exclusiveT2I_loss,
                               gen_exclusiveI2T_loss, gen_exclusiveT2I_loss])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    return Model(
        predict_realI2T=predict_realI2T,
        predict_realT2I=predict_realT2I,
        predict_fakeI2T=predict_fakeI2T,
        predict_fakeT2I=predict_fakeT2I,
        sR_I2T=sR_I2T,
        sR_T2I=sR_T2I,
        eR_I2T=eR_I2T,
        eR_T2I=eR_T2I,
        discrimI2T_loss=ema.average(discrimI2T_loss),
        discrimT2I_loss=ema.average(discrimT2I_loss),
        genI2T_loss=ema.average(genI2T_loss),
        genT2I_loss=ema.average(genT2I_loss),
        discrim_exclusiveI2T_loss=ema.average(discrim_exclusiveI2T_loss),
        discrim_exclusiveT2I_loss=ema.average(discrim_exclusiveT2I_loss),
        gen_exclusiveI2T_loss=ema.average(gen_exclusiveI2T_loss),
        gen_exclusiveT2I_loss=ema.average(gen_exclusiveT2I_loss),
        outputsI2T=outputsI2T,
        outputsT2I=outputsT2I,
        outputsI2Tp=outputsI2Tp,
        outputsT2Ip=outputsT2Ip,
        outputs_exclusiveI2T=outputs_exclusiveI2T,
        outputs_exclusiveT2I=outputs_exclusiveT2I,
        auto_outputI = auto_outputI,
        autoencoderI_loss=ema.average(autoencoderI_loss),
        auto_outputT = auto_outputT,
        autoencoderT_loss=ema.average(autoencoderT_loss),
        feat_recon_loss=ema.average(feat_recon_loss),
        code_recon_loss=ema.average(code_recon_loss),
        code_sR_I2T_recon_loss=ema.average(code_sR_I2T_recon_loss),
        code_sR_T2I_recon_loss=ema.average(code_sR_T2I_recon_loss),
        code_eR_I2T_recon_loss=ema.average(code_eR_I2T_recon_loss),
        code_eR_T2I_recon_loss=ema.average(code_eR_T2I_recon_loss),
        train=tf.group(update_losses, incr_global_step, genI2T_train,
                       genT2I_train, autoencoderI_train, autoencoderT_train,code_recon_train,
                       gen_exclusiveI2T_train,gen_exclusiveT2I_train,feat_recon_train),
    )

if __name__ == "__main__":
    val_dataset = MSCOCODataset("val")
    ids, inputsI, inputsT = val_dataset.next_batch()
    model = create_model(inputsI, inputsT)