import tensorflow as tf
from glob import glob
import config
import os
import json
import pickle

import numpy as np
import scipy
import scipy.ndimage
import nltk.tokenize

class Vocabulary:
    def __init__(self):
        temp = []
        temp2 = {}
        with open(config.vocab, "r") as f:
            for i, sentence in enumerate(f):
                word = sentence.strip().split(" ")[0]
                temp.append(word)
                temp2[word] = i+1

        config.vocab_size = len(temp) + 2 # one extra each for the padding character and the unknown character
        self._id_to_word = tf.constant(temp, dtype=tf.string)
        self._word_to_id = temp2
        temp2["<UNK>"] = config.vocab_size-1

    def get_word(self, id):
        if id==0:
            return ""
        elif id==config.vocab_size-1:
            return "<UNK>"
        return self._id_to_word[id-1]

    def get_index(self, word):
        if word=="":
            return 0
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:
            return self._word_to_id["<UNK>"]

vocabulary = Vocabulary()

class MSCOCODataset:

    def _process_caption(self, caption):
        x = []
        
        tokenized_caption = ["<S>"]
        tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
        tokenized_caption.append("</S>")

        for i in range(min(config.max_length, len(tokenized_caption))):
            x.append(vocabulary.get_index(tokenized_caption[i]))

        if len(x) < config.max_length:
            x.extend([0]*(config.max_length-len(x)))

        return np.array(x)

    def _read_captions(self, captions_file):
        x = open(captions_file, "r")
        captions = [y.strip() for y in x]
        processed_captions = []
        for c in captions:
            processed_captions.append(self._process_caption(c))

        return np.stack((*processed_captions,))

    def __init__(self, mode):
        if not (mode=="train" or mode=="test" or mode=="val"):
            raise Exception("Incorrect mode: {}".format(mode))

        if mode=="train":
            captions_file = os.path.join(config.input_dir, "coco_train_caps.txt")
            imgs_file = os.path.join(config.input_dir, "coco_train_ims.npy")
            # metafile = os.path.join(config.input_dir, "coco_train.txt")
        elif mode=="test":
            captions_file = os.path.join(config.input_dir, "coco_test_caps.txt")
            imgs_file = os.path.join(config.input_dir, "coco_test_ims.npy")
            # metafile = os.path.join(config.input_dir, "coco_test.txt")
        else:
            captions_file = os.path.join(config.input_dir, "coco_dev_caps.txt")
            imgs_file = os.path.join(config.input_dir, "coco_dev_ims.npy")
            # metafile = os.path.join(config.input_dir, "coco_val.txt")

        captions = self._read_captions(captions_file)
        images = self._read_images(imgs_file)

        self.total_size = len(captions)

        with tf.name_scope("load_images"):
            dataset = tf.data.Dataset.from_tensor_slices((images, captions))
            dataset = dataset.repeat()
            dataset = dataset.batch(config.batch_size)

        self.dataset = dataset

    def next_batch(self):
        return self.dataset.make_one_shot_iterator().get_next()

class TestDataset:

    def __init__(self, pkl_file, inputs, batch_size):
        data = pickle.load(open(pkl_file, "rb"))
        self._inputs = inputs
        self._batch_size = batch_size
        self._curr_index = 0

        if inputs=="image":
            self._images = data["images"]
        elif inputs=="text":
            self._captions = data["captions"]
        else:
            raise("Error")

    def _process_caption(self, caption):
        x = []
        
        tokenized_caption = ["<S>"]
        tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
        tokenized_caption.append("</S>")

        for i in range(min(config.max_length, len(tokenized_caption))):
            x.append(vocabulary.get_index(tokenized_caption[i]))

        if len(x) < config.max_length:
            x.extend([0]*(config.max_length-len(x)))

        return np.array(x)

    def next_batch(self):
        if self._inputs == "image":
            temp = self._curr_index + self._batch_size
            temp = min(temp, len(self._images))
            batch = self._images[self._curr_index:temp]
            self._curr_index = temp
            for data_item in batch:
                data_item["processed_input"] = data_item["image"]
                data_item["processed_choice_list"] = []
                for ch in data_item["choice_list"]:
                    data_item["processed_choice_list"].append(self._process_caption(ch))
        else:
            temp = self._curr_index + self._batch_size
            temp = min(temp, len(self._captions))
            batch = self._captions[self._curr_index:temp]
            self._curr_index = temp
            for data_item in batch:
                caption = data_item["caption"]
                data_item["processed_input"] = self._process_caption(caption)
                data_item["processed_choice_list"] = data_item["choice_list"]

        return batch

    def max_steps(self):
        if self._inputs == "image":
            array = self._images
        else:
            array = self._captions

        if len(array)%self._batch_size==0:
            return len(array)//self._batch_size
        else:
            return (len(array)//self._batch_size) + 1
                


if __name__ == "__main__":
    x = MSCOCODataset("val").next_batch()
    print(x)
    sess = tf.Session()
    y = sess.run(x)
    print(y)
