import tensorflow as tf
from glob import glob
import config
import os

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

    def _process_image(self, encoded_image):
        img = tf.image.decode_jpeg(encoded_image, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize_images(
            img, size=[config.image_size, config.image_size], method=tf.image.ResizeMethod.BILINEAR
        )

        # range: [0,1] ==> [-1,+1]
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)

        return img

    def _extract_fn(self, tfrecord):
        context, sequence = tf.io.parse_single_sequence_example(
            tfrecord,
            context_features={
                "image/image_id": tf.FixedLenFeature([], dtype=tf.string),
                "image/data": tf.FixedLenFeature([], dtype=tf.string)
            },
            sequence_features={
                "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64)
            }
        )

        sample_id = context["image/image_id"]

        encoded_image = context["image/data"]
        processed_image = self._process_image(encoded_image)

        caption = sequence["image/caption_ids"]

        def true_fn():
            return tf.pad(
                caption, tf.convert_to_tensor([[0, config.max_length - tf.shape(caption)[0]]])
            )

        def false_fn():
            return caption[:config.max_length]

        padded_caption = tf.cond(
            tf.shape(caption)[0] < config.max_length, true_fn=true_fn, false_fn=false_fn
        )
        # ### NEEDS FIXING
        # if tf.shape(caption)[0] < config.max_length:
        #     padded_caption = tf.pad(
        #         caption, tf.convert_to_tensor([[0, config.max_length - tf.shape(caption)[0]]])
        #     )
        # else:
        #     padded_caption = caption[:config.max_length]

        return sample_id, processed_image, padded_caption


    def __init__(self, mode):
        if not (mode=="train" or mode=="test" or mode=="val"):
            raise Exception("Incorrect mode: {}".format(mode))

        if mode=="train":
            glob_string = "train-*" 
            self.total_size = 586368
        elif mode=="test":
            glob_string = "test-*"
            self.total_size = 20267
        else:
            glob_string = "val-*"
            self.total_size = 10132


        record_filenames = glob(os.path.join(config.input_dir, glob_string))

        with tf.name_scope("load_images"):
            dataset = tf.data.TFRecordDataset(record_filenames)
            
            dataset = dataset.map(self._extract_fn)
            dataset = dataset.repeat()
            dataset = dataset.batch(config.batch_size)

        self.dataset = dataset

    def next_batch(self):
        return self.dataset.make_one_shot_iterator().get_next()

class TestDataset:

    def __init__(self, json_file, inputs, input_dir, batch_size):
        data = json.load(open(json_file, "rb"))
        self._id_to_file = data["metadata"]
        self._inputs = inputs
        self._batch_size = batch_size
        self._curr_index = 0
        self._input_dir = input_dir
        if inputs=="image":
            self._images = data["images"]
        elif inputs=="text":
            self._captions = data["captions"]
        else:
            raise("Error")

    def _process_image(self, filepath):
        f = open(filepath,"rb")
        encoded_image = f.read()
        f.close()
        img = tf.image.decode_jpeg(encoded_image, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize_images(
            img, size=[config.image_size, config.image_size], method=tf.image.ResizeMethod.BILINEAR
        )

        # range: [0,1] ==> [-1,+1]
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)

        return img

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

    def next_batch():
        if self._inputs == "image":
            temp = self._curr_index + self.batch_size
            temp = max(temp, len(self._images))
            batch = self._images[self._curr_index:temp]
            self._curr_index = temp
            for data_item in batch:
                filepath = os.path.join(self._input_dir, self._id_to_file[str(data_item["image_id"])])
                data_item["processed_input"] = self._process_image(filepath)
                data_item["processed_choice_list"] = []
                for ch in data_item["choice_list"]:
                    data_item["processed_choice_list"].append(self._process_caption(ch))
        else:
            temp = self._curr_index + self.batch_size
            temp = max(temp, len(self._captions))
            batch = self._captions[self._curr_index:temp]
            self._curr_index = temp
            for data_item in batch:
                caption = data_item["caption"]
                data_item["processed_input"] = self._process_caption(caption)
                data_item["processed_choice_list"] = []
                for img_id in data_item["choice_list"]:
                    filepath = os.path.join(self._input_dir, self._id_to_file[str(img_id)])
                    data_item["processed_choice_list"].append(self._process_image(filepath))

        sess = tf.Session()

        for data_item in batch:
            data_item["processed_input"], data_item["processed_choice_list"] = sess.run(
                data_item["processed_input"],
                data_item["processed_choice_list"]
            )

        return batch

    def max_steps(self):
        if self._inputs == "image":
            array = self._images
        else:
            array = self._captions

        if len(array)%self._batch_size==0:
            return len(array)//self.batch_size
        else:
            return (len(array)//self.batch_size) + 1
                


if __name__ == "__main__":
    x = MSCOCODataset("val").next_batch()
    print(x)
    sess = tf.Session()
    y = sess.run(x)
    print(y)
