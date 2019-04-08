import tensorflow as tf
from glob import glob
import config
import os

class MSCOCODataset:

    def _process_image(self, encoded_image):
        img = tf.image.decode_jpeg(encoded_image, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # img = tf.image.resize_images(
        #     img, size=[config.image_size, config.image_size], method=tf.image.ResizeMethod.BILINEAR
        # )

        # range: [0,1] ==> [-1,+1]
        # img = tf.subtract(img, 0.5)
        # img = tf.multiply(img, 2.0)

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

        sample_id = str(context["image/image_id"].decode('utf8'))
        encoded_image = context["image/data"]
        processed_image = self._process_image(encoded_image)

        caption = sequence["image/caption_ids"]

        # padded_caption = tf.pad(caption, tf.convert_to_tensor([[0, config.max_length - tf.shape(caption)[0]]]))

        return sample_id, processed_image, caption


    def __init__(self, mode):
        if not (mode=="train" or mode=="test" or mode=="val"):
            raise Exception("Incorrect mode: {}".format(mode))

        if mode=="train":
            glob_string = "train-*" 
        elif mode=="test":
            glob_string = "test-*"
        else:
            glob_string = "val-*"

        record_filenames = glob(os.path.join(config.input_dir, glob_string))

        with tf.name_scope("load_images"):
            dataset = tf.data.TFRecordDataset(record_filenames)
            
            dataset = dataset.map(self._extract_fn)
            # dataset = dataset.repeat()
            dataset = dataset.batch(1)

        self.dataset = dataset

    def next_batch(self):
        return self.dataset.make_one_shot_iterator().get_next()

# comment out dataset.repeat from MSCOCODataset before running this
dataset = MSCOCODataset("val")
p, x, y = dataset.next_batch()
sess = tf.Session()

total_len = 0
max_caption_length = 0
length_freq = {}
aspect_ratio_freq = {}
while True:
    try:
        sample_ids, images, captions = sess.run((p,x,y))
        print(sample_ids)
        if captions.shape[1] > max_caption_length:
            max_caption_length = captions.shape[1]
        length_freq.setdefault(captions.shape[1], 0)
        length_freq[captions.shape[1]] += 1

        ar = images.shape[2]/images.shape[1]
        aspect_ratio_freq.setdefault(ar, 0)
        aspect_ratio_freq[ar] += 1

        total_len += 1
    except tf.errors.OutOfRangeError:
        break

print("Total dataset size:", total_len)
print("Max caption length:", max_caption_length)

import json

with open("length_freq.json", "w") as f:
    f.write(json.dumps(length_freq, sort_keys=True, indent=4))

with open("aspect_ratio_freq.json", "w") as f:
    f.write(json.dumps(aspect_ratio_freq, sort_keys=True, indent=4))

