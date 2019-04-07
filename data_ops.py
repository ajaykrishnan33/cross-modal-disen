import tensorflow as tf
from glob import glob
import config
import os

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
                "image/data": tf.FixedLenFeature([], dtype=tf.string)
            },
            sequence_features={
                "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64)
            }
        )

        encoded_image = context["image/data"]
        processed_image = self._process_image(encoded_image)

        caption = sequence["image/caption_ids"]

        if tf.shape(caption)[0] < config.max_length:
            padded_caption = tf.pad(
                caption, tf.convert_to_tensor([[0, config.max_length - tf.shape(caption)[0]]])
            )
        else:
            padded_caption = caption[:config.max_length]

        return processed_image, padded_caption


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

if __name__ == "__main__":
    x = MSCOCODataset("val").next_batch()
    print(x)
    sess = tf.Session()
    y = sess.run(x)
    print(y)
