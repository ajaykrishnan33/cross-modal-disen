import tensorflow as tf
import config
from data_ops import MSCOCODataset

# comment out dataset.repeat from MSCOCODataset before running this
x, y = MSCOCODataset(config.mode).next_batch()
lenx = tf.shape(x)[0]
sess = tf.Session()

total_len = 0
while True:
	actual_lenx = sess.run(lenx)
	if actual_lenx:
		total_len += actual_lenx
	else:
		break

print("Total dataset size:", total_len)


