import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from model import Model 

tf.app.flags.DEFINE_string('image',None,'path to image file')

FLAGS = tf.app.flags.FLAGS

restore_checkpoint = './logs/train/model.ckpt-13000'

def main(_):
	path_to_image = FLAGS.image 
	image = tf.image.decode_png(tf.read_file(path_to_image),channels=3)
	image = tf.image.resize_images(image,[64,64])
	image = tf.image.convert_image_dtype(image,dtype=tf.float32)
	image = tf.multiply(tf.subtract(image,0.5),2)
	image = tf.image.resize_images(image,[54,54])
	images = tf.reshape(image,[1,54,54,3])
	digit_logits = Model.inference(images,drop_rate=0.0)
	digit_predict=tf.argmax(digit_logits,axis=2)

	with tf.Session() as sess:
		restorer = tf.train.Saver()
		restorer.restore(sess,restore_checkpoint)

		digit_predictions=sess.run([digit_predict])
		while digit_predictions[-1] == 10:
			_ = digit_predictions.pop(-1)
