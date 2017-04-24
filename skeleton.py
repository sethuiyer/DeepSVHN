import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #surpress tensorflow warnings
import tensorflow as tf
from model import Model 
import train
import convert_to_tfrecords
import random
import cPickle as pickle
import gzip
import numpy as np
import cv2
class SVHN(object):

    path = ""

    def __init__(self, data_dir):
        """
            data_directory : path like /home/rajat/mlproj/dataset/
                            includes the dataset folder with '/'
            Initialize all your variables here
        """
        self.path_to_train_dir = data_dir
        self.path_to_train_csv = data_dir+'train.csv'
        self.path_to_train_tfrecords_file = [data_dir+'train.tfrecords',data_dir+'val.tfrecords']
        self.path_to_checkpoint_file = './logs/train/model.ckpt-13000'
        for path_to_tfrecords_file in self.path_to_train_tfrecords_file:
            if not os.path.exists(path_to_tfrecords_file) and not os.path.exists(path_to_tfrecords_file):
                print 'Processing training data ....'
                self.num_train,self.num_val=convert_to_tfrecords.convert_to_tfrecords(self.path_to_train_dir,self.path_to_train_tfrecords_file, lambda paths: 0 if random.random() > 0.25 else 1)
                attrs={}
                attrs['num_train'] = self.num_train
                attrs['num_val'] = self.num_val
                pickle.dump(attrs,open('meta.pb','w'))
                print 'Done.'
        self.images = tf.placeholder(tf.float32,shape=(1,54,54,3))
        self.digit_logits = Model.inference(self.images,drop_rate=0.0)
        self.digit_predict=tf.argmax(self.digit_logits,axis=2)
        if self.path_to_checkpoint_file == None:
            print 'No Checkpoint file found to get the output, please train first'
            return None

        self.session=tf.Session()
        restorer = tf.train.Saver()
        restorer.restore(self.session,self.path_to_checkpoint_file)

    def train(self):
        """
            Trains the model on data given in path/train.csv

            No return expected
        """
        self.path_to_train_tfrecords_file = self.path_to_train_dir+'train.tfrecords'
        self.path_to_val_tfrecords_file = self.path_to_train_dir+'val.tfrecords'
        self.path_to_train_log_dir = './logs/train'
        self.path_to_checkpoint_file=train._train(path_to_train_tfrecords_file,self.num_train,path_to_val_tfrecords_file,self.num_val,path_to_train_log_dir,self.path_to_checkpoint_file)




    def get_sequence(self, image):
        """
            image : a variable resolution RGB image in the form of a numpy array

            return: list of integers with the sequence of digits. Example: [5,0,3] for an image having 503 as the sequence.

        """
        image = tf.image.resize_images(image,[64,64])
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
        image = tf.multiply(tf.subtract(image,0.5),2)
        image = tf.image.resize_images(image,[54,54])
        images = tf.reshape(image,[1,54,54,3])
        with tf.Session() as sess:
            imag2=sess.run(images)
        digit_predictions=self.session.run([self.digit_predict],feed_dict={self.images: imag2})
        digit_predictions = digit_predictions[0][0]
        index = np.argwhere(digit_predictions==10)
        digit_predictions=np.delete(digit_predictions,index)
        return digit_predictions

    def save_model(self, **params):

        # file_name = params['name']
        # pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk

            no return expected
        """
        pickle.dump(self,gzip.open(params['name'],'wb'))

    @staticmethod
    def load_model(**params):

        # file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-trained instance of SVHN class
        """
        return pickle.load(gzip.open(params['name'],'rb'))

if __name__ == "__main__":
         obj = SVHN('release/data/')
         obj.save_model(name="svhn.gz")

