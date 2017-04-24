import tensorflow as tf
from batch import Batch 
from model import Model

class Evaluator(object):
    def __init__(self,path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)

    def evaluate(self,path_to_checkpoint,path_to_tfrecords_file,num_examples,global_step):
        batch_size = 32 # testing example by example
        num_batches = num_examples / batch_size 

        with tf.Graph().as_default():
            image_batch,digits_batch = Batch.build_batch(path_to_tfrecords_file,num_examples,batch_size=batch_size)
            print image_batch.shape
            digits_logits = Model.inference(image_batch,drop_rate=0.0)
            digits_predictions=tf.argmax(digits_logits,axis=2)

            labels = digits_batch
            predictions = digits_predictions

            labels_string = tf.reduce_join(tf.as_string(labels),axis=1)
            predictions_string = tf.reduce_join(tf.as_string(predictions),axis=1)

            accuracy, update_accuracy = tf.metrics.accuracy(labels=labels_string,predictions=predictions_string)
            tf.summary.image('image',image_batch)
            tf.summary.scalar('accuracy',accuracy)
            tf.summary.histogram('variables',tf.concat([tf.reshape(var,[-1]) for var in tf.trainable_variables()],axis=0))
            summary = tf.summary.merge_all()
            with tf.Session() as sess:
                sess.run([tf.initialize_all_variables(),tf.initialize_local_variables()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess,path_to_checkpoint)

                for _ in xrange(num_batches):
                    sess.run(update_accuracy)

                accuracy_val,summary_val = sess.run([accuracy,summary])
                self.summary_writer.add_summary(summary_val,global_step=global_step)

                coord.request_stop()
                coord.join(threads)

        return accuracy_val