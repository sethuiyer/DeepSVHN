import os
import tensorflow as tf
from model import Model 
from batch import Batch
import datetime
import time
from evaluator import Evaluator
### define the hyper parameters ###


batch_size = 32
learning_rate=1e-2
patience = 2
decay_steps = 10000
decay_rate=0.9
### ends here ###

def _train(path_to_train_tfrecords_file,num_train_examples,path_to_val_tfrecords_file,num_val_examples,path_to_train_log_dir,path_to_restore_checkpoint_file):
    global patience
    global batch_size
    global learning_rate
    global decay_rate
    global decay_steps

    num_steps_to_show_loss = 100
    num_steps_to_check = 1000 
    initial_patience = patience
    with tf.Graph().as_default():
        image_batch,digits_batch = Batch.build_batch(path_to_train_tfrecords_file,num_examples=num_train_examples,batch_size=batch_size)
        print image_batch.shape
        digit_logits=Model.inference(image_batch,drop_rate=0.2)
        loss = Model.loss(digit_logits,digits_batch)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        learning_rate_tf = tf.train.exponential_decay(learning_rate,global_step=global_step,decay_steps=decay_steps,decay_rate=decay_rate)
        optimizer=tf.train.GradientDescentOptimizer(learning_rate)
        train_op=optimizer.minimize(loss,global_step=global_step)

        tf.summary.image('image',image_batch)
        tf.summary.scalar('loss',loss)
        tf.summary.scalar('learning_rate',learning_rate)
        summary =tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir,sess.graph)
            evaluator = Evaluator(os.path.join(path_to_train_log_dir,'eval/val'))
            sess.run(tf.initialize_all_variables())
            coord = tf.train.Coordinator()
            threads= tf.train.start_queue_runners(sess=sess,coord=coord)

            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), '%s not found '% path_to_restore_checkpoint_file
                saver.restore(sess,path_to_restore_checkpoint_file)
                print 'Model restored from file: %s' %path_to_restore_checkpoint_file
            print 'Start training'
            patience = initial_patience  # just like epochs
            best_accuracy = 0.0 
            duration = 0.0 

            while True:
                start_time = time.time()
                _,loss_val,summary_val,global_step_val,learning_rate_val = sess.run([train_op,loss,summary,global_step,learning_rate_tf])
                duration +=time.time() - start_time
                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0 
                    print '=> step %d, loss = %d )%.1f examples/sec)' %(global_step_val,loss_val,examples_per_sec)
                if global_step_val % num_steps_to_check != 0:
                    continue

                summary_writer.add_summary(summary_val,global_step=global_step_val)
                print '=> Evaluatiing on validation dataset ....'
                path_to_latest_checkpoint_file = saver.save(sess,os.path.join(path_to_train_log_dir,'latest.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file,path_to_val_tfrecords_file,num_val_examples,global_step_val)
                print ' ==> accuracy = %f , best accuracy %f ' % (accuracy,best_accuracy)

                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess,os.path.join(path_to_train_log_dir,'model.ckpt'),global_step=global_step_val)
                    print ' =>Model saved to file %s ' % path_to_checkpoint_file
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1

                print '=> pateince = %d ' % patience 
                if patience == 0:
                    break
            coord.request_stop()
            coord.join(threads)
            print 'Finished'
    return path_to_checkpoint_file