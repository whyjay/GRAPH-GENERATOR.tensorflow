import tensorflow as tf
from model import save_images
import os
import time
import numpy as np
from utils import *
from ops import *
from IPython import embed

def train(model, sess):
    optim = model.build_model()
    coord, threads, merged_sum = init_training(model, sess)
    start_time = time.time()
    print_time = time.time()

    dataset = load_dataset(model)
    N = dataset.num_examples
    max_iter = int(N/model.batch_size) * model.config.epoch

    print "[*] Traing Start : N=%d, Batch=%d, epoch=%d, max_iter=%d" \
        %(N, model.batch_size, model.config.epoch, max_iter)

    lr_mult_ = 1.0
    try:
        for idx in xrange(max_iter):
            batch_start_time = time.time()
            kl_mult_ = min(1.0, idx * 1e-4)
            lr_mult_ = lr_mult_ * 0.9999

            image = dataset.next_batch(model.batch_size)
            image = np.reshape(image, [model.batch_size, model.image_shape[0], model.image_shape[1], 1])
            _, summary = sess.run([optim, merged_sum], feed_dict={model.image:image,
                                                                  model.kl_mult:kl_mult_,
                                                                  model.lr_mult:lr_mult_})
            if idx % 10 == 0:
                model.writer.add_summary(summary, idx)

            # save checkpoint
            if (idx*model.batch_size) % N < model.batch_size and idx > 0:
                epoch = int(idx*model.batch_size/N)
                print_time = time.time()
                total_time = print_time - start_time
                sec_per_epoch = (print_time - start_time)
                _save_samples(model, sess, epoch, dataset)
                model.save(sess, model.checkpoint_dir, epoch)

                print '[Epoch %(epoch)d] time: %(total_time)4.4f, sec_per_epoch: %(sec_per_epoch)4.4f' % locals()

    except tf.errors.OutOfRangeError:
        print "Done training; epoch limit reached."
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

def _save_samples(model, sess, epoch, dataset):

    # generator hard codes the batch size
    for i in xrange(model.sample_size // model.batch_size):
      image = dataset.next_batch(model.batch_size)
      image = np.reshape(image, [model.batch_size, model.image_shape[0], model.image_shape[1], 1])
      gt, rec, img = sess.run([model.image, model.image_, model.gen_image], feed_dict={model.image:image})


    print  "Save Samples at %s/%s" % (model.sample_dir, 'samples_%s' % (epoch))
    with open(os.path.join(model.sample_dir, 'samples_gt_%d.npy'%(epoch)), 'w') as f:
        np.save(f, gt)
    with open(os.path.join(model.sample_dir, 'samples_rec_%d.npy'%(epoch)), 'w') as f:
        np.save(f, rec)
    with open(os.path.join(model.sample_dir, 'samples_%d.npy'%(epoch)), 'w') as f:
        np.save(f, img)

def init_training(model, sess):
    config = model.config
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    merged_sum = tf.summary.merge_all()
    model.writer = tf.summary.FileWriter(config.log_dir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if model.load(sess, model.checkpoint_dir):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    if not os.path.exists(config.dataset_path):
        print(" [!] Data does not exist : %s" % config.dataset_path)
    return coord, threads, merged_sum

def load_dataset(model):
    import sna
    return sna.read_data_sets(model.dataset_path, dtype=tf.uint8, reshape=False, validation_size=0).train
