import tensorflow as tf
from ops import make_z, reparameterize

from IPython import embed
import numpy as np
slim = tf.contrib.slim

def build_model(model):
    config = model.config

    # input
    model.image = tf.placeholder(tf.float32, shape=[model.batch_size]+model.image_shape)
    model.label = tf.placeholder(tf.float32, shape=[model.batch_size])

    A = tf.reshape(model.image, [model.batch_size]+model.image_shape[:-1])
    A2 = tf.matmul(A, A)
    A3 = tf.matmul(A2, A)
    I = tf.constant(np.identity(model.image_shape[0], dtype=np.float32))
    I = tf.tile(tf.reshape(I, [1, model.image_shape[0], model.image_shape[0]]), [model.batch_size, 1, 1])
    augmented = tf.stack([I, A, A2, A3], axis=3)

    model.z = make_z(shape=[model.batch_size, model.z_dim])
    z_mu, z_sigma = model.encoder(augmented)
    model.z_ = reparameterize(z_mu, z_sigma)
    model.image_ = model.decoder(model.z_)
    print model.image_.get_shape().as_list()

    recon_loss = -tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(
        model.image * tf.log(1e-10 + model.image_) + \
        (1-model.image) * tf.log(1e-10 + 1 - model.image_), 1), 1))
    # bernoulli
    kl_div = 0.5*tf.reduce_mean(tf.reduce_sum(z_sigma + z_mu**2 -  tf.log(z_sigma), 1))
    loss = recon_loss + kl_div

    # optimizer
    model.get_vars()
    opt = tf.train.AdamOptimizer(config.learning_rate)
    optimize = slim.learning.create_train_op(loss, opt, variables_to_train=tf.trainable_variables())

    # logging
    tf.summary.scalar("recon_loss", recon_loss)
    tf.summary.scalar("kl_div", kl_div)
    tf.summary.scalar("loss", loss)
    model.recon_loss = recon_loss
    model.kl_div = kl_div
    model.loss = loss

    model.saver = tf.train.Saver(max_to_keep=None)

    return optimize
