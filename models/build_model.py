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
    model.kl_mult = tf.placeholder(tf.float32)
    model.lr_mult = tf.placeholder(tf.float32)

    A = tf.reshape(model.image, [model.batch_size]+model.image_shape[:-1])
    A2 = tf.matmul(A, A)
    A3 = tf.matmul(A2, A)
    I = tf.constant(np.identity(model.image_shape[0], dtype=np.float32))
    I = tf.tile(tf.reshape(I, [1, model.image_shape[0], model.image_shape[0]]), [model.batch_size, 1, 1])
    augmented = tf.stack([A, A2, A3, I], axis=3)

    model.z = make_z(shape=[model.batch_size, model.z_dim])
    model.gen_image = model.decoder(model.z)
    z_mu, z_logvar = model.encoder(augmented)
    model.z_ = reparameterize(z_mu, z_logvar)
    model.image_ = model.decoder(model.z_, reuse=True)

    recon_loss = -tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(
        model.image * tf.log(1e-5 + model.image_) + \
        (1-model.image) * tf.log(1e-5 + 1 - model.image_), 1), 1))
    # bernoulli
    kl_div = 0.5*tf.reduce_mean(tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - z_logvar, 1))
    loss = recon_loss + kl_div * model.kl_mult

    # optimizer
    model.get_vars()
    # opt = tf.train.GradientDescentOptimizer(model.lr_mult * config.learning_rate)
    opt = tf.train.AdamOptimizer(1000.0 * config.learning_rate)
    optimize = slim.learning.create_train_op(loss, opt, clip_gradient_norm=10000.0,
                                             variables_to_train=tf.trainable_variables())

    # logging
    tf.summary.scalar("recon_loss", recon_loss)
    tf.summary.scalar("kl_div", kl_div)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("kl_mult", model.kl_mult)
    # [tf.summary.histogram(x.name, x) for x in tf.trainable_variables()]
    model.recon_loss = recon_loss
    model.kl_div = kl_div
    model.loss = loss

    model.saver = tf.train.Saver(max_to_keep=None)

    return optimize
