import tensorflow as tf
from ops import *

from IPython import embed
slim = tf.contrib.slim

def base_g_zx(model, z, reuse=False):
    if model.dataset_name == 'mnist':
        n_layer = 2
    else:
        n_layer = 5

    bs = model.batch_size
    w_start = model.image_shape[0]/2**(n_layer)
    c_start = model.c_dim * 2**(n_layer)

    with tf.variable_scope('g_z_to_x', reuse=reuse) as scope:
        h = z
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        # h = slim.fully_connected(h, w_start*w_start*c_start, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = tf.reshape(h, [model.batch_size, w_start, w_start, -1])
        c_start = h.get_shape().as_list()[-1]

        for i in range(1, n_layer):
            c = c_start/(2**i)
            h = slim.conv2d_transpose(h, c, 2, 2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
#            w = slim.conv2d(h, c, 1, 1, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
#            w_ = tf.transpose(w, perm=[0,3,1,2])
#            h_ = tf.transpose(h, perm=[0,3,1,2])
#            h_ = tf.matmul(h_, w_)
#            h_ = tf.transpose(h_, perm=[0,2,3,1])
#            h = h + h_
            height, width = h.get_shape().as_list()[1:3]
            #h = tf.image.resize_nearest_neighbor(h, [2*height, 2*width])
            #h = slim.conv2d(h, c, [2, 2], 2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)

        h = slim.conv2d_transpose(h, model.c_dim, 2, 2, activation_fn=None, normalizer_fn=None)
        return tf.nn.sigmoid(h + tf.transpose(h, [0,2,1,3]))

def base_g_xz(model, x, reuse=False):
    if model.dataset_name == 'mnist':
        n_layer = 2
    else:
        n_layer = 5

    bs = model.batch_size

    with tf.variable_scope('g_x_to_z', reuse=reuse) as scope:

        h = x
        for i in range(n_layer):
            input_channel = h.get_shape().as_list()[-1]
            if i == 0:
              h = slim.conv2d(h, 16, 2, 2, activation_fn=lrelu, normalizer_fn=slim.batch_norm)
            else:
#            w = slim.conv2d(h, input_channel, 1, 1, activation_fn=lrelu, normalizer_fn=slim.batch_norm)
#            w_ = tf.transpose(w, perm=[0,3,1,2])
#            h_ = tf.transpose(h, perm=[0,3,1,2])
#            h_ = tf.matmul(h_, w_)
#            h_ = tf.transpose(h_, perm=[0,2,3,1])
#            h = h + h_
              h = slim.conv2d(h, input_channel*2, 2, 2, activation_fn=lrelu, normalizer_fn=slim.batch_norm)

        h = tf.reshape(h, [bs, -1])
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        mu = slim.fully_connected(h, model.z_dim, activation_fn=None)
        logvar = slim.fully_connected(h, model.z_dim, activation_fn=None)

    return mu, logvar


def cross_g_zx(model, z, reuse=False):
    if model.dataset_name == 'mnist':
        n_layer = 2
    else:
        n_layer = 5

    bs = model.batch_size
    w_start = model.image_shape[0]/2**(n_layer)
    c_start = model.c_dim * 2**(n_layer)

    with tf.variable_scope('g_z_to_x', reuse=reuse) as scope:
        h = z
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        # h = slim.fully_connected(h, w_start*w_start*c_start, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = tf.reshape(h, [model.batch_size, w_start, w_start, -1])
        c_start = h.get_shape().as_list()[-1]

        for i in range(1, n_layer):
            c = c_start/(2**i)
            h = slim.conv2d_transpose(h, c, 2, 2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
#            w = slim.conv2d(h, c, 1, 1, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
#            w_ = tf.transpose(w, perm=[0,3,1,2])
#            h_ = tf.transpose(h, perm=[0,3,1,2])
#            h_ = tf.matmul(h_, w_)
#            h_ = tf.transpose(h_, perm=[0,2,3,1])
#            h = h + h_
            height, width = h.get_shape().as_list()[1:3]
            #h = tf.image.resize_nearest_neighbor(h, [2*height, 2*width])
            #h = slim.conv2d(h, c, [2, 2], 2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)

        h = slim.conv2d_transpose(h, model.c_dim, 2, 2, activation_fn=None, normalizer_fn=None)
        return tf.nn.sigmoid(h + tf.transpose(h, [0,2,1,3]))


def cross_g_xz(model, x, reuse=False):
    if model.dataset_name == 'mnist':
        n_layer = 2
    else:
        n_layer = 5

    bs = model.batch_size

    with tf.variable_scope('g_x_to_z', reuse=reuse) as scope:

        h = x
        for i in range(n_layer):
            input_channel = h.get_shape().as_list()[-1]
            w = slim.conv2d(h, input_channel, 1, 1, activation_fn=lrelu, normalizer_fn=slim.batch_norm)
            r = tf.reduce_sum(h, axis=1, keep_dims=True)
            r = lrelu(r)
            c = tf.reduce_sum(h, axis=2, keep_dims=True)
            c = lrelu(c)

            if i == 0:
              h = slim.conv2d(h, 16, 2, 2, activation_fn=lrelu, normalizer_fn=slim.batch_norm)
            else:
#            w_ = tf.transpose(w, perm=[0,3,1,2])
#            h_ = tf.transpose(h, perm=[0,3,1,2])
#            h_ = tf.matmul(h_, w_)
#            h_ = tf.transpose(h_, perm=[0,2,3,1])
#            h = h + h_
              h = slim.conv2d(h, input_channel*2, 2, 2, activation_fn=lrelu, normalizer_fn=slim.batch_norm)

        h = tf.reshape(h, [bs, -1])
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        mu = slim.fully_connected(h, model.z_dim, activation_fn=None)
        logvar = slim.fully_connected(h, model.z_dim, activation_fn=None)

    return mu, logvar

def base_generator(model, z, reuse=False):
    if model.dataset_name == 'mnist':
        n_layer = 2
    else:
        n_layer = 4

    bs = model.batch_size
    w_start = model.image_shape[0]/2**(n_layer)
    c_start = model.c_dim * 2**(n_layer)

    with tf.variable_scope('g_') as scope:
        if reuse:
            scope.reuse_variables()

        h = tf.nn.relu(bn(linear(z, w_start*w_start*c_start, 'h0_lin', stddev=0.05), 'bn_0_lin'))
        h = tf.reshape(h, [-1, w_start, w_start, c_start])

        for i in range(1, n_layer):
            out_shape = [model.batch_size]+[w_start*2**i]*2+[c_start*2**i]
            h = tf.nn.relu(bn(deconv2d(h, out_shape, stddev=0.05, name='h%d'%i), 'bn%d'%i))
            h = tf.nn.relu(bn(conv2d(h, out_shape[-1], k=3, d=1, stddev=0.02, name='h%d_'%i), 'bn%d_'%i))

        i += 1
        out_shape = [model.batch_size]+[w_start*2**i]*2+[c_start*2**i]
        h = tf.nn.relu(bn(deconv2d(h, out_shape, stddev=0.05, name='h%d'%i), 'bn%d'%i))
        x = tf.nn.tanh(bn(conv2d(h, model.c_dim, k=3, d=1, stddev=0.02, name='h%d_'%i), 'bn%d_'%i))

    return x

