import tensorflow as tf
from ops import *

from IPython import embed
slim = tf.contrib.slim

def base_g_zx(model, z, reuse=False):
    n_layer = 5

    bs = model.batch_size
    w_start = model.image_shape[0]/2**(n_layer)
    c_start = model.c_dim * 2**(n_layer)

    with tf.variable_scope('g_z_to_x', reuse=reuse) as scope:
        h = z
        h = slim.fully_connected(h, w_start*w_start*c_start, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = slim.fully_connected(h, w_start*w_start*c_start, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = tf.reshape(h, [-1, w_start, w_start, c_start])

        for i in range(1, n_layer):
            c = c_start/(2**i)
            h = slim.conv2d_transpose(h, c, 1, 1, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
            h = slim.conv2d_transpose(h, c, 2, 1, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
            h = slim.conv2d_transpose(h, c, 2, 2, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)

        h = slim.conv2d_transpose(h, model.c_dim, 2, 2, activation_fn=None)#tf.nn.sigmoid)
        return tf.nn.sigmoid(h + tf.transpose(h, [0,2,1,3]))

def base_g_xz(model, x, reuse=False):
    n_layer = 5

    bs = model.batch_size

    with tf.variable_scope('g_x_to_z', reuse=reuse) as scope:

        h = x
        for i in range(n_layer):
            input_channel = h.get_shape().as_list()[-1]
            h = slim.conv2d(h, input_channel*2, 1, 1, activation_fn=lrelu, normalizer_fn=slim.batch_norm)
            h = slim.conv2d(h, input_channel*2, 2, 1, activation_fn=lrelu, normalizer_fn=slim.batch_norm)
            h = slim.conv2d(h, input_channel*2, 2, 2, activation_fn=lrelu, normalizer_fn=slim.batch_norm)

        h = tf.reshape(h, [bs, -1])
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        h = slim.fully_connected(h, model.z_dim*16, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
        mu = slim.fully_connected(h, model.z_dim, activation_fn=None)
        logvar = slim.fully_connected(h, model.z_dim, activation_fn=None)

    return mu, logvar


