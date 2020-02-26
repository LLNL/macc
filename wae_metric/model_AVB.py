'''Models'''

import tensorflow as tf

from .utils import *
from .utils import spectral_norm as SN


def xavier_init(size,name=None):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev,name=name)

'''FCN Encoder'''
def gen_encoder_FCN(x, n_output, train_mode,reuse=False):
    n_hidden = [32,256,128]
    # n_hidden = [1024,256,32]

    with tf.variable_scope("wae_encoder",reuse=reuse):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        # inputs = tf.concat(axis=1, values=[x, eps])
        inputs = x
        # w0 = tf.get_variable('w0', [x.get_shape()[1] + eps.get_shape()[1], n_hidden[0]], initializer=w_init)
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden[0]], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden[0]], initializer=b_init)
        h0 = tf.matmul(inputs, w0) + b0
        h0 = bn(h0, train_mode,"bn1")
        h0 = tf.nn.elu(h0)



        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden[1]], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden[1]], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = bn(h1, train_mode,"bn2")
        h1 = tf.nn.tanh(h1)



        # 3rd hidden layer
        w2 = tf.get_variable('w2', [h1.get_shape()[1], n_hidden[2]], initializer=w_init)
        b2 = tf.get_variable('b2', [n_hidden[2]], initializer=b_init)
        h2 = tf.matmul(h1, w2) + b2
        h2 = bn(h2, train_mode,"bn3")
        h2 = tf.nn.tanh(h2)

        # h2 = tf.nn.tanh(h2)

        # output layer
        wout = tf.get_variable('wout', [h2.get_shape()[1], n_output], initializer=w_init)
        bout = tf.get_variable('bout', [n_output], initializer=b_init)
        z = tf.matmul(h2, wout) + bout

    return z



def var_decoder_FCN(z, n_output, train_mode, reuse=False):
    n_hidden = [64,128,256]
    # n_hidden = [32,256,1024]
    with tf.variable_scope("wae_decoder", reuse=reuse):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        dw0 = tf.get_variable('dw0', [z.get_shape()[1], n_hidden[0]], initializer=w_init)
        db0 = tf.get_variable('db0', [n_hidden[0]], initializer=b_init)
        h0 = tf.matmul(z, dw0) + db0
        # h0 = bn(h0,train_mode,"bn0")
        h0 = tf.nn.elu(h0)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden[1]], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden[1]], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        # h1 = bn(h1,train_mode,"bn1")
        h1 = tf.nn.tanh(h1)



        # 3rd hidden layer
        w2 = tf.get_variable('w2', [h1.get_shape()[1], n_hidden[2]], initializer=w_init)
        b2 = tf.get_variable('b2', [n_hidden[2]], initializer=b_init)
        h2 = tf.matmul(h1, w2) + b2
        # h2 = bn(h2,train_mode,"bn2")
        h2 = tf.nn.tanh(h2)



        # output layer
        wout = tf.get_variable('wout', [h2.get_shape()[1], n_output], initializer=w_init)
        bout = tf.get_variable('bout', [n_output], initializer=b_init)
        recon = tf.matmul(h2, wout) + bout

        return recon


def discriminator_FCN(x, z, r=None):
    with tf.variable_scope("discriminator", reuse=r):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        D_W1 = tf.get_variable('D_W1', [x.get_shape()[1] + z.get_shape()[1], 512], initializer=w_init)
        D_b1 = tf.get_variable('D_b1', [512], initializer=b_init)
        inputs = tf.concat([x, z], axis=1)
        h1 = tf.nn.leaky_relu(tf.matmul(inputs, SN(D_W1,"sn1")) + D_b1)

        D_W2 = tf.get_variable('D_W2', [512, 256], initializer=w_init)
        D_b2 = tf.get_variable('D_b2', [256], initializer=b_init)
        h2 = tf.nn.leaky_relu(tf.matmul(h1, SN(D_W2,"sn2")) + D_b2)

        D_W3 = tf.get_variable('D_W3', [256, 128], initializer=w_init)
        D_b3 = tf.get_variable('D_b3', [128], initializer=b_init)
        h3 = tf.nn.leaky_relu(tf.matmul(h2, SN(D_W3,"sn3")) + D_b3)

        D_W4 = tf.get_variable('D_W4', [128, 64], initializer=w_init)
        D_b4 = tf.get_variable('D_b4', [64], initializer=b_init)
        h4 = tf.nn.leaky_relu(tf.matmul(h3, SN(D_W4,"sn4")) + D_b4)

        D_W5 = tf.get_variable('D_W5', [64, 1], initializer=w_init)
        D_b5 = tf.get_variable('D_b5', [1], initializer=b_init)
        prob = tf.matmul(h4, SN(D_W5,"sn5")) + D_b5

    return prob
