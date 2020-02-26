import os
import shutil
import pickle as pkl
import argparse

import tensorflow as tf
import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler, scale

from utils import *


class cycModel_MM(object):
    def __init__(self,input_params,outputs,param_dim,output_dim,L_adv,L_cyc,L_rec):
        '''
        multi-modal cycGAN class
        x: parameters in 11-D
        y_sca: scalars: 1x21
        y_img: latent space of images:
        forward assumed to be x-->y-->x
        we will model the following multi-task problem:
        y_img,y_sca = f(x)
        '''

        self.input = input_params
        self.input_dim = param_dim
        self.output = outputs
        self.output_dim = output_dim
        self.lamda_adv = L_adv
        self.lamda_cyc = L_cyc
        self.lamda_rec = L_rec


    def fc_disc0(self,train_mode,reuse=False):

        x = tf.concat([self.output,self.input], axis=1)

        if reuse:
            x = tf.concat([self.output_fake,self.input_cyc], axis=1)
        x_dim = self.output_dim + self.input_dim

        with tf.variable_scope("disc0",reuse=reuse):
            D_W1 = weight_variable([x_dim, 1024],name="D_W1")
            D_b1 = bias_variable([1024],name="D_b1")
            h_fc1 = tf.matmul(x, D_W1) + D_b1
            D_relu1 = tf.nn.leaky_relu(bn(h_fc1,train_mode,name="d_bn1"))

            D_W2 = weight_variable([1024, 256],name="D_W2")
            D_b2 = bias_variable([256],name="D_b2")
            h_fc2 = tf.matmul(D_relu1, D_W2) + D_b2
            D_relu2 = tf.nn.leaky_relu(bn(h_fc2,train_mode,name="D_bn2"))


            D_W4 = weight_variable([256, 256],name="D_W4")
            D_b4 = bias_variable([256],name="D_b4")
            h_fc4 = tf.matmul(D_relu2, D_W4) + D_b4
            D_relu4 = tf.nn.leaky_relu(bn(h_fc4,train_mode,name="D_bn4"))

            D_W3 = weight_variable([256, 1],name="D_W3")
            D_b3 = bias_variable([1],name="D_b3")
            out = tf.matmul(D_relu4, D_W3) + D_b3


            return out, tf.nn.sigmoid(out)

    def fc_disc1(self,train_mode,reuse=False):

        x = tf.concat([self.output,self.input], axis=1)
        if cyc:
            x = tf.concat([self.output_fake,self.input_cyc], axis=1)
        x_dim = self.output_dim + self.input_dim
        _x = tf.nn.tanh(x)

        with tf.variable_scope("disc1",reuse=reuse):
            D_W1 = weight_variable([x_dim, 256],name="D2_W1")
            D_b1 = bias_variable([256],name="D2_b1")
            h_fc1 = tf.matmul(_x, D_W1) + D_b1
            D_relu1 = lrelu(bn(h_fc1,train_mode,name="D2_bn1"))

            D_W2 = weight_variable([256, 64],name="D2_W2")
            D_b2 = bias_variable([16],name="D2_b2")
            h_fc2 = tf.matmul(D_relu1, D_W2) + D_b2
            D_relu2 = lrelu(bn(h_fc2,train_mode,name="D2_bn2"))

            D_W3 = weight_variable([64, 1],name="D2_W3")
            D_b3 = bias_variable([1],name="D2_b3")
            h_fc3 = tf.matmul(D_relu2, D_W3) + D_b3
            out = bn(h_fc3,train_mode,name="D2_bn3")

            return out, tf.nn.sigmoid(out)

    def fc_gen0(self,train_mode,cyc=False,reuse=False):
        x = self.input
        if cyc:
            x = self.input_fake

        x_dim = self.input_dim
        y_dim = self.output_dim

        with tf.variable_scope("gen0",reuse=reuse):
            G_W1 = weight_variable([x_dim, 32],name="G_W1")
            G_b1 = bias_variable([32],name="G_b1")
            h_fc1 = tf.matmul(x, G_W1) + G_b1
            G_relu1 = tf.nn.relu(bn(h_fc1,train_mode,name="G_bn1"))
            # G_relu1 = tf.nn.relu(h_fc1)

            G_W2 = weight_variable([32, 256],name="G_W2")
            G_b2 = bias_variable([256],name="G_b2")
            h_fc2 = tf.matmul(G_relu1, G_W2) + G_b2
            G_relu2 = tf.nn.relu(bn(h_fc2,train_mode,name="G_bn2"))
            # G_relu2 = tf.nn.relu(h_fc2)

            G_W3 = weight_variable([256, 1024],name="G_W3")
            G_b3 = bias_variable([1024],name="G_b3")
            h_fc3 = tf.matmul(G_relu2, G_W3) + G_b3
            G_relu3 = tf.nn.relu(bn(h_fc3,train_mode,name="G_bn3"))
            # G_relu3 = tf.nn.relu(h_fc3)

            G_W4 = weight_variable([1024, y_dim],name="G_W4")
            G_b4 = bias_variable([y_dim],name="G_b4")
            h_fc4 = tf.matmul(G_relu3, G_W4) + G_b4
            out =  h_fc4

        return out


    def fc_gen1(self,train_mode,cyc=False,reuse=False):
        x = self.output
        if cyc:
            x = self.output_fake

        x_dim = self.output_dim
        y_dim = self.input_dim

        with tf.variable_scope("gen1",reuse=reuse):
            G_W1 = weight_variable([x_dim, 16],name="G_W1")
            G_b1 = bias_variable([16],name="G_b1")
            h_fc1 = tf.matmul(x, G_W1) + G_b1
            G_relu1 = tf.nn.relu(bn(h_fc1,train_mode,name="G_bn1"))

            G_W2 = weight_variable([16, 128],name="G_W2")
            G_b2 = bias_variable([128],name="G_b2")
            h_fc2 = tf.matmul(G_relu1, G_W2) + G_b2
            G_relu2 = tf.nn.relu(bn(h_fc2,train_mode,name="G_bn2"))

            G_W3 = weight_variable([128, 512],name="G_W3")
            G_b3 = bias_variable([512],name="G_b3")
            h_fc3 = tf.matmul(G_relu2, G_W3) + G_b3
            G_relu3 = tf.nn.relu(bn(h_fc3,train_mode,name="G_bn3"))


            G_W4 = weight_variable([512, y_dim],name="G_W4")
            G_b4 = bias_variable([y_dim],name="G_b4")
            h_fc4 = tf.matmul(G_relu3, G_W4) + G_b4
            out = bn(h_fc4,train_mode,name="G_bn4")

            return out


    def run(self,train_mode):
        self.output_fake = self.fc_gen0(train_mode)
        self.input_cyc = self.fc_gen1(train_mode,cyc=True)

        D_real,d_out_real = self.fc_disc0(train_mode)
        D_fake,d_out_fake = self.fc_disc0(train_mode,reuse=True)

        D_loss1,G_adv1 = GANloss(D_real,D_fake)

        output_m = self.output
        output_fake_m = self.output_fake

        L_cyc_x = tf.reduce_mean(tf.abs(self.input_cyc-self.input))

        self.L_l2_y =  tf.reduce_mean(tf.abs(output_fake_m-output_m))
        self.L_l2_x =  L_cyc_x

        L_cyc = L_cyc_x #+ L_cyc_y
        self.loss_adv = G_adv1
        self.loss_gen0 = self.lamda_adv*G_adv1 + self.lamda_cyc*L_cyc + self.lamda_rec*self.L_l2_y
        self.loss_gen1  = L_cyc + self.lamda_adv*G_adv1

        self.loss_disc = D_loss1

        t_vars =  tf.trainable_variables()
        self.disc_vars = [var for var in t_vars if 'disc0' in var.name]
        self.gen0_vars = [var for var in t_vars if 'gen0' in var.name]
        self.gen1_vars = [var for var in t_vars if 'gen1' in var.name]

        self.D_solver = tf.train.AdamOptimizer(2e-4).minimize(self.loss_disc, var_list=self.disc_vars)
        self.G0_solver = tf.train.AdamOptimizer(2e-4).minimize(self.loss_gen0+self.loss_gen1, var_list=self.gen0_vars+self.gen1_vars)
        # self.G1_solver = tf.train.AdamOptimizer(2e-4).minimize(self.loss_gen1, var_list=self.gen1_vars)

        return
