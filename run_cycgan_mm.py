# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# @authors: :Rushil Anirudh, Jayaraman J. Thiagarajan, Timo Bremer
#
# SPDX-License-Identifier: MIT

import tensorflow as tf
import numpy as np
np.random.seed(4321)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from modelsv2 import *
from utils import *
import shutil
import cPickle as pkl
import argparse
from sklearn.preprocessing import MinMaxScaler, scale

import wae_metric.model_AVB as wae
from wae_metric.utils import special_normalize
from wae_metric.run_WAE import LATENT_SPACE_DIM, load_dataset

IMAGE_SIZE = 64
batch_size = 64


def run(**kwargs):
    fdir = kwargs.get('fdir','./outs')
    modeldir = kwargs.get('modeldir','./pretrained_model')
    ae_path = kwargs.get('ae_dir','./wae_metric/pretrained_model')
    datapath = kwargs.get('datapath','./data/')
    visdir = './tensorboard_plots'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if not os.path.exists(visdir):
        os.makedirs(visdir)


    jag_inp, jag_sca, jag_img = load_dataset(datapath)
    tr_id = np.random.choice(jag_sca.shape[0],int(jag_sca.shape[0]*0.95),replace=False)

    te_id = list(set(range(jag_sca.shape[0])) - set(tr_id))
    X_train = jag_inp[tr_id,:]
    y_sca_train = jag_sca[tr_id,:]
    y_img_train = jag_img[tr_id,:]

    np.random.shuffle(te_id)

    X_test = jag_inp[te_id,:]
    y_sca_test = jag_sca[te_id,:]
    y_img_test = jag_img[te_id,:]
    y_img_test_mb = y_img_test[-16:,:]

    y_img_test_mb = y_img_test_mb.reshape(16,64,64,4)

    for k in range(4):
        fig = plot(y_img_test_mb[:,:,:,k],immax=np.max(y_img_test_mb[:,:,:,k].reshape(-1,4096),axis=1),
                   immin=np.min(y_img_test_mb[:,:,:,k].reshape(-1,4096),axis=1))
        plt.savefig('{}/gt_img_{}_{}.png'
                    .format(fdir,str(k).zfill(3),str(k)), bbox_inches='tight')
        plt.close()



    print("Dataset dimensions: ",X_test.shape,y_sca_test.shape,y_img_test.shape)
    dim_x = X_train.shape[1]
    dim_y_sca = y_sca_train.shape[1]
    dim_y_img = y_img_train.shape[1]
    dim_y_img_latent = LATENT_SPACE_DIM #latent space

    ### Metric params

    ''' TEST mini-batch '''
    x_test_mb = X_test[-100:,:]
    y_sca_test_mb = y_sca_test[-100:,:]
    y_img_test_mb = y_img_test[-100:,:]

    y_sca = tf.placeholder(tf.float32, shape=[None, dim_y_sca])
    y_img = tf.placeholder(tf.float32, shape=[None, dim_y_img])
    x = tf.placeholder(tf.float32, shape=[None, dim_x])
    train_mode = tf.placeholder(tf.bool,name='train_mode')

    y_mm = tf.concat([y_img,y_sca],axis=1)

    '''**** Encode the img, scalars ground truth --> latent vector for loss computation ****'''
    y_latent_img = wae.gen_encoder_FCN(y_mm, dim_y_img_latent,train_mode=False)

    '''**** Train cycleGAN input params <--> latent space of (images, scalars) ****'''

    cycGAN_params = {'input_params':x,
                     'outputs':y_latent_img,
                     'param_dim':dim_x,
                     'output_dim':dim_y_img_latent,
                     'L_adv':1e-2,
                     'L_cyc':1e-1,
                     'L_rec':1}

    JagNet_MM = cycModel_MM(**cycGAN_params)
    JagNet_MM.run(train_mode)

    '''**** Decode the prediction from latent vector --> img, scalars ****'''
    y_img_out = wae.var_decoder_FCN(JagNet_MM.output_fake, dim_y_img+dim_y_sca,train_mode=False)
    img_loss = tf.reduce_mean(tf.square(y_img_out[:,:16384] - y_img))
    sca_loss = tf.reduce_mean(tf.square(y_img_out[:,16384:] - y_sca))

    fwd_img_summary = tf.summary.scalar(name='Image Loss', tensor=img_loss)
    fwd_sca_summary = tf.summary.scalar(name='Scalar Loss', tensor=sca_loss)
    merged = tf.summary.merge_all()

    t_vars = tf.global_variables()
    m_vars = [var for var in t_vars if 'wae' in var.name]
    metric_saver = tf.train.Saver(m_vars)
    saver = tf.train.Saver(list(set(t_vars)-set(m_vars)))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(modeldir)
    ckpt_metric = tf.train.get_checkpoint_state(ae_path)

    if ckpt_metric and ckpt_metric.model_checkpoint_path:
           metric_saver.restore(sess, ckpt_metric.model_checkpoint_path)
           print("************ Image Metric Restored! **************")

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("************ Model restored! **************")

    writer = tf.summary.FileWriter(visdir+'/{}'.format(modeldir), sess.graph)

    i = 0
    for it in range(100000):

        randid = np.random.choice(X_train.shape[0],batch_size,replace=False)
        x_mb = X_train[randid,:]
        y_img_mb = y_img_train[randid,:]
        y_sca_mb = y_sca_train[randid,:]

        fd = {x: x_mb, y_sca: y_sca_mb,y_img:y_img_mb,train_mode:True}

        for _ in range(10):
            _,dloss = sess.run([JagNet_MM.D_solver,JagNet_MM.loss_disc],feed_dict=fd)

        gloss0,gloss1,gadv = sess.run([JagNet_MM.loss_gen0,
                                           JagNet_MM.loss_gen1,
                                           JagNet_MM.loss_adv],
                                          feed_dict=fd)

        for _ in range(1):
            _ = sess.run([JagNet_MM.G0_solver],feed_dict=fd)

        if it % 100 == 0:
            print('Fidelity -- Iter: {}; Forward: {:.4f}; Inverse: {:.4f}'
                  .format(it, gloss0, gloss1))
            print('Adversarial -- Disc: {:.4f}; Gen: {:.4f}\n'.format(dloss,gadv))


        if it % 500 == 0:

            nTest=16
            x_test_mb = X_test[-nTest:,:]
            summary_val = sess.run(merged,feed_dict={x:X_test,y_sca:y_sca_test,y_img:y_img_test,train_mode:False})

            writer.add_summary(summary_val, it)

            samples,samples_x = sess.run([y_img_out,JagNet_MM.input_cyc],
                                           feed_dict={x: x_test_mb,train_mode:False})
            data_dict= {}
            data_dict['samples'] = samples
            data_dict['samples_x'] = samples_x
            data_dict['y_sca'] = y_sca_test
            data_dict['y_img'] = y_img_test
            data_dict['x'] = x_test_mb

            test_imgs_plot(fdir,it,data_dict)

            save_path = saver.save(sess, "./"+modeldir+"/model_"+str(it)+".ckpt")

if __name__=='__main__':
    run()

### srun -n 1 -N 1 -p pbatch -A lbpm --time=3:00:00 --pty /bin/sh
