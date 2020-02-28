import os
import shutil
import pickle as pkl
import argparse

import tensorflow as tf
import numpy as np
np.random.seed(4321)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler

from .model_AVB import *

LATENT_SPACE_DIM = 20


def sample_z(L,dim,type='uniform'):
    if type == 'hypercircle':
        theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,dim))]
    elif type == 'uniform':
        theta = 5*(1-2*np.random.rand(L, dim))
    elif type == 'normal':
        theta = np.random.randn(L,dim)

    return np.asarray(theta)

def load_dataset(datapath):
    jag_img = np.load(datapath+'jag10K_images.npy')
    jag_sca = np.load(datapath+'jag10K_0_scalars.npy')
    jag_inp = np.load(datapath+'jag10K_params.npy')

    return jag_inp,jag_sca,jag_img

def run(**kwargs):

    fdir = kwargs.get('fdir','./expt_out')
    modeldir = kwargs.get('modeldir','./expt_model')
    datapath = kwargs.get('datapath','../../data/')
    vizdir = kwargs.get('vizdir','graphs')
    lam = kwargs.get('lam',1e-4)
    dim_z = kwargs.get('dimz',LATENT_SPACE_DIM)


    # if not os.path.exists(fdir):
    #     os.makedirs(fdir)

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    jag_inp, jag_sca, jag_img = load_dataset(datapath)
    jag = np.hstack((jag_img,jag_sca))


    tr_id = np.random.choice(jag.shape[0],int(jag.shape[0]*0.95),replace=False)
    te_id = list(set(range(jag.shape[0])) - set(tr_id))

    X_train = jag[tr_id,:]
    y_train = jag[tr_id,:]


    X_test_set = jag[te_id,:]
    y_test_set = jag[te_id,:]

    batch_size = 100

    dim_image = jag.shape[1]


    # Image  to Image
    y = tf.placeholder(tf.float32, shape=[None, dim_image])
    z = tf.placeholder(tf.float32, shape=[None, dim_z])
    train_mode = tf.placeholder(tf.bool,name='train_mode')


    z_sample = gen_encoder_FCN(y, dim_z,train_mode)

    y_recon = var_decoder_FCN(z_sample, dim_image,train_mode)

    D_sample = discriminator_FCN(y_recon, z_sample)
    D_prior = discriminator_FCN(y, z, True)



    img_loss = tf.reduce_mean(tf.square(y_recon-y))
    rec_error = tf.reduce_mean(tf.nn.l2_loss(y_recon-y))
    D_loss,gen_error = compute_adv_loss(D_prior,D_sample)
    G_loss = rec_error+lam*gen_error

    t_vars = tf.global_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = list(set(t_vars)-set(d_vars))

    D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(D_loss,var_list=d_vars)
    G_solver = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(G_loss,var_list=g_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    rec_summary = tf.summary.scalar(name='Recon_MSE', tensor=img_loss)
    disc_summary = tf.summary.scalar(name='Disc_Loss',tensor=D_loss)
    gen_summary = tf.summary.scalar(name='Gen_Loss',tensor=gen_error)
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./train_graphs/{}'.format(vizdir), sess.graph)
    writer_test = tf.summary.FileWriter('./test_graphs/{}'.format(vizdir), sess.graph)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(modeldir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("************ Model restored! **************")


    for it in range(0,100000):
        randid = np.random.choice(X_train.shape[0],batch_size,replace=False)
        y_mb = X_train[randid,:]
        X_mb = X_train[randid,:]


        z_mb = sample_z(batch_size,dim_z)

        fd = {y:y_mb, train_mode: True,z:z_mb}
        for i in range(1):
            _, G_loss_curr,tmp1,tmp2 = sess.run([G_solver, G_loss,rec_error,gen_error],
                                    feed_dict=fd)
        for i in range(1):
            _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict=fd)

        if (it+1) % 100 ==0:
            summary = sess.run(merged,feed_dict=fd)
            writer.add_summary(summary, it)

        if (it+1) % 1000 == 0:
            print('Iter: {}; Recon_Error = : {:.4}, G_Loss: {:.4}; D_Loss: {:.4}'
                  .format(it+1, tmp1, tmp2, D_loss_curr))

            z_test = sample_z(len(te_id),dim_z)
            fd = {train_mode:False, y:X_test_set, z:z_test}
            samples,summary_val = sess.run([y_recon,merged],feed_dict=fd)

            writer_test.add_summary(summary_val, it)

        if (it+1)%1000==0:
            save_path = saver.save(sess, "./"+modeldir+"/model_"+str(it)+".ckpt")
    return


if __name__=='__main__':
    run(datapath='../data/',vizdir='viz',modeldir='pretrained_model',lam=1e-2,dimz=LATENT_SPACE_DIM)
