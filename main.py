# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# @authors: :Rushil Anirudh, Jayaraman J. Thiagarajan, Timo Bremer
#
# SPDX-License-Identifier: MIT

import numpy as np
import argparse
import run_cycgan_mm as cycGAN
import os
from wae_metric import run_WAE as metric

parser = argparse.ArgumentParser()
parser.add_argument('-o', type=str, default='out',
                    help='Saving Results Directory')
parser.add_argument('-m', type=str, default='weights',
                    help='location to store weights')
parser.add_argument('-train_ae', type=str, default='True',
                    help=' "-train_ae 0" to use pre-trained auto-encoder. "-train_ae 1": will train a new autoencoder before running the surrogate training.')
parser.add_argument('-ae_dir', type=str, default='wae_metric/pretrained',
                    help='Ignored if train_ae=True; else will load existing autoencoder')
parser.add_argument('-d', type=str, default='./data/',
                    help='path to dataset - images, scalars, and input params')

args = parser.parse_args()
fdir = args.o
mdir = args.m
train_ae = args.train_ae
ae_dir = args.ae_dir
datapath = args.d

if train_ae in ['True', '1', 't', 'y', 'yes']:
    train_ae_b = True
    ae_dir = 'wae_metric/model_'+mdir
    ae_dir_outs = 'wae_metric/outs'
else:
    train_ae_b = False

if train_ae_b:
    print('****** Training the autoencoder *******')
    metric.run(fdir=ae_dir_outs,modeldir=ae_dir,datapath=datapath)
    print('****** Training the macc surrogate *******')
    # cycGAN.run(fdir,mdir,ae_dir)
    cycGAN.run(fdir=fdir,modeldir=mdir,ae_dir=ae_dir,datapath=datapath)
else:
    print('****** Training the macc surrogate with pre-trained autoencoder *******')
    cycGAN.run(fdir=fdir,modeldir=mdir,ae_dir=ae_dir,datapath=datapath)
