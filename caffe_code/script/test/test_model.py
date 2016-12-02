#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import caffe
import h5py
import numpy as np
from os.path import join


def test_model():
    caffe.set_mode_gpu()
    caffe.set_device(0)

    root_path = '../../../'
    ###########  testing on SUNCG use uncommoent the following three lines  ############### 
    model_path =  'deploy_net_suncg.txt'
    pretrained_path = root_path + '/models/suncg.caffemodel'
    output_file = root_path + '/results/result_suncg.hdf5'
    numof_files = 470


    # ##########  testing on NYUCAD uncommoent the following three lines ############### 
    # model_path = 'deploy_net_nyucad.txt'
    # pretrained_path = root_path + '/models/suncg_ft_nyucad.caffemodel'
    # output_file = root_path + '/results/result_nyucad.hdf5';
    # numof_files = 654    

    # #############  testing on NYU  uncommoent the following three lines ############### 
    # model_path  =  'deploy_net_nyu.txt'
    # pretrained_path = root_path + '/models/suncg_ft_nyu.caffemodel'
    # output_file = root_path + '/results/result_nyu.hdf5'
    # numof_files = 654
    
    net = caffe.Net(model_path, pretrained_path, caffe.TEST)

    predictions = []
    for i in range(numof_files):
        print("testing:"+ pretrained_path + "  "+ str(i))
        out = net.forward() 
        predictions.append(np.array(net.blobs['prob'].data))
    predictions = np.vstack(predictions)

    print('Writing:'+ output_file)
    fp = h5py.File(output_file, "w")
    result = fp.create_dataset("result", predictions.shape, dtype='f')
    result[...] = predictions
    fp.close()

   
    

if __name__ == '__main__':
    test_model()
