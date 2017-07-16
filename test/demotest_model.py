#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Quick demo for semantic completion and vsilzation
## Please referece the full pipeline for training and evaluation 


from __future__ import print_function, division
import caffe
import h5py
import numpy as np
from os.path import join


def vol2points(vol,tsdf,seg_label):
    classlabel = np.argmax(vol, axis=1) 
    colorMap = np.array([[ 22,191,206],[214, 38, 40],[ 43,160, 43],[158,216,229],[114,158,206],[204,204, 91],[255,186,119],[147,102,188],[ 30,119,181],[188,188, 33],[255,127, 12],[196,175,214],[153,153,153]])
    points = []
    rgb    = []
    for x in range(classlabel.shape[1]):
        for y in range(classlabel.shape[2]):
            for z in range(classlabel.shape[3]):
                tsdfvalue = tsdf[0][0][4*x][4*y][4*z];
                if (classlabel[0][x][y][z] > 0 and seg_label[0][0][x][y][z] <= 254 and ( tsdfvalue < 0 or tsdfvalue > 0.8)):
                    points.append(np.array([x,y,z]))
                    rgb.append(np.array(colorMap[classlabel[0][x][y][z],:]))
    points = np.vstack(points)
    rgb = np.vstack(rgb)
    return {'points':points, 'rgb':rgb}


def writeply(filename, points,rgb):
    target = open(filename, 'w')
    # write the  header 
    target.write('ply\n');
    target.write('format ascii 1.0 \n')
    target.write('element vertex ' + str(points.shape[0]) + '\n')
    target.write('property float x\n')
    target.write('property float y\n')
    target.write('property float z\n')
    target.write('property uchar red\n')
    target.write('property uchar green\n')
    target.write('property uchar blue\n')
    target.write('end_header\n')
    # write the  points 
    for i in range(points.shape[0]):
        target.write('%f %f %f %d %d %d\n'%(points[i,0],points[i,1],points[i,2], rgb[i,0],rgb[i,1],rgb[i,2]))
    target.close()

def test_model():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    model_path =  'demo.txt'
    pretrained_path = '../models/suncg_ft_nyu.caffemodel'
    output_file = 'demo'
    numof_files = 1

    net = caffe.Net(model_path, pretrained_path, caffe.TEST)

    predictions = []
    
    for i in range(numof_files):
        out = net.forward() 
        # write volumne
        fp = h5py.File(output_file + '.hdf5', "w")
        predictions = np.array(net.blobs['prob'].data)
        tsdf = np.array(net.blobs['data'].data)
        seg_label = np.array(net.blobs['seg_label'].data)
        result = fp.create_dataset("result", predictions.shape, dtype='f')
        result[...] = predictions

        label = fp.create_dataset("label", seg_label.shape, dtype='f')
        label[...] = seg_label
        fp.close()
        # write point cloud
        pd = vol2points(predictions,tsdf,seg_label)
        writeply(output_file+'.ply', pd['points'],pd['rgb'])

    print('output: '+ output_file)
    
    

if __name__ == '__main__':
    test_model()
