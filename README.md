# Semantic Scene Completion from a Single Depth Image
Shuran Song, Fisher Yu, Andy Zeng, Angel X. Chang, Manolis Savva, Thomas Funkhouser  

### Contents
0. [Orgnization](#Oognization)
0. [Installation](#installation)
0. [Requirements: hardware](#requirements-hardware)
0. [Testing](#testing)
0. [Training](#training)
0. [Visulization and Evaluation](#visulization-and-evaluation)
0. [Data Preparation](#data-preparation)
0. [Resources](#resources)

### Orgnization
The code orgnized as follow 
``` shell
sscnet_release
    |-- data
        |-- depthbin
            |-- NYUtrain 
            |-- NYUtest
            |-- NYUCADtrain
            |-- NYUCADtest
            |-- SUNCGtest
            |-- SUNCGtrain01
            |-- ...
        |-- eval
            |-- NYUtest
                |- xxx_gt_d4.mat
                |- xxx_vol_d4.mat
            |-- NYUCADtest
            |-- SUNCGtest
    |-- models
    |-- results
    |-- matlab_code
    |-- caffe_code
            |-- caffe3d_suncg
            |-- script
                 |-train
                 |-test
```


### Installation
Requirements: matlab, opencv
Hardware Requirements:  At least 12G GPU memory.
You need to install caffe and pycaffe. 

### Testing:
0. There are three different testset:
suncg:
nyucad:
nyu: 

0. download test data
    cd sscnet_release/data/
    ./download_nyutest.sh

0. run the testing script
    cd sscnet_release/caffe_code/script/test
    python test_model.py

0. the output result will be stored in sscnet_release/results in hdf5 format
0. To test on other testset, e.g. suncg or nyucad:
   1. download the corespounding test data
   2. modify test_model.py pathes.
    

### Training:
0. download the training data
   cd sscnet_release/data/
   ./download_nyutest.sh

### Visulization and Evaluation:
0. After testing the result should stored in sscnet_release/results


### Data 
0. Date format 
depth map : 16 bit png. Please reference: ./matlab_code/read_xxx.m about the depth format.
volume: Please reference: ./matlab_code/read_xxx.m, ./matlab_code/write_xxx.m  for details about the volume format.

0. Example code to convert NYU ground truth data: 
TODO


### License

Code is released under the MIT License (refer to the LICENSE file for details).