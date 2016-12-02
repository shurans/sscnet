# Semantic Scene Completion from a Single Depth Image
Shuran Song, Fisher Yu,  Andy Zeng,  Angel X. Chang,  Manolis Savva,  Thomas Funkhouser  


### Contents
0. [Organization](#organization)
0. [Installation](#installation)
0. [Requirements: hardware](#requirements-hardware)
0. [Testing](#testing)
0. [Training](#training)
0. [Visualization and Evaluation](#visualization-and-evaluation)
0. [Data Preparation](#data)



### Organization
The code organized as follow 
``` shell
    sscnet_release
         |-- matlab_code
         |-- caffe_code
                    |-- caffe3d_suncg
                    |-- script
                         |-train
                         |-test   
         |-- data
                |-- depthbin
                    |-- NYUtrain 
                        |-- xxxxx_0000.png
                        |-- xxxxx_0000.bin
                    |-- NYUtest
                    |-- NYUCADtrain
                    |-- NYUCADtest
                    |-- SUNCGtest
                    |-- SUNCGtrain01
                    |-- SUNCGtrain02
                    |-- ...
                |-- eval
                    |-- NYUtest
                    |-- NYUCADtest
                    |-- SUNCGtest
            |-- models
            |-- results
```
### Download 
Download 




### Installation
0. Requirements: matlab, opencv, cudnn
0. Hardware Requirements:  At least 12G GPU memory.
0. Install caffe and pycaffe. 
    1. Modify the config file base on your system 
    2. Compile  
    `cd sscnet_release/caffe_code/caffe3d_suncg/
    make
    make pycaffe`

0. Export path
    ``` shell 
    export LD_LIBRARY_PATH=~/build_master_release/lib:/usr/local/cudnn/v5/lib64:~/anaconda2/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=~/build_master_release/python:$PYTHONPATH
    ```


### Testing:
0. run the testing script
    cd sscnet_release/caffe_code/script/test
    python test_model.py
0. the output result will be stored in sscnet_release/results in .hdf5 format
0. To test on other testset, e.g. suncg or nyu, nyucad you need to modify paths in “test_model.py”.
    


### Training:
0. Fine Turning on NYU 
    `cd sscnet_release/caffe_code/train/ftnyu
      ./train.sh`
0. Training from scratch 
    ` cd sscnet_release/caffe_code/train/trainsuncg
    ./train.sh`
0. To get more training data from SUNCG, please reference the SUNCG toolbox 
    


### Visualization and Evaluation:
0. After testing the result should stored in sscnet_release/results
0. You can also download the precomputed result:
    `  cd sscnet_release/
    ./download_results.sh`
0. Run the evaluation code 
    `cd sscnet_release/matlab_code
    evluation_script('../results/','nyucad')`
0. The visualization “ply” files will be stored in sscnet_release/results/nyucad






### Data 
0. Date format 
    1. Depth map : 
        16 bit png with bit shifting.
        Please reference: ./matlab_code/readDepth.m about the depth format.
    2. 3D volume: 
        First three float stores the origin of the 3D volume in world coordinate.
        Followed by 16 float of camera pose in world coordinate.
        Followed the 3D volume encoded by run-length encoding.
        Please reference: ./matlab_code/utils/readRLEfile.m for more detail.
0. Example code to convert NYU ground truth data: 
    ./matlab_code/perpareNYUCADdata.m
    This function provide a example of how to convert the NYU ground truth from 3D CAD model annotation provided by:
    Guo, Ruiqi, Chuhang Zou, and Derek Hoiem. 
    "Predicting complete 3d models of indoor scenes."
    You need to download the original annotation by `./downlad_UIUCCAD.sh`  




### License
Code is released under the MIT License (refer to the LICENSE file for details).

    
