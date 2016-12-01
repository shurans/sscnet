~/build_master_release/tools/caffe train -weights=../../../../models/suncg.caffemodel -solver=solver.txt -gpu 1 2>&1 | tee log.txt
