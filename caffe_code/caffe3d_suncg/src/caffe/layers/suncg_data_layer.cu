#include <vector>
#include <utility>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/suncg_data_layer.hpp"
#include "caffe/util/rng.hpp"

#include "suncg_util.hpp"
#include "suncg_fusion.hpp"
// #include "suncg_fusion.cu"

DEFINE_bool(shuran_chatter, false,
            "If you are Shuran and want chatter, turn this on.");

using std::vector;

namespace caffe {

template<typename Dtype>
SuncgDataLayer<Dtype>::~SuncgDataLayer<Dtype>() {
  this->StopInternalThread();
}


template<typename Dtype>
void SuncgDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  LOG(INFO) << "Read SUNCG parameters";
  const SuncgDataParameter& data_param =
      this->layer_param_.suncg_data_param();

  for (int i = 0; i < data_param.file_data_size(); ++i) {
    file_data.push_back(data_param.file_data(i));
  }
  file_list = data_param.file_list();
  vox_unit = data_param.vox_unit();
  vox_margin = data_param.vox_margin();
  add_height = data_param.with_height();
  
  for (int i = 0; i < data_param.seg_class_map_size(); ++i) {
    segmentation_class_map.push_back(data_param.seg_class_map(i));
  }
  
  num_segmentation_class = *std::max_element(segmentation_class_map.begin(), segmentation_class_map.end())+1;
  LOG(INFO) << "num_segmentation_class"<< num_segmentation_class;

  for (int i = 0; i < data_param.seg_class_weight_size(); ++i) {
    segmentation_class_weight.push_back(data_param.seg_class_weight(i));
  }
  for (int i = 0; i < data_param.occ_class_weight_size(); ++i) {
    occupancy_class_weight.push_back(data_param.occ_class_weight(i));
  }
  shuffle_data = data_param.shuffle();
  occ_emptyonly = data_param.occ_empty_only();
  data_num_channel = add_height ? 2 : 1;
  surf_only = data_param.surf_only();

  CHECK_EQ(data_param.vox_size_size(), 3);
  CHECK_EQ(data_param.crop_size_size(), 3);

  for (int i = 0; i < data_param.vox_size_size(); ++i) {
    data_full_vox_size.push_back(data_param.vox_size(i));
  }

  for (int i = 0; i < data_param.crop_size_size(); ++i) {
    data_crop_vox_size.push_back(data_param.crop_size(i));;
  }

  for (int i = 0; i < data_param.label_size_size(); ++i) {
    label_vox_size.push_back(data_param.label_size(i));
  }

  sample_neg_obj_ratio = data_param.neg_obj_sample_ratio();

  batch_size = data_param.batch_size();

  offset_value = 0;

  epoch_prefetch = 0;
  counter = 0;
  
  // List all files in data folder and shuffle them if necessary
  GetFiles(file_data, data_filenames, "camera_list_train.list", "0000.png");
  if (shuffle_data) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
    Shuffle();
  }

  LOG(INFO) << "Read camera information";
  // Copy camera information to GPU
  cam_info[0] = Dtype(frame_width);
  cam_info[1] = Dtype(frame_height);
  for (int i = 0; i < 9; ++i)
    cam_info[i + 2] = cam_K[i];
  for (int i = 0; i < 16; ++i)
    cam_info[i + 11] = 0.0f;
  CUDA_CHECK(cudaMalloc(&cam_info_GPU, 27 * sizeof(Dtype)));

  LOG(INFO) << "Set voxel volume parameters and copy them to GPU";
  vox_info[0] = vox_unit;
  vox_info[1] = vox_margin;
  for (int i = 0; i < 3; ++i)
    vox_info[i + 2] = Dtype(data_crop_vox_size[i]);
  CUDA_CHECK(cudaMalloc(&vox_info_GPU, 8 * sizeof(Dtype)));

  LOG(INFO) << "Allocating data";
  LOG(INFO) << "data_num_channel: "<< data_num_channel;
  // GPU malloc depth data
  CUDA_CHECK(cudaMalloc(&depth_data_GPU,
                        frame_height * frame_width * sizeof(Dtype)));

  // GPU malloc voxel volume weights
  CUDA_CHECK(cudaMalloc(&vox_weight_GPU,
                        data_crop_vox_size[0] * data_crop_vox_size[1] *
                        data_crop_vox_size[2] * sizeof(Dtype)));


  size_t memoryBytes = 0;
//  std::cout << (train_me ? "* " : "  ");
//  std::cout << name << std::endl;

  // Determine if data should be cropped
  is_cropping_data = data_crop_vox_size[0] < data_full_vox_size[0] ||
                     data_crop_vox_size[1] < data_full_vox_size[1] ||
                     data_crop_vox_size[2] < data_full_vox_size[2];
  int num_crop_voxels =
      data_crop_vox_size[0] * data_crop_vox_size[1] * data_crop_vox_size[2];
  int num_full_voxels =
      data_full_vox_size[0] * data_full_vox_size[1] * data_full_vox_size[2];

  if (is_cropping_data) {
    CUDA_CHECK(cudaMalloc(&vox_data_GPU,
                          batch_size * data_num_channel * num_crop_voxels *
                          sizeof(Dtype)));
    memoryBytes +=
        batch_size * data_num_channel * num_crop_voxels * sizeof(Dtype);
  } else {
    CUDA_CHECK(cudaMalloc(&vox_data_GPU,
                          batch_size * data_num_channel * num_full_voxels *
                          sizeof(Dtype)));
    memoryBytes +=
        batch_size * data_num_channel * num_full_voxels * sizeof(Dtype);
  }

  int num_label_voxels =
      label_vox_size[0] * label_vox_size[1] * label_vox_size[2];
  CUDA_CHECK(cudaMalloc(&occupancy_label_GPU,
                        batch_size * num_label_voxels * sizeof(Dtype)));
  CUDA_CHECK(cudaMalloc(&occupancy_weight_GPU,
                        batch_size * 2 * num_label_voxels * sizeof(Dtype)));
  memoryBytes += batch_size * 3 * num_label_voxels * sizeof(Dtype);

  CUDA_CHECK(cudaMalloc(&segmentation_label_GPU,
                        batch_size * num_label_voxels * sizeof(Dtype)));
  CUDA_CHECK(cudaMalloc(&segmentation_weight_GPU,
                        batch_size * num_segmentation_class * num_label_voxels *
                        sizeof(Dtype)));
  CUDA_CHECK(cudaMalloc(&segmentation_surf_weight_GPU,
                        batch_size * num_segmentation_class * num_label_voxels *
                        sizeof(Dtype)));
  memoryBytes += batch_size * (num_segmentation_class + 1) * num_label_voxels *
                 sizeof(Dtype);

  LOG(INFO) << "Resize tops";
  // out[0]->need_diff = false;
  std::vector<int> data_dim;
  data_dim.resize(5);
  data_dim[0] = batch_size;
  data_dim[1] = data_num_channel;
  if (is_cropping_data) {
    data_dim[2] = data_crop_vox_size[0];
    data_dim[3] = data_crop_vox_size[1];
    data_dim[4] = data_crop_vox_size[2];
  } else {
    data_dim[2] = data_full_vox_size[0];
    data_dim[3] = data_full_vox_size[1];
    data_dim[4] = data_full_vox_size[2];
  }

  top[0]->Reshape(data_dim);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    // this->prefetch_[i].Resize(5);
    this->prefetch_[i].mutable_blob(0)->Reshape(data_dim);
  }

  data_dim[1] = 1;
  data_dim[2] = label_vox_size[0];
  data_dim[3] = label_vox_size[1];
  data_dim[4] = label_vox_size[2];
  for (int i = 1; i < top.size(); ++i) {
    top[i]->Reshape(data_dim);
    for (int j = 0; j < this->PREFETCH_COUNT; ++j) {
      this->prefetch_[j].mutable_blob(i)->Reshape(data_dim);
    }
  }

//  if (top.size() > 5) {
//    data_dim[2] = 480;
//    data_dim[3] = 640;
//    data_dim[4] = 1;
//    top[5]->Reshape(data_dim);
//    for (int j = 0; j < this->PREFETCH_COUNT; ++j) {
//      this->prefetch_[j].mutable_blob(5)->Reshape(data_dim);
//    }
//  }

//
//  out[0]->receptive_field.resize(data_dim.size() - 2);  fill_n(out[0]->receptive_field.begin(),  data_dim.size() - 2, 1);
//  out[0]->receptive_gap.resize(data_dim.size() - 2);    fill_n(out[0]->receptive_gap.begin(),    data_dim.size() - 2, 1);
//  out[0]->receptive_offset.resize(data_dim.size() - 2); fill_n(out[0]->receptive_offset.begin(), data_dim.size() - 2, 0);
//  memoryBytes += out[0]->Malloc(data_dim);
//
//  // Occupancy label
//  out[1]->need_diff = false;
//  data_dim[1] = 1;
//  data_dim[2] = label_vox_size[0];
//  data_dim[3] = label_vox_size[1];
//  data_dim[4] = label_vox_size[2];
//  memoryBytes += out[1]->Malloc(data_dim);
//
//  // Occupancy weight
//  out[2]->need_diff = false;
//  data_dim[1] = 2;
//  memoryBytes += out[2]->Malloc(data_dim);
//
//  // Segmentation label
//  out[3]->need_diff = false;
//  data_dim[1] = 1;
//  memoryBytes += out[3]->Malloc(data_dim);
//
//  // Segmentation weight
//  out[4]->need_diff = false;
//  data_dim[1] = num_segmentation_class;
//  memoryBytes += out[4]->Malloc(data_dim);
//
//  // Segmentation surface weight
//  out[5]->need_diff = false;
//  memoryBytes += out[5]->Malloc(data_dim);
//
//  // prefetch();
//  lock = std::async(std::launch::async, &SUNCGDataLayer::prefetch, this);

  // return memoryBytes;
}


template<typename Dtype>
void SuncgDataLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
  // LOG(INFO) << "Loading " << batch;
  const SuncgDataParameter& data_param =
      this->layer_param_.suncg_data_param();
  Blob<Dtype> *tsdf = nullptr, *occ_label = nullptr, *occ_weight = nullptr;
  if (batch->size() > 3) {
    occ_label = batch->mutable_blob(3);
    occ_weight = batch->mutable_blob(4);
  }
  // if (data_param.data_type() == SuncgDataParameter_DATA_TSDF) {
  //   tsdf = batch->mutable_blob(0);
  // } else if(data_param.data_type() == SuncgDataParameter_DATA_OCCUPANCY) {
  //   occ_label = batch->mutable_blob(0);
  // }
  tsdf = batch->mutable_blob(0);
  Blob<Dtype> *seg_label = batch->mutable_blob(1);
  Blob<Dtype> *seg_weight = batch->mutable_blob(2);

  for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    // Get random image
    bool is_valid_image = false;
    std::string depth_path = "";
    while (!is_valid_image) {
      depth_path = data_filenames[counter];
      is_valid_image =
          FileExists(depth_path.substr(0, depth_path.length() - 3) + "bin") &&
          FileExists(depth_path);
      // increase counter
      counter++;
      if (counter >= data_filenames.size()) {
        counter = 0;
        ++epoch_prefetch;
        if (shuffle_data) Shuffle();
      }
    }

    // Get depth image and copy to GPU
    Dtype *depth_data = new Dtype[frame_height * frame_width];
    ReadDepthImage(depth_path, depth_data, frame_width, frame_height);
    // LOG(INFO) << "Depth data: " << depth_data[frame_width * frame_height / 2];
    CUDA_CHECK(cudaMemcpy(depth_data_GPU, depth_data,
                                   frame_height * frame_width * sizeof(Dtype),
                                   cudaMemcpyHostToDevice));

    // Get voxel origin (in world coordinates), camera pose, and voxel labels
    Dtype vox_origin[3];
    Dtype cam_pose[16];
    int num_crop_voxels =
        data_crop_vox_size[0] * data_crop_vox_size[1] * data_crop_vox_size[2];
    int num_full_voxels =
        data_full_vox_size[0] * data_full_vox_size[1] * data_full_vox_size[2];
    Dtype *occupancy_label_full = new Dtype[num_full_voxels];
    Dtype *segmentation_label_full = new Dtype[num_full_voxels];
    ReadVoxLabel(depth_path.substr(0, depth_path.length() - 3) + "bin",
                 vox_origin, cam_pose, occupancy_label_full,
                 segmentation_class_map, segmentation_label_full);

    // Find cropping origin
    int crop_origin[3] = {0, 0, 0};
    if (is_cropping_data) {
      bool crop_vox_found = false;
      int max_iter = 100;
      int sample_iter = 0;

      // Crop a random box out of the full volume
      while (!crop_vox_found && sample_iter < max_iter) {

        // Compute random cropping origin
        crop_origin[0] = 0.0f;
        crop_origin[1] = 0.0f;
        crop_origin[2] = 0.0f;
        if (data_full_vox_size[0] - data_crop_vox_size[0] > 0)
          crop_origin[0] = (int) std::floor(GenRandFloat(0.0f, (float) (
              data_full_vox_size[0] - data_crop_vox_size[0])));
        if (data_full_vox_size[1] - data_crop_vox_size[1] > 0)
          crop_origin[1] = (int) std::floor(GenRandFloat(0.0f, (float) (
              data_full_vox_size[1] - data_crop_vox_size[1])));
        if (data_full_vox_size[2] - data_crop_vox_size[2] > 0)
          crop_origin[2] = (int) std::floor(GenRandFloat(0.0f, (float) (
              data_full_vox_size[2] - data_crop_vox_size[2])));
        sample_iter++;

        // Check cropped box is non-empty and contains object classes other than only floor, wall, ceiling
        int num_non_empty_voxels = 0;
        int num_object_voxels = 0;
        for (int x = crop_origin[0];
             x < crop_origin[0] + data_crop_vox_size[0]; ++x)
          for (int y = crop_origin[1];
               y < crop_origin[1] + data_crop_vox_size[1]; ++y)
            for (int z = crop_origin[2];
                 z < crop_origin[2] + data_crop_vox_size[2]; ++z) {
              int full_voxel_idx =
                  z * data_full_vox_size[0] * data_full_vox_size[1] +
                  y * data_full_vox_size[0] + x;
              if (segmentation_label_full[full_voxel_idx] > 0 & segmentation_label_full[full_voxel_idx] < 255)
                num_non_empty_voxels++;
              if (segmentation_label_full[full_voxel_idx] > 3)
                num_object_voxels++;
            }
        if (num_non_empty_voxels <
            data_crop_vox_size[0] * data_crop_vox_size[0] ||
            num_object_voxels < data_crop_vox_size[0])
          continue;
        crop_vox_found = true;
      }
    }

    if (FLAGS_shuran_chatter) {
      LOG(INFO) << depth_path << " " << crop_origin[0] << " " << crop_origin[1]
                << " " << crop_origin[2];
    }

    // Update voxel parameters with new voxel origin (+ cropping origin) in world coordinates
    vox_info[5] = vox_origin[0] + (float) (crop_origin[2]) * vox_unit;
    vox_info[6] = vox_origin[1] + (float) (crop_origin[0]) * vox_unit;
    vox_info[7] = vox_origin[2] + (float) (crop_origin[1]) * vox_unit;

    // Update camera information with new camera pose
    for (int i = 0; i < 16; ++i)
      cam_info[i + 11] = cam_pose[i];

    // Update camera information and voxel parameters in GPU
    CUDA_CHECK(cudaMemcpy(cam_info_GPU, cam_info, 27 * sizeof(Dtype),
                                   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vox_info_GPU, vox_info, 8 * sizeof(Dtype),
                                   cudaMemcpyHostToDevice));

    // Get voxel volume
    Dtype *tmp_tsdf_data_GPU =
        vox_data_GPU + batch_idx  * num_crop_voxels;
    GPU_set_value(num_crop_voxels, tmp_tsdf_data_GPU, Dtype(1.0));
    Dtype *tmp_vox_height_GPU = NULL;
    if (add_height) {
      tmp_vox_height_GPU = vox_data_GPU + 
          batch_idx * data_num_channel * num_crop_voxels +
          num_crop_voxels;
      GPU_set_zeros(num_crop_voxels, tmp_vox_height_GPU);
    }

    // Reset voxel weights in GPU
    Dtype *vox_weight_CPU = new Dtype[num_crop_voxels];
    memset(vox_weight_CPU, 0, num_crop_voxels * sizeof(Dtype));
    CUDA_CHECK(cudaMemcpy(vox_weight_GPU, vox_weight_CPU,
                                   num_crop_voxels * sizeof(Dtype),
                                   cudaMemcpyHostToDevice));

    // Retreive cropped labels
    Dtype *occupancy_label_crop = new Dtype[num_crop_voxels];
    Dtype *segmentation_label_crop = new Dtype[num_crop_voxels];
    for (int x = 0; x < data_crop_vox_size[0]; ++x)
      for (int y = 0; y < data_crop_vox_size[1]; ++y)
        for (int z = 0; z < data_crop_vox_size[2]; ++z) {
          int crop_voxel_idx =
              z * data_crop_vox_size[0] * data_crop_vox_size[1] +
              y * data_crop_vox_size[0] + x;
          int full_voxel_idx = (z + crop_origin[2]) * data_full_vox_size[0] *
                               data_full_vox_size[1] +
                               (y + crop_origin[1]) * data_full_vox_size[0] +
                               (x + crop_origin[0]);
          occupancy_label_crop[crop_voxel_idx] = occupancy_label_full[full_voxel_idx];
          segmentation_label_crop[crop_voxel_idx] = segmentation_label_full[full_voxel_idx];
        }

    // Fuse frame into voxel volume
    if (data_param.data_type() == SuncgDataParameter_DATA_OCCUPANCY ){
        Dtype *  occupancy_label_crop_GPU; 
        cudaMalloc(&occupancy_label_crop_GPU, num_crop_voxels* sizeof(Dtype));
        cudaMemcpy(occupancy_label_crop_GPU, occupancy_label_crop, num_crop_voxels * sizeof(Dtype), cudaMemcpyHostToDevice);
        int THREADS_NUM = 1024;
        int BLOCK_NUM = int((num_crop_voxels + size_t(THREADS_NUM) - 1) / THREADS_NUM);
        CompleteTSDF<<< BLOCK_NUM, THREADS_NUM >>>(vox_info_GPU, occupancy_label_crop_GPU, tmp_tsdf_data_GPU);
        
        cudaFree(occupancy_label_crop_GPU);
        CUDA_CHECK(cudaGetLastError());

    }else{
      if(data_param.with_projection_tsdf()){
        int num_blocks = data_crop_vox_size[2];
        int num_threads = data_crop_vox_size[1];
        GPU_set_zeros(num_crop_voxels, vox_weight_GPU);
        Integrate <<< data_crop_vox_size[2], data_crop_vox_size[1] >>> (cam_info_GPU, vox_info_GPU, depth_data_GPU, tmp_tsdf_data_GPU, vox_weight_GPU, tmp_vox_height_GPU);
        CUDA_CHECK(cudaGetLastError());
      }else{
        ComputeTSDF(cam_info, vox_info, cam_info_GPU, vox_info_GPU, depth_data_GPU, tmp_tsdf_data_GPU, tmp_vox_height_GPU);
        CUDA_CHECK(cudaGetLastError());
      }      
    }


    // Copy voxel volume back to CPU
    Dtype *vox_tsdf = new Dtype[num_crop_voxels];
    CUDA_CHECK(cudaMemcpy(vox_tsdf, tmp_tsdf_data_GPU,
                                   num_crop_voxels * sizeof(Dtype),
                                   cudaMemcpyDeviceToHost));

    
    if (tsdf != nullptr) {
      memcpy(tsdf->mutable_cpu_data() + num_crop_voxels * batch_idx * data_num_channel,
             vox_tsdf, num_crop_voxels * sizeof(Dtype));
    }

    //adding height to floor
    if (add_height){
      cudaMemcpy(tsdf->mutable_cpu_data() + num_crop_voxels * batch_idx * data_num_channel + num_crop_voxels,
                 tmp_vox_height_GPU, num_crop_voxels * sizeof(Dtype), cudaMemcpyDeviceToHost);
    }

    
    

    // Downsample label with scale
    int num_label_voxels =
        label_vox_size[0] * label_vox_size[1] * label_vox_size[2];
    
    int label_downscale = (data_crop_vox_size[0] / label_vox_size[0]);
    
    Dtype *occupancy_label_downscale = new Dtype[num_label_voxels];
    Dtype *segmentation_label_downscale = new Dtype[num_label_voxels];
    Dtype *tsdf_data_downscale = new Dtype[num_label_voxels];
    if (label_downscale > 1){
       
       DownsampleLabel(data_crop_vox_size, label_vox_size, label_downscale,
                      occupancy_label_crop, occupancy_label_downscale,
                      segmentation_label_crop, segmentation_label_downscale,
                      vox_tsdf, tsdf_data_downscale);
    }else{
      if (FLAGS_shuran_chatter) {
        LOG(INFO) << "label_downscale: " << label_downscale;
      }
      memcpy(occupancy_label_downscale , occupancy_label_crop, num_label_voxels * sizeof(Dtype));
      memcpy(segmentation_label_downscale , segmentation_label_crop, num_label_voxels * sizeof(Dtype));
      memcpy(tsdf_data_downscale , vox_tsdf, num_label_voxels * sizeof(Dtype));
    }
    // Copy labels to GPU
//    CUDA_CHECK(
//              cudaMemcpy(occupancy_label_GPU + batch_idx * num_label_voxels,
//                         occupancy_label_downscale,
//                         num_label_voxels * sizeof(Dtype),
//                         cudaMemcpyHostToDevice));
//    CUDA_CHECK(
//              cudaMemcpy(segmentation_label_GPU + batch_idx * num_label_voxels,
//                         segmentation_label_downscale,
//                         num_label_voxels * sizeof(Dtype),
//                         cudaMemcpyHostToDevice));

    if (occ_label != nullptr) {
      memcpy(occ_label->mutable_cpu_data() + batch_idx * num_label_voxels,
             occupancy_label_downscale, num_label_voxels * sizeof(Dtype));
    }
    memcpy(seg_label->mutable_cpu_data() + batch_idx * num_label_voxels,
           segmentation_label_downscale, num_label_voxels * sizeof(Dtype));

    // Find number of occupied voxels
    // Save voxel indices of background
    // Set label weights of occupied voxels as 1
    int num_occ_voxels = 0;
    std::vector<int> bg_voxel_idx;
    Dtype *occupancy_weight = new Dtype[num_label_voxels];
    Dtype *segmentation_weight = new Dtype[num_label_voxels];
    //Dtype *segmentation_surf_weight = new Dtype[num_label_voxels];

    memset(occupancy_weight, 0, num_label_voxels * sizeof(Dtype));
    memset(segmentation_weight, 0, num_label_voxels * sizeof(Dtype));
    //memset(segmentation_surf_weight, 0, num_label_voxels * sizeof(Dtype));

    for (int i = 0; i < num_label_voxels; ++i) {
      if (Dtype(occupancy_label_downscale[i]) > 0) {
          if (tsdf_data_downscale[i] < -0.5) {
            // forground voxels in unobserved region
            num_occ_voxels++;
            occupancy_weight[i] = Dtype(occupancy_class_weight[1]);
          }
      } else {
        if (tsdf_data_downscale[i] < -0.5) {
          bg_voxel_idx.push_back(i); // background voxels in unobserved regoin 
        }
      }

      if (Dtype(segmentation_label_downscale[i]) > 0 && Dtype(segmentation_label_downscale[i]) < 255) {
        // foreground voxels within room 
        if (surf_only){
          if(abs(tsdf_data_downscale[i]) < 0.5){
          segmentation_weight[i] = Dtype(segmentation_class_weight[(int) segmentation_label_downscale[i]]);
          }
        }else{
          segmentation_weight[i] = Dtype(segmentation_class_weight[(int) segmentation_label_downscale[i]]);
        }
        // if (abs(tsdf_data_downscale[i]) < 0.5) {
        //   segmentation_surf_weight[i] = Dtype(
        //       segmentation_class_weight[(int) (segmentation_label_downscale[i])]);
        // }
      }

    }

    // Raise the weight for a few indices of background voxels
    std::random_device tmp_rand_rd;
    std::mt19937 tmp_rand_mt(tmp_rand_rd());
    int segnegcout = 0;
    int segnegtotal = floor(sample_neg_obj_ratio * (float) num_occ_voxels);

    if (bg_voxel_idx.size() > 0) {
      std::uniform_real_distribution<double> tmp_rand_dist(
          0, (float) (bg_voxel_idx.size()) - 0.0001);
      for (int i = 0; i < num_occ_voxels; ++i) {
        int rand_idx = (int) (std::floor(tmp_rand_dist(tmp_rand_mt)));
        occupancy_weight[bg_voxel_idx[rand_idx]] = Dtype(
            occupancy_class_weight[0]);
        if (segnegcout < segnegtotal && Dtype(segmentation_label_downscale[bg_voxel_idx[rand_idx]]) < 255 ) {
          // background voxels within room 
          segmentation_weight[bg_voxel_idx[rand_idx]] = Dtype(
              segmentation_class_weight[0]);
          segnegcout++;
        }
      }
    }
    
    if (occ_weight != nullptr) {
      memcpy(occ_weight->mutable_cpu_data() + batch_idx * num_label_voxels,
             occupancy_weight, num_label_voxels * sizeof(Dtype));
    }
    memcpy(seg_weight->mutable_cpu_data() + batch_idx * num_label_voxels,
           segmentation_weight, num_label_voxels * sizeof(Dtype));

    // // Visualize
    //SaveVox2Ply("vis_tsdf_" + std::to_string(batch_idx) + ".ply", data_crop_vox_size, vox_tsdf); // "vis_tsdf_" + data_filenames[counter] + ".ply"
    // if (add_height) {
    //   Dtype * vox_height = new Dtype[num_crop_voxels];
    //   CUDA_CHECK(cudaMemcpy(vox_height, tmp_vox_height_GPU, num_crop_voxels * sizeof(Dtype), cudaMemcpyDeviceToHost));
    //   SaveVoxHeight2Ply("vis_height_" + std::to_string(batch_idx) + ".ply", data_crop_vox_size, vox_height);
    //   delete [] vox_height;
    // }
    // SaveVox2Ply("vis_tsdf_" + std::to_string(batch_idx) + ".ply", label_vox_size, tsdf_data_downscale);
    // SaveVoxLabel2Ply("vis_occ_label_" + std::to_string(batch_idx) + ".ply", label_vox_size, label_downscale, occupancy_label_downscale);
    // SaveVoxLabel2Ply("vis_seg_label_" + std::to_string(batch_idx) + ".ply", label_vox_size, label_downscale, segmentation_label_downscale);
    // SaveVoxWeight2Ply("vis_occ_weight_" + std::to_string(batch_idx) + ".ply", label_vox_size, label_downscale, occupancy_weight);
    // SaveVoxWeight2Ply("vis_seg_weight_" + std::to_string(batch_idx) + ".ply", label_vox_size, label_downscale, segmentation_weight);
    // SaveVoxWeight2Ply("vis_seg_surf_weight_" + std::to_string(batch_idx) + ".ply", label_vox_size, label_downscale, segmentation_surf_weight);

    if (data_param.tsdf_type() > 0) {
        // transfrom TSDF if necsessary
        int THREADS_NUM = 1024;
        int BLOCK_NUM = int((num_crop_voxels + size_t(THREADS_NUM) - 1) / THREADS_NUM);
        tsdfTransform <<< BLOCK_NUM, THREADS_NUM >>> (vox_info_GPU, tmp_tsdf_data_GPU, data_param.tsdf_type());
        CUDA_CHECK(cudaMemcpy(tsdf->mutable_cpu_data() + num_crop_voxels * batch_idx, tmp_tsdf_data_GPU,
                              num_crop_voxels * sizeof(Dtype),
                              cudaMemcpyDeviceToHost));
    }

    // // Free memory
    delete[] depth_data;
    delete[] vox_tsdf;
    delete[] vox_weight_CPU;
    delete[] tsdf_data_downscale;
    delete[] occupancy_label_full;
    delete[] occupancy_label_crop;
    delete[] occupancy_label_downscale;
    delete[] occupancy_weight;
    delete[] segmentation_label_full;
    delete[] segmentation_label_crop;
    delete[] segmentation_label_downscale;
    delete[] segmentation_weight;
    //delete[] segmentation_surf_weight;
  }
}


template<typename Dtype>
int SuncgDataLayer<Dtype>::numofitems() {
  return data_filenames.size();
};


template<typename Dtype>
void SuncgDataLayer<Dtype>::Shuffle() {
  //std::shuffle(sceneMetaList.begin(),sceneMetaList.end(), rng );
  caffe::rng_t *rng = static_cast<caffe::rng_t *>(rng_->generator());
  shuffle(data_filenames.begin(), data_filenames.end(), rng);
  return;
};

INSTANTIATE_CLASS(SuncgDataLayer);
REGISTER_LAYER_CLASS(SuncgData);

} // caffe