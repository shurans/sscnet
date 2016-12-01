#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_

#include <future>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

DECLARE_bool(shuran_chatter);

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class SuncgDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit SuncgDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~SuncgDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SuncgData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 5; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  int numofitems();
  void Shuffle();

  std::future<void> lock;
  int epoch_prefetch;
  int batch_size;
  int counter;
  bool shuffle_data;
  bool surf_only;
  // File information
  std::vector<std::string> file_data;
  std::string file_list;
  std::vector<std::string> data_filenames;

  // Camera information
  int frame_width = 640; // in pixels
  int frame_height = 480;
  Dtype cam_K[9] = {518.8579f, 0.0f, (float)frame_width / 2.0f, 0.0f, 518.8579f, (float)frame_height / 2.0f, 0.0f, 0.0f, 1.0f};
  Dtype cam_info[27];

  // Volume parameters
  Dtype vox_unit;
  Dtype vox_margin;
  Dtype vox_info[8];
  std::vector<int> data_full_vox_size;
  std::vector<int> data_crop_vox_size;
  std::vector<int> label_vox_size;

  // Data options
  bool  add_height;
  int   data_num_channel;
  bool  is_cropping_data;
  float sample_neg_obj_ratio;
  bool  occ_emptyonly;
  float offset_value;

  // Segmentation parameters
  std::vector<int> segmentation_class_map;
  int num_segmentation_class;
  std::vector<float> segmentation_class_weight;
  std::vector<float> occupancy_class_weight;

  // Internal GPU variables
  Dtype * cam_info_GPU;
  Dtype * vox_info_GPU;
  Dtype * depth_data_GPU;
  Dtype * vox_weight_GPU;

  // Outgoing GPU variables
  Dtype * vox_data_GPU;
  Dtype * occupancy_label_GPU;
  Dtype * occupancy_weight_GPU;
  Dtype * segmentation_label_GPU;
  Dtype * segmentation_weight_GPU;
  Dtype * segmentation_surf_weight_GPU;

  shared_ptr<Caffe::RNG> rng_;
};


}  // namespace caffe

#endif  // CAFFE_SUNCG_DATA_LAYER_HPP_
