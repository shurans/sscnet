// /*-------------------- CUDA Utilities ---------------------*/

// // Return fatal error and safely exit program
// void FatalError(const int lineNumber = 0) {
//   std::cerr << "FatalError";
//   if (lineNumber != 0) std::cerr << " at LINE " << lineNumber;
//   std::cerr << ". Program Terminated." << std::endl;
//   cudaDeviceReset();
//   exit(EXIT_FAILURE);
// }

// // Check CUDA line and return CUDA error status
// void checkCUDA(const int lineNumber, cudaError_t status) {
//   if (status != cudaSuccess) {
//     std::cerr << "CUDA failure at LINE " << lineNumber << ": " << status << std::endl;
//     FatalError();
//   }
// }

#include <cmath>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "caffe/common.hpp"

#include "suncg_fusion.hpp"

#define ComputeT float
#define StorageT float

#define CPUStorage2ComputeT(x) (x)
#define GPUCompute2StorageT(x) (x)
#define GPUStorage2ComputeT(x) (x)
#define CPUCompute2StorageT(x) (x)

float GenRandFloat(float min, float max);


/*-------------------- ComputeAccurateTSDF --------------------*/
__global__ 
void CompleteTSDF (ComputeT * vox_info, ComputeT * occupancy_label_crop_GPU , StorageT * vox_tsdf) {
    // Get voxel volume parameters
    ComputeT vox_unit = vox_info[0];
    ComputeT vox_margin = vox_info[1];
    int vox_size[3];
    for (int i = 0; i < 3; ++i)
      vox_size[i] = vox_info[i + 2];
   

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (vox_idx >= vox_size[0] * vox_size[1] * vox_size[2]){
        return;
    }
    int z = float((vox_idx / ( vox_size[0] * vox_size[1]))%vox_size[2]) ;
    int y = float((vox_idx / vox_size[0]) % vox_size[1]);
    int x = float(vox_idx % vox_size[0]);
    int search_region = (int)round(vox_margin/vox_unit);

    if (occupancy_label_crop_GPU[vox_idx] >0 ){
        vox_tsdf[vox_idx] = -0.001;// inside mesh
        return;
    }

    for (int iix = max(0,x-search_region); iix < min((int)vox_size[0],x+search_region+1); iix++){
      for (int iiy = max(0,y-search_region); iiy < min((int)vox_size[1],y+search_region+1); iiy++){
        for (int iiz = max(0,z-search_region); iiz < min((int)vox_size[2],z+search_region+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (occupancy_label_crop_GPU[iidx] > 0){
                float xd = abs(x - iix);
                float yd = abs(y - iiy);
                float zd = abs(z - iiz);
                float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/(float)search_region;
                if (tsdf_value < abs(vox_tsdf[vox_idx])){
                  vox_tsdf[vox_idx] = GPUCompute2StorageT(tsdf_value);
                }
            }
        }
      }
    }
}


__global__
void depth2Grid(ComputeT *  cam_info, ComputeT *  vox_info,  ComputeT * depth_data, ComputeT * vox_binary_GPU){
  // Get camera information
  int frame_width = cam_info[0];
  //int frame_height = cam_info[1];
  ComputeT cam_K[9];
  for (int i = 0; i < 9; ++i)
    cam_K[i] = cam_info[i + 2];
  ComputeT cam_pose[16];
  for (int i = 0; i < 16; ++i)
    cam_pose[i] = cam_info[i + 11];

  // Get voxel volume parameters
  ComputeT vox_unit = vox_info[0];
  //ComputeT vox_margin = vox_info[1];
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = vox_info[i + 2];
  ComputeT vox_origin[3];
  for (int i = 0; i < 3; ++i)
    vox_origin[i] = vox_info[i + 5];


  // Get point in world coordinate
  int pixel_x = blockIdx.x;
  int pixel_y = threadIdx.x;

  ComputeT point_depth = depth_data[pixel_y * frame_width + pixel_x];

  ComputeT point_cam[3] = {0};
  point_cam[0] =  (pixel_x - cam_K[2])*point_depth/cam_K[0];
  point_cam[1] =  (pixel_y - cam_K[5])*point_depth/cam_K[4];
  point_cam[2] =  point_depth;

  ComputeT point_base[3] = {0};

  point_base[0] = cam_pose[0 * 4 + 0]* point_cam[0] + cam_pose[0 * 4 + 1]*  point_cam[1] + cam_pose[0 * 4 + 2]* point_cam[2];
  point_base[1] = cam_pose[1 * 4 + 0]* point_cam[0] + cam_pose[1 * 4 + 1]*  point_cam[1] + cam_pose[1 * 4 + 2]* point_cam[2];
  point_base[2] = cam_pose[2 * 4 + 0]* point_cam[0] + cam_pose[2 * 4 + 1]*  point_cam[1] + cam_pose[2 * 4 + 2]* point_cam[2];

  point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
  point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
  point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];


  //printf("vox_origin: %f,%f,%f\n",vox_origin[0],vox_origin[1],vox_origin[2]);
  // World coordinate to grid coordinate
  int z = (int)floor((point_base[0] - vox_origin[0])/vox_unit);
  int x = (int)floor((point_base[1] - vox_origin[1])/vox_unit);
  int y = (int)floor((point_base[2] - vox_origin[2])/vox_unit);
  //printf("point_base: %f,%f,%f, %d,%d,%d, %d,%d,%d \n",point_base[0],point_base[1],point_base[2], z, x, y, vox_size[0],vox_size[1],vox_size[2]);

  // mark vox_binary_GPU


  if( x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]){
    int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
    vox_binary_GPU[vox_idx] = ComputeT(1.0);
  }
}
__global__
void SquaredDistanceTransform(ComputeT * cam_info, ComputeT * vox_info, ComputeT * depth_data, ComputeT * vox_binary_GPU , StorageT * vox_tsdf, StorageT * vox_height) {
  // debug
  int frame_width = cam_info[0];
  int frame_height = cam_info[1];
  ComputeT cam_K[9];
  for (int i = 0; i < 9; ++i)
    cam_K[i] = cam_info[i + 2];
  ComputeT cam_pose[16];
  for (int i = 0; i < 16; ++i)
    cam_pose[i] = cam_info[i + 11];

  // Get voxel volume parameters
  ComputeT vox_unit = vox_info[0];
  ComputeT vox_margin = vox_info[1];
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = vox_info[i + 2];
  ComputeT vox_origin[3];
  for (int i = 0; i < 3; ++i)
    vox_origin[i] = vox_info[i + 5];

  int z = blockIdx.x;
  int y = threadIdx.x;
  int search_region = (int)round(vox_margin/vox_unit);

  for (int x = 0; x < vox_size[0]; ++x) {
    int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;

    //vox_tsdf[vox_idx] = 1.0 - vox_binary_GPU[vox_idx];

    // Get point in world coordinates XYZ -> YZX
    ComputeT point_base[3] = {0};
    point_base[0] = ComputeT(z) * vox_unit + vox_origin[0];
    point_base[1] = ComputeT(x) * vox_unit + vox_origin[1];
    point_base[2] = ComputeT(y) * vox_unit + vox_origin[2];

    // Encode height from floor
    if (vox_height != NULL) {
      ComputeT height_val = ((point_base[2] + 0.05f) / 3.0f);
      vox_height[vox_idx] = GPUCompute2StorageT(fmin(1.0f, fmax(height_val, 0.0f)));
    }

    // Get point in current camera coordinates
    ComputeT point_cam[3] = {0};
    point_base[0] = point_base[0] - cam_pose[0 * 4 + 3];
    point_base[1] = point_base[1] - cam_pose[1 * 4 + 3];
    point_base[2] = point_base[2] - cam_pose[2 * 4 + 3];
    point_cam[0] = cam_pose[0 * 4 + 0] * point_base[0] + cam_pose[1 * 4 + 0] * point_base[1] + cam_pose[2 * 4 + 0] * point_base[2];
    point_cam[1] = cam_pose[0 * 4 + 1] * point_base[0] + cam_pose[1 * 4 + 1] * point_base[1] + cam_pose[2 * 4 + 1] * point_base[2];
    point_cam[2] = cam_pose[0 * 4 + 2] * point_base[0] + cam_pose[1 * 4 + 2] * point_base[1] + cam_pose[2 * 4 + 2] * point_base[2];
    if (point_cam[2] <= 0)
      continue;

    // Project point to 2D
    int pixel_x = roundf(cam_K[0] * (point_cam[0] / point_cam[2]) + cam_K[2]);
    int pixel_y = roundf(cam_K[4] * (point_cam[1] / point_cam[2]) + cam_K[5]);
    if (pixel_x < 0 || pixel_x >= frame_width || pixel_y < 0 || pixel_y >= frame_height){ // outside FOV
      //vox_tsdf[vox_idx] = GPUCompute2StorageT(-1.0);
      continue;
    }


    // Get depth
    ComputeT point_depth = depth_data[pixel_y * frame_width + pixel_x];
    if (point_depth < ComputeT(0.0f) || point_depth > ComputeT(10.0f))
      continue;
    if (roundf(point_depth) == 0){ // mising depth
      vox_tsdf[vox_idx] = GPUCompute2StorageT(-1.0);
      continue;
    }


    // Get depth difference
    ComputeT point_dist = (point_depth - point_cam[2]) * sqrtf(1 + powf((point_cam[0] / point_cam[2]), 2) + powf((point_cam[1] / point_cam[2]), 2));
    ComputeT sign = point_dist/abs(point_dist);
    vox_tsdf[vox_idx] = GPUCompute2StorageT(sign);
    //vox_tsdf[vox_idx] = sign*fmin(1, abs(point_dist)/vox_margin);
    if (abs(point_dist) < 2 * vox_margin){
      for (int iix = max(0,x-search_region); iix < min((int)vox_size[0],x+search_region+1); iix++){
        for (int iiy = max(0,y-search_region); iiy < min((int)vox_size[1],y+search_region+1); iiy++){
          for (int iiz = max(0,z-search_region); iiz < min((int)vox_size[2],z+search_region+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_binary_GPU[iidx] > 0){
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = (sqrtf(xd * xd + yd * yd + zd * zd)*vox_unit)/vox_margin;
              //printf("%f, %f, %f, %f, %f\n",tsdf_value,abs(vox_tsdf[vox_idx]), xd,yd,zd);
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = GPUCompute2StorageT(tsdf_value*sign);
              }
            }
          }
        }
      }
    }
  }
}


void ComputeTSDF(ComputeT * cam_info_CPU, ComputeT * vox_info_CPU,
                 ComputeT * cam_info_GPU, ComputeT * vox_info_GPU,
                 ComputeT * depth_data_GPU,  StorageT * vox_tsdf_GPU, StorageT * vox_height_GPU) {

  int frame_width  = cam_info_CPU[0];
  int frame_height = cam_info_CPU[1];
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = vox_info_CPU[i + 2];
  int num_crop_voxels = vox_size[0] * vox_size[1] * vox_size[2];

  ComputeT *  vox_binary_CPU = new ComputeT[num_crop_voxels];
  memset(vox_binary_CPU, 0, num_crop_voxels * sizeof(ComputeT));
  ComputeT *  vox_binary_GPU;
  CUDA_CHECK(cudaMalloc(&vox_binary_GPU, num_crop_voxels * sizeof(ComputeT)));
  GPU_set_zeros(num_crop_voxels, vox_binary_GPU);
  CUDA_CHECK(cudaGetLastError());

  // from depth map to binaray voxel representation
  depth2Grid<<<frame_width,frame_height>>>(cam_info_GPU, vox_info_GPU, depth_data_GPU, vox_binary_GPU);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(vox_binary_CPU, vox_binary_GPU, num_crop_voxels * sizeof(ComputeT), cudaMemcpyDeviceToHost));

  // distance transform
  SquaredDistanceTransform <<< vox_size[2], vox_size[1] >>> (cam_info_GPU, vox_info_GPU, depth_data_GPU, vox_binary_GPU, vox_tsdf_GPU, vox_height_GPU);

  delete [] vox_binary_CPU;
  CUDA_CHECK(cudaFree(vox_binary_GPU));
}


// Save voxel volume to point cloud ply file for visualization
void SaveVox2Ply(const std::string &filename, std::vector<int> vox_size, StorageT * vox_tsdf) {
  float tsdf_threshold = 0.4f;

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i)
    if (CPUStorage2ComputeT(abs(vox_tsdf[i])) < tsdf_threshold)
      num_points++;

  // Create header for ply file
  FILE *fp = fopen(filename.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i) {

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (CPUStorage2ComputeT(abs(vox_tsdf[i])) < tsdf_threshold) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (vox_size[0] * vox_size[1]));
      int y = floor((i - (z * vox_size[0] * vox_size[1])) / vox_size[0]);
      int x = i - (z * vox_size[0] * vox_size[1]) - (y * vox_size[0]);

      // Convert voxel indices to float, and save coordinates to ply file
      float float_x = (float) x;
      float float_y = (float) y;
      float float_z = (float) z;
      fwrite(&float_x, sizeof(float), 1, fp);
      fwrite(&float_y, sizeof(float), 1, fp);
      fwrite(&float_z, sizeof(float), 1, fp);
      unsigned char color_r = (unsigned char) 255;
      unsigned char color_g = (unsigned char) 0;
      unsigned char color_b = (unsigned char) 0;
      if (CPUStorage2ComputeT(vox_tsdf[i]) < 0) {
        color_r = (unsigned char) 0;
        color_g = (unsigned char) 255;
        color_b = (unsigned char) 0;
      }
      fwrite(&color_r, sizeof(unsigned char), 1, fp);
      fwrite(&color_g, sizeof(unsigned char), 1, fp);
      fwrite(&color_b, sizeof(unsigned char), 1, fp);
    }
  }
  fclose(fp);
}

// Save voxel volume to point cloud ply file for visualization
void SaveVoxHeight2Ply(const std::string &filename, std::vector<int> vox_size, StorageT * vox_height) {

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i)
    num_points++;

  // Create header for ply file
  FILE *fp = fopen(filename.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i) {

    // Compute voxel indices in int for higher positive number range
    int z = floor(i / (vox_size[0] * vox_size[1]));
    int y = floor((i - (z * vox_size[0] * vox_size[1])) / vox_size[0]);
    int x = i - (z * vox_size[0] * vox_size[1]) - (y * vox_size[0]);

    // Convert voxel indices to float, and save coordinates to ply file
    float float_x = (float) x;
    float float_y = (float) y;
    float float_z = (float) z;
    fwrite(&float_x, sizeof(float), 1, fp);
    fwrite(&float_y, sizeof(float), 1, fp);
    fwrite(&float_z, sizeof(float), 1, fp);
    unsigned char color_r = (unsigned char)(255.0f * std::abs(CPUStorage2ComputeT(vox_height[i])) / 3.0);
    unsigned char color_g = (unsigned char)(255.0f * std::abs(CPUStorage2ComputeT(vox_height[i])) / 3.0);
    unsigned char color_b = (unsigned char)(255.0f * std::abs(CPUStorage2ComputeT(vox_height[i])) / 3.0);
    fwrite(&color_r, sizeof(unsigned char), 1, fp);
    fwrite(&color_g, sizeof(unsigned char), 1, fp);
    fwrite(&color_b, sizeof(unsigned char), 1, fp);
  }
  fclose(fp);
}

// Save voxel volume labels to point cloud ply file for visualization
void SaveVoxLabel2Ply(const std::string &filename, std::vector<int> vox_size, int label_downscale, StorageT * vox_label) {

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i)
    if (CPUStorage2ComputeT(vox_label[i]) > 0)
      num_points++;

  // Create header for ply file
  FILE *fp = fopen(filename.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create different colors for each class
  const int num_classes = 36;
  int class_colors[num_classes * 3];
  for (int i = 0; i < num_classes; ++i) {
    class_colors[i * 3 + 0] = (int)(std::round(GenRandFloat(0.0f, 255.0f)));
    class_colors[i * 3 + 1] = (int)(std::round(GenRandFloat(0.0f, 255.0f)));
    class_colors[i * 3 + 2] = (int)(std::round(GenRandFloat(0.0f, 255.0f)));
  }

  // Create point cloud content for ply file
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i) {

    // If class of voxel non-empty, add voxel coordinates to point cloud
    if (CPUStorage2ComputeT(vox_label[i]) > 0) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (vox_size[0] * vox_size[1]));
      int y = floor((i - (z * vox_size[0] * vox_size[1])) / vox_size[0]);
      int x = i - (z * vox_size[0] * vox_size[1]) - (y * vox_size[0]);

      // Convert voxel indices to float, and save coordinates to ply file
      float float_x = (float)x * (float)label_downscale + (float)label_downscale / 2;
      float float_y = (float)y * (float)label_downscale + (float)label_downscale / 2;
      float float_z = (float)z * (float)label_downscale + (float)label_downscale / 2;
      fwrite(&float_x, sizeof(float), 1, fp);
      fwrite(&float_y, sizeof(float), 1, fp);
      fwrite(&float_z, sizeof(float), 1, fp);

      // Save color of class into voxel
      unsigned char color_r = (unsigned char) class_colors[(int)CPUStorage2ComputeT(vox_label[i]) * 3 + 0];
      unsigned char color_g = (unsigned char) class_colors[(int)CPUStorage2ComputeT(vox_label[i]) * 3 + 1];
      unsigned char color_b = (unsigned char) class_colors[(int)CPUStorage2ComputeT(vox_label[i]) * 3 + 2];
      fwrite(&color_r, sizeof(unsigned char), 1, fp);
      fwrite(&color_g, sizeof(unsigned char), 1, fp);
      fwrite(&color_b, sizeof(unsigned char), 1, fp);
    }
  }
  fclose(fp);
}

// Save voxel volume weights to point cloud ply file for visualization
void SaveVoxWeight2Ply(const std::string &filename, std::vector<int> vox_size, int label_downscale, StorageT * vox_label_weight) {

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i)
    if (CPUStorage2ComputeT(vox_label_weight[i]) > 0)
      num_points++;

  // Create header for ply file
  FILE *fp = fopen(filename.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // // Create different colors for each class
  // const int num_classes = 36;
  // int class_colors[num_classes * 3];
  // for (int i = 0; i < num_classes; ++i) {
  //   class_colors[i * 3 + 0] = (int)(std::round(GenRandFloat(0.0f, 255.0f)));
  //   class_colors[i * 3 + 1] = (int)(std::round(GenRandFloat(0.0f, 255.0f)));
  //   class_colors[i * 3 + 2] = (int)(std::round(GenRandFloat(0.0f, 255.0f)));
  // }

  // Create point cloud content for ply file
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i) {

    // If class of voxel non-empty, add voxel coordinates to point cloud
    if (CPUStorage2ComputeT(vox_label_weight[i]) > 0) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (vox_size[0] * vox_size[1]));
      int y = floor((i - (z * vox_size[0] * vox_size[1])) / vox_size[0]);
      int x = i - (z * vox_size[0] * vox_size[1]) - (y * vox_size[0]);

      // Convert voxel indices to float, and save coordinates to ply file
      float float_x = (float)x * (float)label_downscale + (float)label_downscale / 2;
      float float_y = (float)y * (float)label_downscale + (float)label_downscale / 2;
      float float_z = (float)z * (float)label_downscale + (float)label_downscale / 2;
      fwrite(&float_x, sizeof(float), 1, fp);
      fwrite(&float_y, sizeof(float), 1, fp);
      fwrite(&float_z, sizeof(float), 1, fp);

      // Save color of class into voxel
      unsigned char color_r = (unsigned char) 0;
      unsigned char color_g = (unsigned char) 0;
      unsigned char color_b = (unsigned char) 255;
      if (CPUStorage2ComputeT(vox_label_weight[i]) > 1) {
        color_r = (unsigned char) 255;
        color_g = (unsigned char) 0;
        color_b = (unsigned char) 0;
      }
      fwrite(&color_r, sizeof(unsigned char), 1, fp);
      fwrite(&color_g, sizeof(unsigned char), 1, fp);
      fwrite(&color_b, sizeof(unsigned char), 1, fp);
    }
  }
  fclose(fp);
}

// Transform voxel label from single channel to 36 channel volume
__global__
void SetVoxLabel(int num_classes, float * vox_info, float * vox_label_src, float * vox_label_dst) {

  // Get voxel volume parameters
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = (int)(vox_info[i + 2]);

  int z = blockIdx.x;
  int y = threadIdx.x;
  for (int x = 0; x < vox_size[0]; ++x) {
    int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
    int vox_val = vox_label_src[vox_idx];
    for (int i = 0; i < num_classes; ++i) {
      if (vox_val == i) {
        vox_label_dst[i * vox_size[0] * vox_size[1] * vox_size[2] + vox_idx] = 1.0f;
      } else {
        vox_label_dst[i * vox_size[0] * vox_size[1] * vox_size[2] + vox_idx] = 0.0f;
      }
    }
  }
}

/*-------------------- Main Function ---------------------*/

// int main(int argc, char **argv) {

//   // Get camera intrinsics
//   int frame_width = 640; // in pixels
//   int frame_height = 480;
//   float cam_K[9] = {518.8579f, 0.0f, (float)frame_width / 2.0f, 0.0f, 518.8579f, (float)frame_height / 2.0f, 0.0f, 0.0f, 1.0f};

//   // Set voxel volume parameters
//   float vox_unit = 0.02f; // in meters
//   float vox_margin = vox_unit * 5.0f; // in voxels
//   int vox_size[3] = {210, 120, 210};
//   float vox_origin[3] = {}; // in camera coordinates
//   vox_origin[0] = (-(float)vox_size[0] / 2.0f + 0.5f) * vox_unit;
//   vox_origin[1] = (-(float)vox_size[1] / 2.0f + 0.5f) * vox_unit;
//   vox_origin[2] = 0.5f * vox_unit;

//   // CPU malloc voxel volume
//   float * vox_tsdf   = new float[vox_size[0] * vox_size[1] * vox_size[2]];
//   float * vox_weight = new float[vox_size[0] * vox_size[1] * vox_size[2]];
//   memset(vox_weight, 0, sizeof(float) * vox_size[0] * vox_size[1] * vox_size[2]);
//   for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i) {
//     vox_tsdf[i] = 1.0f;
//     vox_weight[i] = 0.0f;
//   }

//   // CPU malloc voxel labels
//   int * vox_label  = new int[vox_size[0] * vox_size[1] * vox_size[2]];
//   memset(vox_label, 0, sizeof(int) * vox_size[0] * vox_size[1] * vox_size[2]);

//   // Get depth image
//   std::string depth_path = "data/depth/0000_000514ade3bcc292a613a4c2755a5050_fl001_rm0001_0000.png";
//   float * depth_data = new float[frame_height * frame_width];
//   ReadDepthImage(depth_path, depth_data, frame_width, frame_height);

//   // Get camera pose
//   // float cam_pose[16] = {-0.9948, -0.0201, 0.1002, 42.9729, -0.1022, 0.1954, -0.9754, 51.5820, 0, -0.9805, -0.1965, 1.3050, 0, 0, 0, 1};
//   float cam_pose[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

//   // Copy camera information to GPU
//   float cam_info[27];
//   cam_info[0] = (float)frame_width;
//   cam_info[1] = (float)frame_height;
//   for (int i = 0; i < 9; ++i)
//     cam_info[i + 2] = cam_K[i];
//   for (int i = 0; i < 16; ++i)
//     cam_info[i + 11] = cam_pose[i];
//   float * d_cam_info;
//   CUDA_CHECKcudaMalloc(&d_cam_info, 27 * sizeof(float)));
//   CUDA_CHECKcudaMemcpy(d_cam_info, cam_info, 27 * sizeof(float), cudaMemcpyHostToDevice));

//   // Copy voxel volume to GPU
//   float * d_vox_tsdf;
//   float * d_vox_weight;
//   CUDA_CHECKcudaMalloc(&d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float)));
//   CUDA_CHECKcudaMalloc(&d_vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float)));
//   CUDA_CHECKcudaMemcpy(d_vox_tsdf, vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyHostToDevice));
//   CUDA_CHECKcudaMemcpy(d_vox_weight, vox_weight, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyHostToDevice));

//   // Copy voxel volume parameters to GPU
//   float vox_info[8];
//   vox_info[0] = vox_unit;
//   vox_info[1] = vox_margin;
//   for (int i = 0; i < 3; ++i)
//     vox_info[i + 2] = vox_size[i];
//   for (int i = 0; i < 3; ++i)
//     vox_info[i + 5] = vox_origin[i];
//   float * d_vox_info;
//   CUDA_CHECKcudaMalloc(&d_vox_info, 8 * sizeof(float)));
//   CUDA_CHECKcudaMemcpy(d_vox_info, vox_info, 8 * sizeof(float), cudaMemcpyHostToDevice));

//   // Copy depth data to GPU
//   float * d_depth_data;
//   CUDA_CHECKcudaMalloc(&d_depth_data, frame_height * frame_width * sizeof(float)));
//   CUDA_CHECKcudaMemcpy(d_depth_data, depth_data, frame_height * frame_width * sizeof(float), cudaMemcpyHostToDevice));

//   // Fuse frame into voxel volume
//   int CUDA_NUM_BLOCKS = vox_size[2];
//   int CUDA_NUM_THREADS = vox_size[1];
//   Integrate<<< CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>>(d_cam_info, d_vox_info, d_depth_data, d_vox_tsdf, d_vox_weight);
//   CUDA_CHECKcudaGetLastError());

//   // Copy voxel volume back to CPU
//   CUDA_CHECKcudaMemcpy(vox_tsdf, d_vox_tsdf, vox_size[0] * vox_size[1] * vox_size[2] * sizeof(float), cudaMemcpyDeviceToHost));
//   SaveVox2Ply("vox.ply", vox_size, vox_tsdf);

//   return 0;
// }
