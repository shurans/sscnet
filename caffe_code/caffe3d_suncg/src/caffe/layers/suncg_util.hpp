#include <ctime>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <limits>
#include <cstring>
#include <dirent.h>
#include <sstream>
#include <iomanip>
#include <thread>
#include <pwd.h>
#include <cstdlib>
#include <cerrno>
//#include <libio.h>
#include <sys/types.h>
#include <sys/stat.h>

/*-------------------- Useful Utility Functions ---------------------*/
// Read in camera List

// find mode  in an vector besides zeros
template <typename Dtype>
float modeLargerZero(const std::vector<Dtype>& values) {
  Dtype old_mode = 0;
  Dtype old_count = 0;
  for (size_t n = 0; n < values.size(); ++n) {
    if (values[n] > 0 && values[n] < 255) {
      Dtype mode = values[n];
      Dtype count = std::count(values.begin() + n + 1, values.end(), mode);

      if (count > old_count) {
        old_mode = mode;
        old_count = count;
      }
    }
  }
  return old_mode;
}

// find mode of in an vector
template <typename Dtype>
float mode(const std::vector<Dtype>& values) {
  Dtype old_mode = 0;
  Dtype old_count = 0;
  for (size_t n = 0; n < values.size(); ++n) {
    float mode = values[n];
    float count = std::count(values.begin() + n + 1, values.end(), mode);

    if (count > old_count) {
      old_mode = mode;
      old_count = count;
    }
  }
  return old_mode;
}

// Load an MxN matrix from a comma or space delimited text file
std::vector<float> ReadMatFile(std::string filename, int M, int N) {
  std::vector<float> matrix;
  FILE *fp = fopen(filename.c_str(), "r");
  for (int i = 0; i < M * N; i++) {
    float tmp;
    int iret = fscanf(fp, "%f", &tmp);
    matrix.push_back(tmp);
  }
  fclose(fp);
  return matrix;
}

// Simple timer: toc prints the time elapsed since tic
std::clock_t tic_toc_timer;
void tic() {
  tic_toc_timer = clock();
}
void toc() {
  std::clock_t toc_timer = clock();
  printf("Elapsed time is %f seconds.\n", double(toc_timer - tic_toc_timer) / CLOCKS_PER_SEC);
}

// Run a system command
void SysCommand(std::string str) {
  if (system(str.c_str()))
    return;
}

// Generate random float
float GenRandFloat(float min, float max) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(min, max - 0.0001);
  return dist(mt);
}

// Generate random string
std::string GenRandStr(size_t len) {
  auto randchar = []() -> char {
      // const char charset[] =
      //   "0123456789"
      //   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      //   "abcdefghijklmnopqrstuvwxyz";
      const char charset[] =
          "0123456789"
              "abcdefghijklmnopqrstuvwxyz";
      // const size_t max_index = (sizeof(charset) - 1);
      // return charset[rand() % max_index];
      return charset[((int) std::floor(GenRandFloat(0.0f, (float) sizeof(charset) - 1)))];
  };
  std::string str(len, 0);
  std::generate_n(str.begin(), len, randchar);
  return str;
}


/*-------------------- Useful Utility Functions ---------------------*/
// Read in camera List

struct SceneMeta{
    std::string sceneId;
    int roomId;
    int floorId;
    std::vector< std::vector<float> > extrinsics;
    int activityId;
    //std::vector<AnnoMeta> annotationList;
};

std::vector <SceneMeta> readList(std::string file_list){
  std::cout<<"loading file "<<file_list<<"\n";
  FILE* fp = fopen(file_list.c_str(),"rb");
  if (fp==NULL) { std::cout<<"fail to open file: "<<file_list<<std::endl; exit(EXIT_FAILURE); }

  std::vector <SceneMeta> sceneMetaList;
  float vx, vy, vz, tx, ty, tz, ux, uy, uz, rx, ry, rz;
  int idx = 0;
  while (feof(fp)==0) {
    SceneMeta Scene;
    int cameraPoseLen = 0;
    int numAct = 0;
    int anno_Id =0;
    char SceneId[100];
    int res = fscanf(fp, "%s %d %d %d %d %d %d", SceneId, &Scene.floorId, &Scene.roomId, &Scene.activityId, &anno_Id, &numAct, &cameraPoseLen);

    if (res==7){
      Scene.sceneId.append(SceneId);
      Scene.extrinsics.resize(cameraPoseLen);

      for (int i = 0; i < cameraPoseLen; i++){
        int res1 = fscanf(fp, "%f%f%f%f%f%f%f%f%f%f%f%f", &vx, &vy, &vz, &tx, &ty, &tz, &ux, &uy, &uz, &rx, &ry, &rz);
        Scene.extrinsics[i].resize(16);
        Scene.extrinsics[i][0]  = rx; Scene.extrinsics[i][1]  = -ux; Scene.extrinsics[i][2]  = tx;  Scene.extrinsics[i][3]  = vx;
        Scene.extrinsics[i][4]  = ry; Scene.extrinsics[i][5]  = -uy; Scene.extrinsics[i][6]  = ty;  Scene.extrinsics[i][7]  = vy;
        Scene.extrinsics[i][8]  = rz; Scene.extrinsics[i][9]  = -uz; Scene.extrinsics[i][10] = tz;  Scene.extrinsics[i][11] = vz;
        Scene.extrinsics[i][12] = 0 ; Scene.extrinsics[i][13] = 0;   Scene.extrinsics[i][14] = 0;   Scene.extrinsics[i][15] = 1;
      }
      sceneMetaList.push_back(Scene);
      idx++;
    }else{
      break;
    }
  }
  fclose(fp);
  std::cout<<"finish loading scene "<<"\n";
  return sceneMetaList;
}


// Check if file exists
bool FileExists(const std::string &filepath) {
  std::ifstream file(filepath);
  return (!file.fail());
}

// Return all files in directory using search string
void GetFilesInDir(const std::vector<std::string> &directories, std::vector<std::string> &file_list, const std::string &search_string) {
  DIR *dir;
  struct dirent *ent;
  for (int i =0;i<directories.size();i++){
    std::string directory = directories[i];
    if ((dir = opendir (directory.c_str())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
        std::string filename(ent->d_name);
        if (filename.find(search_string) != std::string::npos && filename != "." && filename != ".."){
          filename = directory +"/"+ filename;
          file_list.push_back(filename);
        }
      }
      closedir (dir);
    } else {
      perror ("Error: could not look into directory!");
    }
  }
  LOG(INFO) << "total number of files: "<<file_list.size();
  std::sort(file_list.begin(),  file_list.end());
  //std::cin.ignore();
}

void GetFilesByList(const std::vector<std::string> &directories, std::vector<std::string> &file_list, std::string cameraListFile){
  for (int i =0;i<directories.size();i++){
    std::string list_file = directories[i] + "/" + cameraListFile;
    std::vector <SceneMeta>  sceneMetaList =readList(list_file);
    int curr_frame = 0;
    for (int j = 0; j < sceneMetaList.size(); j++){
      char buff2[100];
      sprintf(buff2, "%08d_%s_fl%03d_rm%04d_%04d",j, sceneMetaList[j].sceneId.c_str(), sceneMetaList[j].floorId,sceneMetaList[j].roomId,curr_frame);
      std::string  fileDepth = directories[i] + "/" + buff2 + ".png";
      file_list.push_back(fileDepth);
    }
    std::cout<<"read in :"<< list_file<< " : "<<sceneMetaList.size()<<std::endl;

  }
  std::cout<<"Total number of data : "<< file_list.size() <<std::endl;
}

void GetFiles(const std::vector<std::string> &directories, std::vector<std::string> &file_list,  const std::string cameraListFile, const std::string &search_string){
  std::string list_file = directories[0] + "/" + cameraListFile;
  if (FileExists(list_file)) {
    std::cout<<"List file exist : "<< list_file <<std::endl;
    GetFilesByList( directories, file_list, cameraListFile);
  }else{
    std::cout<<"List file not exist : "<< list_file <<std::endl;
    GetFilesInDir( directories, file_list, search_string);
  }
}

// Return indices of all occurances of substr in str
std::vector<size_t> FindSubstr(const std::string &str, const std::string &substr) {
  std::vector<size_t> substr_idx;
  size_t tmp_idx = str.find(substr, 0);
  while (tmp_idx != std::string::npos) {
    substr_idx.push_back(tmp_idx);
    tmp_idx = str.find(substr, tmp_idx + 1);
  }
  return substr_idx;
}

// Load a 4x4 identity matrix
// Matrices are stored from left-to-right, top-to-bottom
void LoadIdentityMat(float mOut[16]) {
  mOut[0] = 1.0f;  mOut[1] = 0.0f;  mOut[2] = 0.0f;  mOut[3] = 0.0f;
  mOut[4] = 0.0f;  mOut[5] = 1.0f;  mOut[6] = 0.0f;  mOut[7] = 0.0f;
  mOut[8] = 0.0f;  mOut[9] = 0.0f;  mOut[10] = 1.0f; mOut[11] = 0.0f;
  mOut[12] = 0.0f; mOut[13] = 0.0f; mOut[14] = 0.0f; mOut[15] = 1.0f;
}

// 4x4 matrix multiplication
// Matrices are stored from left-to-right, top-to-bottom
void MulMat(const float m1[16], const float m2[16], float mOut[16]) {
  mOut[0]  = m1[0] * m2[0]  + m1[1] * m2[4]  + m1[2] * m2[8]   + m1[3] * m2[12];
  mOut[1]  = m1[0] * m2[1]  + m1[1] * m2[5]  + m1[2] * m2[9]   + m1[3] * m2[13];
  mOut[2]  = m1[0] * m2[2]  + m1[1] * m2[6]  + m1[2] * m2[10]  + m1[3] * m2[14];
  mOut[3]  = m1[0] * m2[3]  + m1[1] * m2[7]  + m1[2] * m2[11]  + m1[3] * m2[15];

  mOut[4]  = m1[4] * m2[0]  + m1[5] * m2[4]  + m1[6] * m2[8]   + m1[7] * m2[12];
  mOut[5]  = m1[4] * m2[1]  + m1[5] * m2[5]  + m1[6] * m2[9]   + m1[7] * m2[13];
  mOut[6]  = m1[4] * m2[2]  + m1[5] * m2[6]  + m1[6] * m2[10]  + m1[7] * m2[14];
  mOut[7]  = m1[4] * m2[3]  + m1[5] * m2[7]  + m1[6] * m2[11]  + m1[7] * m2[15];

  mOut[8]  = m1[8] * m2[0]  + m1[9] * m2[4]  + m1[10] * m2[8]  + m1[11] * m2[12];
  mOut[9]  = m1[8] * m2[1]  + m1[9] * m2[5]  + m1[10] * m2[9]  + m1[11] * m2[13];
  mOut[10] = m1[8] * m2[2]  + m1[9] * m2[6]  + m1[10] * m2[10] + m1[11] * m2[14];
  mOut[11] = m1[8] * m2[3]  + m1[9] * m2[7]  + m1[10] * m2[11] + m1[11] * m2[15];

  mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8]  + m1[15] * m2[12];
  mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9]  + m1[15] * m2[13];
  mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
  mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
}

// 4x4 matrix inversion
// Matrices are stored from left-to-right, top-to-bottom
bool InvMat(const float m[16], float invOut[16]) {
  float inv[16], det;
  int i;
  inv[0] = m[5]  * m[10] * m[15] -
           m[5]  * m[11] * m[14] -
           m[9]  * m[6]  * m[15] +
           m[9]  * m[7]  * m[14] +
           m[13] * m[6]  * m[11] -
           m[13] * m[7]  * m[10];

  inv[4] = -m[4]  * m[10] * m[15] +
           m[4]  * m[11] * m[14] +
           m[8]  * m[6]  * m[15] -
           m[8]  * m[7]  * m[14] -
           m[12] * m[6]  * m[11] +
           m[12] * m[7]  * m[10];

  inv[8] = m[4]  * m[9] * m[15] -
           m[4]  * m[11] * m[13] -
           m[8]  * m[5] * m[15] +
           m[8]  * m[7] * m[13] +
           m[12] * m[5] * m[11] -
           m[12] * m[7] * m[9];

  inv[12] = -m[4]  * m[9] * m[14] +
            m[4]  * m[10] * m[13] +
            m[8]  * m[5] * m[14] -
            m[8]  * m[6] * m[13] -
            m[12] * m[5] * m[10] +
            m[12] * m[6] * m[9];

  inv[1] = -m[1]  * m[10] * m[15] +
           m[1]  * m[11] * m[14] +
           m[9]  * m[2] * m[15] -
           m[9]  * m[3] * m[14] -
           m[13] * m[2] * m[11] +
           m[13] * m[3] * m[10];

  inv[5] = m[0]  * m[10] * m[15] -
           m[0]  * m[11] * m[14] -
           m[8]  * m[2] * m[15] +
           m[8]  * m[3] * m[14] +
           m[12] * m[2] * m[11] -
           m[12] * m[3] * m[10];

  inv[9] = -m[0]  * m[9] * m[15] +
           m[0]  * m[11] * m[13] +
           m[8]  * m[1] * m[15] -
           m[8]  * m[3] * m[13] -
           m[12] * m[1] * m[11] +
           m[12] * m[3] * m[9];

  inv[13] = m[0]  * m[9] * m[14] -
            m[0]  * m[10] * m[13] -
            m[8]  * m[1] * m[14] +
            m[8]  * m[2] * m[13] +
            m[12] * m[1] * m[10] -
            m[12] * m[2] * m[9];

  inv[2] = m[1]  * m[6] * m[15] -
           m[1]  * m[7] * m[14] -
           m[5]  * m[2] * m[15] +
           m[5]  * m[3] * m[14] +
           m[13] * m[2] * m[7] -
           m[13] * m[3] * m[6];

  inv[6] = -m[0]  * m[6] * m[15] +
           m[0]  * m[7] * m[14] +
           m[4]  * m[2] * m[15] -
           m[4]  * m[3] * m[14] -
           m[12] * m[2] * m[7] +
           m[12] * m[3] * m[6];

  inv[10] = m[0]  * m[5] * m[15] -
            m[0]  * m[7] * m[13] -
            m[4]  * m[1] * m[15] +
            m[4]  * m[3] * m[13] +
            m[12] * m[1] * m[7] -
            m[12] * m[3] * m[5];

  inv[14] = -m[0]  * m[5] * m[14] +
            m[0]  * m[6] * m[13] +
            m[4]  * m[1] * m[14] -
            m[4]  * m[2] * m[13] -
            m[12] * m[1] * m[6] +
            m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] +
           m[1] * m[7] * m[10] +
           m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] -
           m[9] * m[2] * m[7] +
           m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] -
           m[0] * m[7] * m[10] -
           m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] +
           m[8] * m[2] * m[7] -
           m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] +
            m[0] * m[7] * m[9] +
            m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] +
            m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] -
            m[0] * m[6] * m[9] -
            m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] -
            m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0 / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}


template<typename Dtype>
void DownsampleLabel(
    std::vector<int> data_vox_size, std::vector<int> label_vox_size,
    int label_downscale,
    Dtype *occupancy_label_full, Dtype *occupancy_label_downscale,
    Dtype *segmentation_label_full,
    Dtype *segmentation_label_downscale,
    Dtype *tsdf_data_full, Dtype *tsdf_data_downscale) {

  Dtype emptyT = (0.95 * label_downscale * label_downscale * label_downscale);
  for (int i = 0;
       i < label_vox_size[0] * label_vox_size[1] * label_vox_size[2]; ++i) {
    int z = floor(i / (label_vox_size[0] * label_vox_size[1]));
    int y = floor(
        (i - (z * label_vox_size[0] * label_vox_size[1])) / label_vox_size[0]);
    int x = i - (z * label_vox_size[0] * label_vox_size[1]) -
            (y * label_vox_size[0]);

    std::vector<Dtype> field_vals;
    std::vector<Dtype> tsdf_vals;
    int zero_count = 0;
    for (int tmp_x = x * label_downscale;
         tmp_x < (x + 1) * label_downscale; ++tmp_x) {
      for (int tmp_y = y * label_downscale;
           tmp_y < (y + 1) * label_downscale; ++tmp_y) {
        for (int tmp_z = z * label_downscale;
             tmp_z < (z + 1) * label_downscale; ++tmp_z) {
          int tmp_vox_idx = tmp_z * data_vox_size[0] * data_vox_size[1] +
                            tmp_y * data_vox_size[0] + tmp_x;
          field_vals.push_back(segmentation_label_full[tmp_vox_idx]);
          tsdf_vals.push_back(Dtype(tsdf_data_full[tmp_vox_idx]));
          if (segmentation_label_full[tmp_vox_idx] < Dtype(0.001f) || segmentation_label_full[tmp_vox_idx] > Dtype(254)) {
            zero_count++;
          }
        }
      }
    }

    tsdf_data_downscale[i] =
        std::accumulate(tsdf_vals.begin(), tsdf_vals.end(), 0.0) /
        tsdf_vals.size();
    if (zero_count > emptyT) {
      occupancy_label_downscale[i] = Dtype(0.0f);
      segmentation_label_downscale[i] = Dtype(mode(field_vals));
    } else {
      occupancy_label_downscale[i] = Dtype(1.0f);
      segmentation_label_downscale[i] = Dtype(
          modeLargerZero(field_vals)); // object label mode without zeros
    }
  }

}