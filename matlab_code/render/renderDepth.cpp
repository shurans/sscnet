// g++ -o renderDepth renderDepth.cpp -lGLU -lOSMesa -I/opt/X11/include/ -L/opt/X11/lib/ `pkg-config --cflags opencv` `pkg-config --libs opencv` -g 
//renderDepth 
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "renderMesh.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

// Check if file exists
bool FileExists(const std::string &filepath) {
  std::ifstream file(filepath.c_str());
  return (!file.fail());
}

struct AnnoMeta{
  int activityId;
  int poseId;
  std::vector< int > inv_obj;
  float poseTs[12];
};

struct SceneMeta{
	std::string sceneId;
	int roomId;
	int floorId;
	std::vector< std::vector<float> > extrinsics;
    int activityId;
    std::vector<AnnoMeta> annotationList;
};

std::vector <SceneMeta> readList(std::string file_list){
	int num_objclass = 36;
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
        
	        if (Scene.activityId>0){
	          Scene.annotationList.resize(numAct);
	          for(int i = 0;i<numAct;i++){
	            
	            int res1 = fscanf(fp, "%d %d %f %f %f %f %f %f %f %f %f %f %f %f", 
	                              &Scene.annotationList[i].activityId,&Scene.annotationList[i].poseId,
	                              &Scene.annotationList[i].poseTs[0], &Scene.annotationList[i].poseTs[1], &Scene.annotationList[i].poseTs[2], &Scene.annotationList[i].poseTs[3], 
	                              &Scene.annotationList[i].poseTs[4], &Scene.annotationList[i].poseTs[5], &Scene.annotationList[i].poseTs[6], &Scene.annotationList[i].poseTs[7],   
	                              &Scene.annotationList[i].poseTs[8], &Scene.annotationList[i].poseTs[9], &Scene.annotationList[i].poseTs[10],&Scene.annotationList[i].poseTs[11]);
	            int inv_objLen =0;
	            res1 = fscanf(fp, "%d",&inv_objLen);
	            Scene.annotationList[i].inv_obj.resize(num_objclass,0);
	            for (int bi = 0; bi < inv_objLen; bi++){
	                int inv_obj_idx = 0;
	                int res1 = fscanf (fp, "%d", &inv_obj_idx);
	                Scene.annotationList[i].inv_obj[inv_obj_idx] = 1;
	            }
	          }
	        }else{
	          Scene.annotationList.resize(0);
	        }

	      	for (int i = 0; i < cameraPoseLen; i++){
	            int res1 = fscanf(fp, "%f%f%f%f%f%f%f%f%f%f%f%f", &vx, &vy, &vz, &tx, &ty, &tz, &ux, &uy, &uz, &rx, &ry, &rz);
	            Scene.extrinsics[i].resize(16);
	            Scene.extrinsics[i][0]  = rx; Scene.extrinsics[i][1]  = -ux; Scene.extrinsics[i][2]  = tx;  Scene.extrinsics[i][3]  = vx;
	            Scene.extrinsics[i][4]  = ry; Scene.extrinsics[i][5]  = -uy; Scene.extrinsics[i][6]  = ty;  Scene.extrinsics[i][7]  = vy;
	            Scene.extrinsics[i][8]  = rz; Scene.extrinsics[i][9]  = -uz; Scene.extrinsics[i][10] = tz;  Scene.extrinsics[i][11] = vz;
	            Scene.extrinsics[i][12] = 0 ; Scene.extrinsics[i][13] = 0;   Scene.extrinsics[i][14] = 0;   Scene.extrinsics[i][15] = 1;
	      	}
	      	sceneMetaList.push_back(Scene);
	        //act_cates[Scene.activityId].push_back(idx);
	        idx++;
		}else{
			break;
		}
	}
	fclose(fp);
	std::cout<<"finish loading scene "<<"\n";
	return sceneMetaList;
}
cv::Mat convertDepthToPNG(float*  depth_data,int im_h,int im_w){
	cv::Mat mat(im_h, im_w, CV_16UC1);
	for (int i = 0; i < im_h; ++i) {
    	for (int j = 0; j < im_w; ++j) {
    		float depthvalue = depth_data[i*im_w+j];
    		ushort s = (ushort)(depthvalue*1000);
    		s = (s >> 13 | s << 3);
    		mat.at<ushort>(i,j)= s;
    	}
    }
    return mat;
}

cv::Mat convertLabelToPNG(unsigned int * label_data ,int im_h,int im_w){
	cv::Mat mat(im_h, im_w, CV_8UC1);
	for (int i = 0; i < im_h; ++i) {
    	for (int j = 0; j < im_w; ++j) {
    		mat.at<uint8_t>(i,j) = (uint8_t)label_data[i*im_w+j];
    	}
    }
    return mat;
}


int main(int argc, char** argv){
	// read in list 
	
	
	int im_w = 640;
	int im_h = 480;
	float camK[9];
	float P[12]={0}; 

	camK[0] = 5.1885790117450188e+02; camK[1] = 0; camK[2] = 320;
	camK[3] = 0; camK[4] = 5.1885790117450188e+02; camK[5] = 240;
	camK[6] = 0; camK[7] = 0; camK[8] = 1;	
	
	if (argc<4){
		// render list 
		//./renderDepth  ../data/depth_49700_49884_test/camera_list_train.list  ../data/depth_49700_49884_test/ 
		//./renderDepth  ../data/depth_1_500/camera_list_train.list  ../data/depth_1_500/ 
		//./renderDepth  ../data/depth_501_1000/camera_list_train.list  ../data/depth_501_1000/ 
		
		//./renderDepth  ../data/depth_1001_3000/camera_list_train.list  ../data/depth_1001_3000/ 
		//./renderDepth  /n/fs/suncg/voxelLabelComplete/data//depth_1_1000_fobj1/camera_list_train.list  /n/fs/suncg/voxelLabelComplete/data//depth_1_1000_fobj1/
		std::string data_root = "/n/fs/modelnet/SUNCG/data/planner5d/";
		std::string file_list = argv[1];
		std::string fileOutPath = argv[2];//"../depth/"

		std::vector <SceneMeta>  sceneMetaList =readList(file_list);
		std::string prev_meshfilename;
		std::string curr_meshfilename;
		MeshScene3D * curr_meshScene = NULL;

    	float* depth_data = (float *) malloc(sizeof(float)*im_w * im_h);
    	unsigned int* label_data = (unsigned int*) malloc(sizeof(unsigned int)*im_w*im_h);
		// for loop the list 
		for (int sceneId_topick = 0; sceneId_topick<sceneMetaList.size(); sceneId_topick++){
			SceneMeta Scene = sceneMetaList[sceneId_topick];
			// load the mesh
	        prev_meshfilename = curr_meshfilename;
			char buff[100];
			sprintf(buff, "fl%03d_rm%04d",Scene.floorId,Scene.roomId);
			curr_meshfilename = data_root + "/meshfiles/" + Scene.sceneId + "/" + buff+".obj";
			std::vector<std::vector<float> > extrinsics = Scene.extrinsics;

	        if (prev_meshfilename==""||prev_meshfilename.compare(curr_meshfilename)!=0){
	            if (curr_meshScene!=NULL) {
	                delete curr_meshScene;
	            }
	             curr_meshScene = new MeshScene3D(curr_meshfilename); 
	         }
	         //
	         for (int curr_frame = 0; curr_frame < 1; curr_frame++) {
	             // save 
	             char buff2[100];
	             sprintf(buff2, "%08d_%s_fl%03d_rm%04d_%04d",sceneId_topick, Scene.sceneId.c_str(), Scene.floorId,Scene.roomId,curr_frame);
	             std::string  fileDepth = fileOutPath + buff2 + ".png";
	             std::string  fileLabel = fileOutPath + buff2 + "_label.png";
	             if (!FileExists(fileDepth)||!FileExists(fileLabel)){
	             	getProjectionMatrix(P, camK, extrinsics[curr_frame]);
	             	renderMesh(curr_meshScene->meshes, P, im_w, im_h, label_data, depth_data);
	             	std::vector<int> compression_params;
					compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
					compression_params.push_back(9);
		            cv::Mat imageDepth = convertDepthToPNG(depth_data,im_h,im_w);
		            cv::imwrite( fileDepth, imageDepth,compression_params );

		            cv::Mat imageLabel = convertLabelToPNG(label_data,im_h,im_w);
		            cv::imwrite( fileLabel, imageLabel, compression_params );
		            std::cout<<fileLabel<<std::endl;
		            std::cout<<fileDepth<<std::endl;
	             }
	         }

		}
		free(depth_data);
		free(label_data);		

	}
	else{

		// render single depth and label
		// ./renderDepth  curr_meshfilename outfile cameraPose[0] ....  cameraPose[12]

		std::string file_in = argv[1];
		std::string curr_meshfilename = file_in + ".obj";

		std::string outfile = argv[2];
		float cameraPose[12]; //int res1 = fscanf(fp, "%f%f%f%f%f%f%f%f%f%f%f%f", &vx, &vy, &vz, &tx, &ty, &tz, &ux, &uy, &uz, &rx, &ry, &rz);
		for (int i = 0; i < 12; i++){
			cameraPose[i] = atof(argv[i+3]);
		}

		std::vector<float> extrinsics; //todo
        extrinsics.resize(16);
        extrinsics[0]  = cameraPose[9];  extrinsics[1]  = -cameraPose[6]; extrinsics[2]  = cameraPose[3];  extrinsics[3]  = cameraPose[0];
        extrinsics[4]  = cameraPose[10]; extrinsics[5]  = -cameraPose[7]; extrinsics[6]  = cameraPose[4];  extrinsics[7]  = cameraPose[1];
        extrinsics[8]  = cameraPose[11]; extrinsics[9]  = -cameraPose[8]; extrinsics[10] = cameraPose[5];  extrinsics[11] = cameraPose[2];
	    extrinsics[12] = 0 ; extrinsics[13] = 0;   extrinsics[14] = 0;  extrinsics[15] = 1;
	    
	    

		MeshScene3D * curr_meshScene = new MeshScene3D(curr_meshfilename);

		float* depth_data = (float *) malloc(sizeof(float)*im_w * im_h);
    	unsigned int* label_data = (unsigned int*) malloc(sizeof(unsigned int)*im_w*im_h);
		 
		getProjectionMatrix(P,camK, extrinsics);
	    renderMesh(curr_meshScene->meshes, P, im_w, im_h, label_data, depth_data);

	   	std::string depthfilename = outfile+".depth";
	   	std::string labelfilename = outfile+".label";

	   	// save
	   	FILE *  pFile = fopen (depthfilename.c_str(), "wb");
		fwrite (depth_data , sizeof(float), im_w * im_h, pFile);
		fclose (pFile);

		pFile = fopen (labelfilename.c_str(), "wb");
		fwrite (label_data , sizeof(unsigned int), im_w * im_h, pFile);
		fclose (pFile);

		std::cout<<"Done render"<<std::endl;
	    free(depth_data);
		free(label_data);
	}
	

		

	

	
}
