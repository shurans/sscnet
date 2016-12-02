/*
% install lOSMesa
% compile
% mex WarpMesh.cpp -lGLU -lOSMesa
% or
% mex WarpMesh.cpp -lGLU -lOSMesa -I/media/Data/usr/Mesa-9.1.2/include
% on mac:
% mex WarpMesh.cpp -lGLU -lOSMesa -I/opt/X11/include/ -L/opt/X11/lib/
*/
// g++ renderMesh.cpp -lGLU -lOSMesa -I/opt/X11/include/ -L/opt/X11/lib/
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath> 

#include <string.h>
#include <GL/osmesa.h>
#include <GL/glu.h>




///////////////////////////////////////////////////////////////// 3D ////////////////////////////////////////////////////////////////
struct Point3D{
  float x;
  float y;
  float z;
  Point3D(){};
  
  Point3D(float* v){
    x = v[0];
    y = v[1];
    z = v[2];
  }

  void transform(float* Rt){
    float tmpx,tmpy,tmpz;

    tmpx = Rt[0] * x + Rt[1] * y + Rt[2] * z + Rt[3];
    tmpy = Rt[4] * x + Rt[5] * y + Rt[6] * z + Rt[7];
    tmpz = Rt[8] * x + Rt[9] * y + Rt[10]* z + Rt[11];

    x = tmpx;
    y = tmpy;
    z = tmpz;
  }
};

class Mesh3D{
public:
  std::vector < Point3D > vertex;
  std::vector < std::vector< int > > face;
  unsigned int instanceId;
  unsigned int categoryId;
  unsigned int itemId;
  std::string instanceName;
  std::string categoryName;
  std::string filename_str;

  Mesh3D(){};
  Mesh3D(std::string filename){ 
    instanceId = 0;
    instanceName = "unknown";
    filename_str = filename;
    readObj(filename); 
  };
  void scaleMesh(float s){
    for (int i=0;i<vertex.size();++i){
      vertex[i].x = s*vertex[i].x;
      vertex[i].y = s*vertex[i].y;
      vertex[i].z = s*vertex[i].z;
    }
  }
  Point3D getCenter(){
    Point3D center;
    center.x = 0;
    center.y = 0;
    center.z = 0;
    for (int i=0;i<vertex.size();++i){
      center.x += vertex[i].x;
      center.y += vertex[i].y;
      center.z += vertex[i].z;
    }
    if (vertex.size()>0){   
      center.x /= float(vertex.size());
      center.y /= float(vertex.size());
      center.z /= float(vertex.size());
    }
    return center;
  };
  void translate(Point3D T){
    for (int i=0;i<vertex.size();++i){
      vertex[i].x += T.x;
      vertex[i].y += T.y;
      vertex[i].z += T.z;
    }
  };
  void zeroCenter(){
    Point3D center = getCenter();
    center.x = - center.x;
    center.y = - center.y;
    center.z = - center.z;
    translate(center);
  };

  void transform(float* Rt){
    //face.resize(1);

    for (int i=0;i<vertex.size();++i){
      float x = vertex[i].x;
      float y = vertex[i].y;
      float z = vertex[i].z;
      vertex[i].x = Rt[0] * x + Rt[1] * y + Rt[2] * z + Rt[3];
      vertex[i].y = Rt[4] * x + Rt[5] * y + Rt[6] * z + Rt[7];
      vertex[i].z = Rt[8] * x + Rt[9] * y + Rt[10]* z + Rt[11];
    }
  };

  void readOFF(const std::string filename){
    std::string readLine;
    std::ifstream fin(filename.c_str());
    getline(fin,readLine);
    if (readLine != "OFF") std::cerr << "The file to read should be OFF." << std::endl;
    int delimiterPos_1, delimiterPos_2, delimiterPos_3;

    //getline(fin,readLine);
    //cout<<"readLine[0]="<<readLine[0]<<endl;
    //cout<<"readLine[0]="<<(!(readLine[0]=='#' || readLine[0]==' ' || readLine[0]=='\n' || readLine[0]=='\r'))<<endl;

    do { getline(fin,readLine); } while((readLine[0]=='#' || readLine[0]==' ' || readLine[0]=='\n' || readLine[0]=='\r'));
    
    //cout<<"endl"<<endl;

    delimiterPos_1 = readLine.find(" ", 0);
    int nv = atoi(readLine.substr(0,delimiterPos_1+1).c_str());
    delimiterPos_2 = readLine.find(" ", delimiterPos_1);
    int nf = atoi(readLine.substr(delimiterPos_1,delimiterPos_2 +1).c_str());

    //cout<<"nv="<<nv<<endl;
    //cout<<"nf="<<nf<<endl;

    vertex.resize(nv);
    face.resize(nf);
    for (int n=0; n<nv; n++){
      do { getline(fin,readLine); } while((readLine[0]=='#' || readLine[0]==' ' || readLine[0]=='\n' || readLine[0]=='\r'));
      delimiterPos_1 = readLine.find(" ", 0);
      vertex[n].x = atof(readLine.substr(0,delimiterPos_1).c_str());
      delimiterPos_2 = readLine.find(" ", delimiterPos_1+1);
      vertex[n].y = atof(readLine.substr(delimiterPos_1,delimiterPos_2 ).c_str());
      delimiterPos_3 = readLine.find(" ", delimiterPos_2+1);
      vertex[n].z = atof(readLine.substr(delimiterPos_2,delimiterPos_3 ).c_str());
    }
    for (int n=0; n<nf; n++){
      do { getline(fin,readLine); } while((readLine[0]=='#' || readLine[0]==' ' || readLine[0]=='\n' || readLine[0]=='\r'));
      delimiterPos_1 = readLine.find(" ", 0);
      face[n].resize(atoi(readLine.substr(0,delimiterPos_1).c_str()));
      for (int i=0;i<face[n].size();++i){
        delimiterPos_2 = readLine.find(" ", delimiterPos_1+1);
        face[n][i] = atoi(readLine.substr(delimiterPos_1,delimiterPos_2).c_str());
        delimiterPos_1 = delimiterPos_2;
      }
    }
    fin.close();
  };
  void writeOFF(const std::string filename){
    std::ofstream fout(filename.c_str());
    fout<<"OFF"<<std::endl;
    fout<<vertex.size()<<" "<<face.size()<<" 0"<<std::endl;
    for(int n=0;n<vertex.size();++n)
      fout<<vertex[n].x<<" "<<vertex[n].y<<" "<<vertex[n].z<<std::endl;
    for(int n=0;n<face.size();++n){
      fout<<face[n].size();
      for (int i=0;i<face[n].size();++i) fout<<" "<<face[n][i];
      fout<<std::endl;
    }
    fout.close();
  };

  void readPLY(const std::string filename){
    std::ifstream fin(filename.c_str());
    int num_vertex, num_face; //, num_edge;
    std::string str;
    getline(fin,str); getline(fin,str); getline(fin,str);
    fin>>str>>str>>num_vertex;  getline(fin,str);
    getline(fin,str); getline(fin,str); getline(fin,str);
    fin>>str>>str>>num_face;  getline(fin,str);
    getline(fin,str);
    getline(fin,str);
    vertex.resize(num_vertex);
    for (int i=0;i<num_vertex;++i){
      fin>>vertex[i].x>>vertex[i].y>>vertex[i].z;
    }
    face.resize(num_face);
    for (int i=0;i<num_face;++i){
      int num_vertex_index;
      fin>>num_vertex_index;
      face[i].resize(num_vertex_index);
      for (int j=0;j<num_vertex_index; j++)
        fin>>face[i][j];
    }
    fin.close();
  };



  void readPLYbin(const std::string filename){
    std::ifstream fin(filename.c_str());
    int num_vertex, num_face; //, num_edge;
    std::string str;
    getline(fin,str); getline(fin,str); //getline(fin,str);
    fin>>str>>str>>num_vertex;  getline(fin,str);
    getline(fin,str); getline(fin,str); getline(fin,str);
    fin>>str>>str>>num_face;  getline(fin,str);
    getline(fin,str);
    getline(fin,str);
    vertex.resize(num_vertex);

    fin.read((char*)(&(vertex[0].x)), num_vertex*3*sizeof(float));

    face.resize(num_face);
    for (int i=0;i<num_face;++i){
      uint8_t num_vertex_index;
      fin.read((char*)(&num_vertex_index),sizeof(uint8_t));
      face[i].resize(num_vertex_index);
      fin.read((char*)(&(face[i][0])),num_vertex_index*sizeof(int));
    }
    fin.close();
  };

  void writePLY(const std::string filename){
    std::ofstream fout(filename.c_str());
    fout<<"ply"<<std::endl;
    fout<<"format ascii 1.0"<<std::endl;
    //fout<<"comment format for RenderMe"<<endl;
    fout<<"element vertex "<<vertex.size()<<std::endl;
    fout<<"property float x"<<std::endl;
    fout<<"property float y"<<std::endl;
    fout<<"property float z"<<std::endl;
    fout<<"element face "<<face.size()<<std::endl;
    fout<<"property list uchar int vertex_index"<<std::endl;
    //fout<<"element edge "<<0<<endl;
    //fout<<"property int vertex1"<<endl;
    //fout<<"property int vertex2"<<endl;
    fout<<"end_header"<<std::endl;
    for (int i=0;i<vertex.size();++i){
      fout<<vertex[i].x<<" "<<vertex[i].y<<" "<<vertex[i].z<<std::endl;
    }
    for (int i=0;i<face.size();++i){
      fout<<face[i].size();
      for (int j=0;j<face[i].size();++j){
        fout<<" "<<face[i][j];
      }
      fout<<std::endl;
    }
    fout.close();
  };

  int readObj(const std::string filename){
     // Open file
     FILE *fp = fopen(filename.c_str(), "r");
     if (!fp) {
        printf("Unable to open file %s", filename.c_str());
        return 0;
     }
     
     // Read body
       char buffer[1024];
       int line_count = 0;
       int material_index =-1;
       while (fgets(buffer, 1023, fp)) {
        // Increment line counter
        line_count++;

        // Skip white space
        char *bufferp = buffer;
        while (isspace(*bufferp)) bufferp++;

        // Skip blank lines and comments
        if (*bufferp == '#') continue;
        if (*bufferp == '\0') continue; 
        if (*bufferp == 'o') continue;

      // Get keyword
      char keyword[80];
      if (sscanf(bufferp, "%s", keyword) != 1) {
        printf("Syntax error on line %d in file %s", line_count, filename.c_str());
        return 0;
      }
      

        if (!strcmp(keyword, "v")) {
          // Read vertex coordinates
          Point3D point;
          if (sscanf(bufferp, "%s%f%f%f", keyword, &point.x, &point.z, &point.y) != 4) {
             printf("Syntax error on line %d in file %s", line_count, filename.c_str());
             return 0;
          }
          vertex.push_back(point);
        }
        else if (!strcmp(keyword, "f")) {
           // Read vertex indices
           int quad = 1;
           char s1[128], s2[128], s3[128], s4[128] = { '\0' };
           if (sscanf(bufferp, "%s%s%s%s%s", keyword, s1, s2, s3, s4) != 5) {
             quad = 0;;
             if (sscanf(bufferp, "%s%s%s%s", keyword, s1, s2, s3) != 4) {
                printf("Syntax error on line %d in file %s", line_count, filename.c_str());
                return 0;
             }
           }
           // Parse vertex indices
           int vi1 = -1, vi2 = -1, vi3 = -1, vi4 = -1;
           int ti1 = -1, ti2 = -1, ti3 = -1, ti4 = -1;
           char *p1 = strchr(s1, '/'); 
         if (p1) { *p1 = 0; vi1 = atoi(s1); p1++; if (*p1) ti1 = atoi(p1); }
         else { vi1 = atoi(s1); ti1 = vi1; }
         char *p2 = strchr(s2, '/'); 
         if (p2) { *p2 = 0; vi2 = atoi(s2); p2++; if (*p2) ti2 = atoi(p2); }
         else { vi2 = atoi(s2); ti2 = vi2; }
         char *p3 = strchr(s3, '/'); 
         if (p3) { *p3 = 0; vi3 = atoi(s3); p3++; if (*p3) ti3 = atoi(p3); }
         else { vi3 = atoi(s3); ti3 = vi3; }
         if (quad) {
        char *p4 = strchr(s4, '/'); 
        if (p4) { *p4 = 0; vi4 = atoi(s4); p4++; if (*p4) ti4 = atoi(p4); }
        else { vi4 = atoi(s4); ti4 = vi4; }
         }
         // Check vertices
           if ((vi1 == vi2) || (vi2 == vi3) || (vi1 == vi3)) continue;
           if ((quad) && ((vi4 == vi1) || (vi4 == vi2) || (vi4 == vi3))) quad = 0;
         // push to the face 
         std::vector< int > thisface;
         if (quad){
           thisface.resize(4);
           thisface[0] = vi1-1;
           thisface[1] = vi2-1;
           thisface[2] = vi3-1;
           thisface[3] = vi4-1;
         }else{
             thisface.resize(3);
           thisface[0] = vi1-1;
           thisface[1] = vi2-1;
           thisface[2] = vi3-1;
         }
         face.push_back(thisface);
        }
        else if (!strcmp(keyword, "vt")) {
          // Read texture coordinates
        }
        else if (!strcmp(keyword, "mtllib")) {

        }
        else if (!strcmp(keyword, "usemtl")) {

        }
       }
       std::cout<< face.size() <<std::endl;
       std::cout<< vertex.size() <<std::endl;
       // Close file
     fclose(fp);

     // Return success
     return 1;
  };
};

class MeshScene3D{
  public:
  std::vector<Mesh3D*> meshes;
  MeshScene3D(const std::string filename){
    readObj(filename);
  };
  ~MeshScene3D(){
    for (int i =0; i< meshes.size();i++){
      delete meshes[i];
    }
  };

  Point3D getBoxCenter(){
    Point3D center;
    float minv[3] = {1000000,1000000,1000000};
    float maxv[3] = {-1000000,-1000000,-1000000};
    for (int mid = 0; mid < meshes.size(); mid++){

      for (int i=0;i<meshes[mid]->vertex.size();++i){
        minv[0] = std::min(minv[0],meshes[mid]->vertex[i].x);
            minv[1] = std::min(minv[1],meshes[mid]->vertex[i].y);
            minv[2] = std::min(minv[2],meshes[mid]->vertex[i].z);

            maxv[0] = std::max(maxv[0],meshes[mid]->vertex[i].x);
            maxv[1] = std::max(maxv[1],meshes[mid]->vertex[i].y);
            maxv[2] = std::max(maxv[2],meshes[mid]->vertex[i].z);
      }
    }
    
    center.x = 0.5*(minv[0]+maxv[0]);
    center.y = 0.5*(minv[1]+maxv[1]);
    center.z = 0.5*(minv[2]+maxv[2]);
    
    return center;
  };

  void zeroCenter(){
    Point3D center = getBoxCenter();
    center.x = - center.x;
    center.y = - center.y;
    center.z = - center.z;
    translate(center);
  };

  void translate(Point3D T){
     for (int mid =0;mid<meshes.size();mid++){
         meshes[mid]->translate(T);
      }
  };

  int readObj(const std::string filename){
     // Open file
     FILE *fp = fopen(filename.c_str(), "r");
     if (!fp) {
        printf("Unable to open file %s", filename.c_str());
        return 0;
     }

     
     // Read body
       char buffer[1024];
       int line_count = 0;
       //int material_index =-1;
       Mesh3D * mesh;
       int prev_vertex = 0;
       while (fgets(buffer, 1023, fp)) {
        // Increment line counter
        line_count++;

        // Skip white space
        char *bufferp = buffer;
        while (isspace(*bufferp)) bufferp++;

        // Skip blank lines and comments
        if (*bufferp == '#') continue;
        if (*bufferp == '\0') continue;

      // Get keyword
      char keyword[80];
      if (sscanf(bufferp, "%s", keyword) != 1) {
        printf("Syntax error on line %d in file %s", line_count, filename.c_str());
        return 0;
      }
      
      if (!strcmp(keyword, "o")) {  
        if (meshes.size()>0){
          prev_vertex += mesh->vertex.size();
        }
        
        mesh = new Mesh3D();
        meshes.push_back(mesh);
        char instanceName[80],categoryName[80];
        sscanf(bufferp, "%s%d%s%d%s%d", keyword, &(mesh->itemId), instanceName, &(mesh->instanceId), categoryName,&(mesh->categoryId));
        mesh->instanceName.append(instanceName);
        mesh->categoryName.append(categoryName);
        //std::cout<<"mesh->itemId : " <<mesh->itemId <<" instanceName :" << mesh->instanceName <<" mesh->instanceId : " << mesh->instanceId <<std::endl;       
      }
        else if (!strcmp(keyword, "v")) {
          // Read vertex coordinates
          Point3D point;
          if (sscanf(bufferp, "%s%f%f%f", keyword, &point.x, &point.y, &point.z) != 4) {
             printf("Syntax error on line %d in file %s", line_count, filename.c_str());
             return 0;
          }
          mesh->vertex.push_back(point);
        }
        else if (!strcmp(keyword, "f")) {
           // Read vertex indices
           int quad = 1;
           char s1[128], s2[128], s3[128], s4[128] = { '\0' };
           if (sscanf(bufferp, "%s%s%s%s%s", keyword, s1, s2, s3, s4) != 5) {
             quad = 0;;
             if (sscanf(bufferp, "%s%s%s%s", keyword, s1, s2, s3) != 4) {
                printf("Syntax error on line %d in file %s", line_count, filename.c_str());
                return 0;
             }
           }
           // Parse vertex indices
           int vi1 = -1, vi2 = -1, vi3 = -1, vi4 = -1;
           int ti1 = -1, ti2 = -1, ti3 = -1, ti4 = -1;
           char *p1 = strchr(s1, '/'); 
         if (p1) { *p1 = 0; vi1 = atoi(s1); p1++; if (*p1) ti1 = atoi(p1); }
         else { vi1 = atoi(s1); ti1 = vi1; }
         char *p2 = strchr(s2, '/'); 
         if (p2) { *p2 = 0; vi2 = atoi(s2); p2++; if (*p2) ti2 = atoi(p2); }
         else { vi2 = atoi(s2); ti2 = vi2; }
         char *p3 = strchr(s3, '/'); 
         if (p3) { *p3 = 0; vi3 = atoi(s3); p3++; if (*p3) ti3 = atoi(p3); }
         else { vi3 = atoi(s3); ti3 = vi3; }
         if (quad) {
        char *p4 = strchr(s4, '/'); 
        if (p4) { *p4 = 0; vi4 = atoi(s4); p4++; if (*p4) ti4 = atoi(p4); }
        else { vi4 = atoi(s4); ti4 = vi4; }
         }
         // Check vertices
           if ((vi1 == vi2) || (vi2 == vi3) || (vi1 == vi3)) continue;
           if ((quad) && ((vi4 == vi1) || (vi4 == vi2) || (vi4 == vi3))) quad = 0;
         // push to the face 
         std::vector< int > thisface;
         if (quad){
           thisface.resize(4);
           thisface[0] = vi1-1-prev_vertex;
           thisface[1] = vi2-1-prev_vertex;
           thisface[2] = vi3-1-prev_vertex;
           thisface[3] = vi4-1-prev_vertex;
         }else{
             thisface.resize(3);
           thisface[0] = vi1-1-prev_vertex;
           thisface[1] = vi2-1-prev_vertex;
           thisface[2] = vi3-1-prev_vertex;
         }
         mesh->face.push_back(thisface);
        }
        else if (!strcmp(keyword, "vt")) {
          // Read texture coordinates
        }
        else if (!strcmp(keyword, "mtllib")) {

        }
        else if (!strcmp(keyword, "usemtl")) {

        }
       }
    
       
       // Close file
     fclose(fp);

     // Return success
     return 1;
  };

  void writeOFF(const std::string filename){
    std::ofstream fout(filename.c_str());
    fout<<"OFF"<<std::endl;
    int nv = 0; int nf =0;
    for (int mid = 0 ; mid < meshes.size(); mid++){
      nv += meshes[mid]->vertex.size();
      nf += meshes[mid]->face.size();
    }
    fout << nv << " " << nf <<" 0"<< std::endl;

    for (int mid = 0 ; mid < meshes.size();mid++){
      for(int n=0;n<meshes[mid]->vertex.size();++n){
        fout<<meshes[mid]->vertex[n].x<<" "<<meshes[mid]->vertex[n].y<<" "<<meshes[mid]->vertex[n].z<<std::endl;
      }
    }

    for (int mid = 0 ; mid < meshes.size();mid++){
      int prev_vertex = 0;
      for (int i =0; i<mid;i++){
        prev_vertex += meshes[i]->vertex.size();
      }

      for(int n=0;n<meshes[mid]->face.size();++n){
        fout<<meshes[mid]->face[n].size();
        for (int i=0;i<meshes[mid]->face[n].size();++i) {
          fout<<" "<<meshes[mid]->face[n][i]+prev_vertex;
        }
        fout<<std::endl;
      }
    }
    fout.close();
  };
};

///////////////////////////////////////////////////////////////// Render Mesh ////////////////////////////////////////////////////////////////

#define Square(x) ((x)*(x))

unsigned int uchar2uint(unsigned char* in){
  unsigned int out = (((unsigned int)(in[0])) << 16) + (((unsigned int)(in[1])) << 8) + ((unsigned int)(in[2]));
  return out;
}


void uint2uchar(unsigned int in, unsigned char* out){
  out[0] = (in & 0x00ff0000) >> 16;
  out[1] = (in & 0x0000ff00) >> 8;
  out[2] =  in & 0x000000ff;
  
  //printf("%d=>[%d,%d,%d]=>%d\n",in,out[0],out[1],out[2], uchar2uint(out));
}


// Input: 
//     arg0: 3x4 Projection matrix, 
//     arg1: image width, 
//     arg2: image height, 
//     arg3: width*height*4 double matrix, 
// Output: you will need to transpose the result in Matlab manually. see the demo




void renderMesh(std::vector<Mesh3D*> models, const float* projection, int m_width, int m_height, unsigned int* result, float* depth_render) {
  //printf("renderMesh\n"); 

  float m_near = 0.3;
  float m_far = 1e8;
  int m_level = 0;
  
  //double dis_threshold_square = Square(0.2);
  
  //printf("output size:\nm_width=%d\nm_height=%d\n", m_width,m_height);


  // Step 1: setup off-screen binding 
  OSMesaContext ctx;
  ctx = OSMesaCreateContextExt(OSMESA_BGR, 32, 0, 0, NULL ); // strange hack not sure why it is not OSMESA_RGB

  unsigned char * pbuffer = new unsigned char [3 * m_width * m_height];
  // Bind the buffer to the context and make it current
  if (!OSMesaMakeCurrent(ctx, (void*)pbuffer, GL_UNSIGNED_BYTE, m_width, m_height)) {
     printf("OSMesaMakeCurrent failed!: ");
     return;
  }
  OSMesaPixelStore(OSMESA_Y_UP, 0);
  // Step 2: Setup basic OpenGL setting
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  //glEnable(GL_CULL_FACE);//glCullFace(GL_BACK); 
  // draw both front and back facing faces
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, m_width, m_height);

  // Step 3: Set projection matrices
  float scale = (0x0001) << m_level;
  float final_matrix[16];

  // new way: faster way by reuse computation and symbolic derive. See sym_derive.m to check the math.
  float inv_width_scale  = 1.0/(m_width*scale);
  float inv_height_scale = 1.0/(m_height*scale);
  float inv_width_scale_1 =inv_width_scale - 1.0;
  float inv_height_scale_1_s = -(inv_height_scale - 1.0);
  float inv_width_scale_2 = inv_width_scale*2.0;
  float inv_height_scale_2_s = -inv_height_scale*2.0;
  float m_far_a_m_near = m_far + m_near;
  float m_far_s_m_near = m_far - m_near;
  float m_far_d_m_near = m_far_a_m_near/m_far_s_m_near;


  final_matrix[ 0]= projection[2+0*3]*inv_width_scale_1 + projection[0+0*3]*inv_width_scale_2;
  final_matrix[ 1]= projection[2+0*3]*inv_height_scale_1_s + projection[1+0*3]*inv_height_scale_2_s;
  final_matrix[ 2]= projection[2+0*3]*m_far_d_m_near;
  final_matrix[ 3]= projection[2+0*3];
  final_matrix[ 4]= projection[2+1*3]*inv_width_scale_1 + projection[0+1*3]*inv_width_scale_2;
  final_matrix[ 5]= projection[2+1*3]*inv_height_scale_1_s + projection[1+1*3]*inv_height_scale_2_s; 
  final_matrix[ 6]= projection[2+1*3]*m_far_d_m_near;    
  final_matrix[ 7]= projection[2+1*3];
  final_matrix[ 8]= projection[2+2*3]*inv_width_scale_1 + projection[0+2*3]*inv_width_scale_2; 
  final_matrix[ 9]= projection[2+2*3]*inv_height_scale_1_s + projection[1+2*3]*inv_height_scale_2_s;
  final_matrix[10]= projection[2+2*3]*m_far_d_m_near;
  final_matrix[11]= projection[2+2*3];
  final_matrix[12]= projection[2+3*3]*inv_width_scale_1 + projection[0+3*3]*inv_width_scale_2;
  final_matrix[13]= projection[2+3*3]*inv_height_scale_1_s + projection[1+3*3]*inv_height_scale_2_s;  
  final_matrix[14]= projection[2+3*3]*m_far_d_m_near - (2*m_far*m_near)/m_far_s_m_near;
  final_matrix[15]= projection[2+3*3];


  // matrix is ready. use it
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(final_matrix);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Step 3: render the mesh with encoded color from their ID
  for (int mid = 0; mid < models.size(); mid++){
      Mesh3D * model = models[mid];
      unsigned char colorBytes[3];
      uint2uchar(model->categoryId,colorBytes);
      // if (model->categoryId>36) {
      //   printf("%s : %d,",model->filename_str.c_str(),model->categoryId);
      //   exit(0);
      // }
      for (unsigned int i = 0; i < model->face.size() ; ++i) {
          glColor3ubv(colorBytes);
          glBegin(GL_POLYGON);
          for (unsigned int j=0; j < model->face[i].size(); ++j){
            int vi = model->face[i][j];
            glVertex3f(model->vertex[vi].x, model->vertex[vi].y, model->vertex[vi].z);
          }
          glEnd();
      }
  }
  glFinish(); // done rendering
  
 
  // Step 5: convert the result from color to interger array  
  unsigned char * pbufferCur = pbuffer;
  for (int j =0;j < m_height;j++){
    for (int i =0;i < m_width;i++){
        //result[(m_width-i-1)+j*m_width] = uchar2uint(pbufferCur);
        result[(m_width-i-1)+j*m_width] = uchar2uint(pbufferCur);
        pbufferCur += 3;
    }
  }
  GLint outWidth, outHeight, bitPerDepth;
  unsigned int* pDepthBuffer;
  OSMesaGetDepthBuffer(ctx, &outWidth, &outHeight, &bitPerDepth, (void**)&pDepthBuffer);
  
  // do the conversion
  float z_far = 10;
  for (int j =0;j < m_height;j++){
    for (int i =0;i < m_width;i++){
        float depth_value = (float)pDepthBuffer[j*m_width+i]/4294967296.0;
        depth_value = m_near/(1-depth_value);
        if (depth_value<m_near|depth_value>z_far){
            depth_value =0;
        }
        depth_render[(m_width-i-1)+(m_height-j-1)*m_width] = (float)depth_value;
        //depth_render[(m_width-i-1)+j*m_width] = (float)depth_value;
    }
  }
  
  OSMesaDestroyContext(ctx);
  delete [] pbuffer;
}


void getProjectionMatrix(float* P,float* K, std::vector<float> RT){
     float RTinv[12];
     RTinv[0] = RT[0]; RTinv[1] = RT[4]; RTinv[2] = RT[8];  
     RTinv[4] = RT[1]; RTinv[5] = RT[5]; RTinv[6] = RT[9];   
     RTinv[8] = RT[2]; RTinv[9] = RT[6]; RTinv[10] = RT[10]; 
     for (int i=0;i<3;i++){
        RTinv[3+i*4] = -1*(RT[0+i]*RT[3]+RT[4+i]*RT[7]+RT[8+i]*RT[11]);
     }

     //inverse first row
     for (int i=0;i<4;i++){
        RTinv[i] = -1*RTinv[i];
     } 

     for (int i =0;i<4;i++){
          P[0+i*3] = (K[0]*RTinv[0+i]+K[2]*RTinv[8+i]);
          P[1+i*3] = (K[4]*RTinv[4+i]+K[5]*RTinv[8+i]);
          P[2+i*3] = RTinv[8+i];
     }
};

// struct Camera{
//       float RT[12];
//       void print(){
//         for(int i = 0;i<3;i++){
//           for(int j = 0;j<4;j++){
//             std::cout<<RT[i*4+j]<<", ";
//           }
//           std::cout<<std::endl;
//         }
//       }
// };



int test(){
  /*
  Mesh3D* meshTrain = new Mesh3D("/Volumes/backup/SUNCG/data/planner5d/meshfiles/0045c48cbc8b5532ab0a95bac829a1c8/fl001_rm0001_ob.obj");
  meshTrain->zeroCenter();
  meshTrain->writeOFF("40.off");
  */

/*
  MeshScene3D* meshScene = new MeshScene3D("/Volumes/backup/SUNCG/data/planner5d/meshfiles/0045c48cbc8b5532ab0a95bac829a1c8/fl001_rm0004.obj");  
  std::vector< std::vector<float> > extrinsics = ReadTrajectory("/Volumes/backup/SUNCG/data/planner5d/render/trajectory/cameraRT");

  float K[9];                
  K[0] = 570.3422; K[1] = 0;        K[2] = 320;
  K[3] = 0;        K[4] = 570.3422; K[5] = 240;
  K[6] = 0;        K[7] = 0;        K[8] = 1;

  int m_width = 640;
  int m_height = 480;
  float* depth_render = (float *) malloc(sizeof(float)*m_width*m_height);
  unsigned int* result = (unsigned int*) malloc(sizeof(unsigned int)*m_width*m_height);
  
  // set up camera
  
  // for loop the camera trajectory 
  for (int i = 0; i < 20; i++){
      float P[12]; 
      getProjectionMatrix(P,K, extrinsics[i]);
      renderMesh(meshScene->meshes, P,m_width,m_height,result,depth_render);
      
      // save file
      char output_filename[1024];
      sprintf(output_filename, "./debug_output/depth_%d.bin", i);
      FILE* fp = fopen(output_filename,"wb");
      fwrite(depth_render,sizeof(float),m_width*m_height,fp);
      fclose(fp);

      sprintf(output_filename, "./debug_output/label_%d.bin", i);
      fp = fopen(output_filename,"wb");
      fwrite(result,sizeof(unsigned int),m_width*m_height,fp);
      fclose(fp);
  }
  */

 
  return 1;
}