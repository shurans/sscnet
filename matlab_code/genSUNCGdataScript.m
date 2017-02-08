function genSUNCGdataScript(sceneId)
    % Example script of generating volumetric grondtruth for SUNCG scenes
    % Parameters
    suncgDataPath = '/n/fs/suncg/planner5d/';
    suncgToolboxPath =' ~/Documents/SceneVox/SUNCGtoolbox/';
    outputdir = '/tmp';
    usemeasa =''; % to use mesa change to: usemeasa = '-mesa' 
    
    if ~esixt('sceneId','var')
       sceneId = '000514ade3bcc292a613a4c2755a5050'; 
    end
    addpath('./utils/')
    pathtogaps = fullfile(suncgToolboxPath, '/gaps/bin/x86_64');
    %% generating camera pose 
    camerafile = sprintf('/%s/%s.cam',outputdir, sceneId);
    cameraInfofile = sprintf('/%s/%s.caminfo',outputdir, sceneId);
    projectpath = fullfile(suncgDataPath,'house',sceneId);
    cmd_gencamera = sprintf('unset LD_LIBRARY_PATH;\n cd  %s \n %s/scn2cam house.json %s -create_room_cameras -output_camera_names %s -eye_height_radius 0.25 -eye_height 1.5 -xfov 0.55 -v %s',...
                            projectpath, pathtogaps, camerafile, cameraInfofile, sceneId, usemeasa);
    system(cmd_gencamera);
    
    %% generating depth images
    output_image_directory = fullfile(outputdir,sceneId, 'images');
    mkdir(output_image_directory);
    cmd_gendepth = sprintf('unset LD_LIBRARY_PATH;\n cd  %s \n %s/scn2img house.json %s/%s.cam -capture_depth_images -capture_color_images -xfov 0.55 -v %s %s',...
                            projectpath,pathtogaps,cameradir,sceneId, output_image_directory, usemeasa);
    system(cmd_gendepth);
    
    %% generating scene voxels in camera view 
    cameraInfo = readCameraName(cameraInfofile);
    cameraPoses = readCameraPose(camerafile);
    voxPath = fullfile(outputdir,sceneId, 'depthVox');
    mkdir(voxPath);
    
    for cameraId = 1:length(cameraInfo)
        
        depthFilename = fullfile(voxPath,sprintf('%08d_%s_fl%03d_rm%04d_0000.png',cameraId-1,sceneId,cameraInfo(cameraId).floorId,cameraInfo(cameraId).roomId));
        sceneVoxFilename = [depthFilename(1:(end-4)),'.bin'];
        
        %% get camera extrisic yup -> zup
        extCam2World = camPose2Extrinsics(cameraPoses(cameraId,:));
        extCam2World = [[1 0 0; 0 0 1; 0 1 0]*extCam2World(1:3,1:3) extCam2World([1,3,2],4)];
        
        %% generating scene voxels in camera view 
        [sceneVox, voxOriginWorld] = getSceneVoxSUNCG(pathToData,sceneId,cameraInfo(cameraId).floorId+1,cameraInfo(cameraId).roomId+1,extCam2World);
        camPoseArr = [extCam2World',[0;0;0;1]];
        camPoseArr = camPoseArr(:);
        
        % Compress with RLE and save to binary file 
        writeRLEfile(sceneVoxFilename, sceneVox,camPoseArr,voxOriginWorld)
        
        % resave depth map with bit shifting
        depthRaw = double(imread(sprintf('%s/%06d_depth.png',output_image_directory,cameraId-1)))/1000;
        saveDepth (depthRaw,depthFilename);
     
    end
end

function cameraInfo = readCameraName(cameraInfofile)
cameraInfo =[];
fid = fopen(cameraInfofile,'r');
tline = fgets(fid);
cnt = 1;
while ischar(tline)
    %Room#0_0_3
   parseline = sscanf(tline, 'Room#%d_%d_%d');
   if isempty(parseline)
       break
   end
   cameraInfo(cnt).floorId = parseline(1);
   cameraInfo(cnt).roomId  = parseline(2);
   cnt = cnt+1;
   tline = fgets(fid);
end
fclose(fid);
end

function cameraPoses = readCameraPose(camereafile)
cameraPoses =[];
fid = fopen(camereafile,'r');
tline = fgets(fid);
while ischar(tline)
    parseline = sscanf(tline, '%f');
    cameraPoses = [cameraPoses;parseline'];
    tline = fgets(fid);
end
fclose(fid);
end