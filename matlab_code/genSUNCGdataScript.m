function genSUNCGdataScript(sceneId)
    % Example script of generating volumetric grondtruth for SUNCG scenes
    % Parameters
    suncgDataPath = '/n/fs/suncg/planner5d/';
    suncgToolboxPath =' ~/Documents/SceneVox/SUNCGtoolbox/';
    outputdir = '/tmp';
    usemeasa =''; % to use mesa change to: usemeasa = '-mesa' 
    
    if ~exist('sceneId','var')
       sceneId = '000514ade3bcc292a613a4c2755a5050'; 
    end
    addpath('./utils/')
    pathtogaps = fullfile(suncgToolboxPath, '/gaps/bin/x86_64');
    %% generating camera pose 
    camerafile = sprintf('/%s/%s.cam',outputdir, sceneId);
    cameraInfofile = sprintf('/%s/%s.caminfo',outputdir, sceneId);
    projectpath = fullfile(suncgDataPath,'house',sceneId);
    cmd_gencamera = sprintf('unset LD_LIBRARY_PATH;\n cd  %s \n %s/scn2cam house.json %s -create_room_cameras -output_camera_names %s -eye_height_radius 0.25 -eye_height 1.5 -xfov 0.55 -v %s',...
                            projectpath, pathtogaps, camerafile, cameraInfofile, usemeasa);
    system(cmd_gencamera);
    
    %% generating depth images
    output_image_directory = fullfile(outputdir,sceneId, 'images');
    mkdir(output_image_directory);
    cmd_gendepth = sprintf('unset LD_LIBRARY_PATH;\n cd  %s \n %s/scn2img house.json %s -capture_depth_images -capture_color_images -xfov 0.55 -v %s %s',...
                            projectpath,pathtogaps,camerafile, output_image_directory, usemeasa);
    system(cmd_gendepth);
    
    %% generating scene voxels in camera view 
    cameraInfo = readCameraName(cameraInfofile);
    cameraPoses = readCameraPose(camerafile);
    voxPath = fullfile(outputdir,sceneId, 'depthVox');
    mkdir(voxPath);
    
    for cameraId = 1:length(cameraInfo)
        
        depthFilename = fullfile(voxPath,sprintf('%08d_%s_fl%03d_rm%04d_0000.png',cameraId-1,sceneId,cameraInfo(cameraId).floorId,cameraInfo(cameraId).roomId));
        sceneVoxFilename = [depthFilename(1:(end-4)),'.bin'];
        sceneVoxMatFilename = [depthFilename(1:(end-4)),'.mat'];
        
        %% get camera extrisic yup -> zup
        extCam2World = camPose2Extrinsics(cameraPoses(cameraId,:));
        extCam2World = [[1 0 0; 0 0 1; 0 1 0]*extCam2World(1:3,1:3) extCam2World([1,3,2],4)];
        
        %% generating scene voxels in camera view 
        [sceneVox, voxOriginWorld] = getSceneVoxSUNCG(pathToData,sceneId,cameraInfo(cameraId).floorId+1,cameraInfo(cameraId).roomId+1,extCam2World);
        camPoseArr = [extCam2World',[0;0;0;1]];
        camPoseArr = camPoseArr(:);
        
        % Compress with RLE and save to binary file 
        writeRLEfile(sceneVoxFilename, sceneVox,camPoseArr,voxOriginWorld)
        save(sceneVoxMatFilename,'sceneVox','camPoseArr','voxOriginWorld')
        % resave depth map with bit shifting
        depthRaw = double(imread(sprintf('%s/%06d_depth.png',output_image_directory,cameraId-1)))/1000;
        saveDepth (depthRaw,depthFilename);
    end
end

%% visulizing 
%{
volume_param;
extWorld2Cam = inv([extCam2World;[0,0,0,1]]);
[gridPtsWorldX,gridPtsWorldY,gridPtsWorldZ] = ndgrid(voxOriginWorld(1):voxUnit:(voxOriginWorld(1)+(voxSize(1)-1)*voxUnit), ...
                                                     voxOriginWorld(2):voxUnit:(voxOriginWorld(2)+(voxSize(2)-1)*voxUnit), ...
                                                     voxOriginWorld(3):voxUnit:(voxOriginWorld(3)+(voxSize(3)-1)*voxUnit));

 gridPtsCam = extWorld2Cam(1:3,1:3)*[gridPtsWorldX(:),gridPtsWorldY(:),gridPtsWorldZ(:)]' + repmat(extWorld2Cam(1:3,4),1,prod(voxSize));
 % Project grid to 2D camera image frame (use 1-indexing for Matlab)
 gridPtsPixX = round((camK(1,1)*gridPtsCam(1,:)./gridPtsCam(3,:))+camK(1,3))+1;
 gridPtsPixY = round((camK(2,2)*gridPtsCam(2,:)./gridPtsCam(3,:))+camK(2,3))+1;
 depthInpaint = depthRaw;
 validDepthrane = [1,1,size(depthRaw)];
 validPix = find(gridPtsPixX > validDepthrane(2) & gridPtsPixX <= validDepthrane(4) & ...
                 gridPtsPixY > validDepthrane(1) & gridPtsPixY <= validDepthrane(3));

 outsideFOV = gridPtsPixX <= validDepthrane(2) | gridPtsPixX > validDepthrane(4) | ...
              gridPtsPixY <= validDepthrane(1) | gridPtsPixY > validDepthrane(3);

 gridPtsPixX = gridPtsPixX(validPix);
 gridPtsPixY = gridPtsPixY(validPix);
 gridPtsPixDepth = depthInpaint(sub2ind(size(depthInpaint),gridPtsPixY',gridPtsPixX'))';
 validDepth = find(gridPtsPixDepth > 0);
 missingDepth = find(gridPtsPixDepth == 0);
 gridPtsPixDepth = gridPtsPixDepth(validDepth);
 validGridPtsInd = validPix(validDepth);

 % Get depth difference
 ptDist = (gridPtsPixDepth-gridPtsCam(3,validGridPtsInd))*...
           sqrt(1+(gridPtsCam(1,validGridPtsInd)/gridPtsCam(3,validGridPtsInd)).^2+(gridPtsCam(2,validGridPtsInd)/gridPtsCam(3,validGridPtsInd)).^2);

 % Compute TSDF   
 distance = nan(voxSize(1),voxSize(2),voxSize(3));
 distance(validPix(validDepth)) = ptDist;
 distance = permute(distance,[2,3,1]);
 show_volume(abs(distance)<0.1)
 hold on;
 show_volume(sceneVox)
%}

