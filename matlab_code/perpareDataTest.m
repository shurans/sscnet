function perpareDataTest()
    % This function provides an example of generating your own testing
    % data without ground truth labels. It will generate a the .bin file
    % with camera pose and an empty volume, without room boundary.
    % To generate training data with NYU dataset and SUNCG dataset with
    % ground truth label, please refer to:
    %   perpareNYUdata.m
    %   getSceneVoxSUNCG.m
    addpath('./utils'); 
    volume_param;
    depthFilename = fullfile('../demo/NYU0001_0000.png');
    sceneVoxFilename = fullfile('../test/NYU0001_0000.bin');
    
    depthInpaint  = readDepth(depthFilename);
    [rgb,XYZ] = read_3d_pts_general(depthInpaint,camK,size(depthInpaint),[]);

    % Find room orientation
    [Rtilt,R] = rectify(XYZ');
    % Find the floor
    point_t = R*XYZ';
    
    floorHeight = prctile(point_t(:,2),2);
    
    % transformation
    transform = [0,0,-1*floorHeight];
    extCam2World = [[1 0 0; 0 -1 0;0 0 1]*[1 0 0; 0 0 1; 0 1 0]*R'*[1 0 0; 0 -1 0;0 0 -1] transform'];
    
    %{
    point_t = extCam2World(1:3,1:3)*XYZ' + repmat(extCam2World(1:3,4)',[length(points3d),1])';
    vis_point_cloud(point_t'); xlabel('x');ylabel('y');zlabel('z');
    %}

    % write out volumne info with camara pose and empty ground truth vol
    sceneVox = zeros(voxSizeTarget');
    camPoseArr = [extCam2World', [0;0;0;1]];  
    camPoseArr = camPoseArr(:);
    
    voxRangeExtremesCam = [[-voxSizeCam(1:2).*voxUnit/2;0],[-voxSizeCam(1:2).*voxUnit/2;2]+voxSizeCam.*voxUnit];
    voxOriginCam = mean(voxRangeExtremesCam,2);
    voxOriginWorld = extCam2World(1:3,1:3)*voxOriginCam + extCam2World(1:3,4) - [voxSize(1)/2*voxUnit;voxSize(2)/2*voxUnit;voxSize(3)/2*voxUnit];
    voxOriginWorld(3) = height_belowfloor;
    
    writeRLEfile(sceneVoxFilename, sceneVox,camPoseArr,voxOriginWorld)
end