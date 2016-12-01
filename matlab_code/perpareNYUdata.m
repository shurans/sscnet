function perpareNYUdata(test)
initPath();
load('cls_color.mat');
load('ClassMapping.mat');
cls_color = [cls_color;[0,1,0.3]];
savemat =1;
dataRootfolder = '/n/fs/suncg/voxelLabelComplete/';


if test
    load('test_NYUv2images.mat')
    allSeq = testSeq;
    outFolder = fullfile(dataRootfolder,'data','/NYUtest/');
else
    load('train_NYUv2images.mat')
    allSeq = trainSeq;
    outFolder = fullfile(dataRootfolder,'data','/NYUtrain/');
end

if savemat
    if test
       outputmatPath = fullfile(dataRootfolder,'mat','NYUtest');
    else
       outputmatPath = fullfile(dataRootfolder,'mat','NYUtrain');
    end
    mkdir(outputmatPath);
end

mkdir(outFolder);
allannote = dir('/n/fs/modelnet/SUNCG/code/voxelComplete/voxlet/CAD_3D/mat/');
allannote = allannote(4:end);
for i = 1:length(allannote)
    ind = find(allannote(i).name == '_');
    count(i) = str2double(allannote(i).name(1:ind(1)-1));
end

% for i = 1:length(allannote)
%     system(['/usr/local/bin/wget http://aqua.cs.uiuc.edu/processed_data/' allannote(i).name])
% end


volume_param;

for i =1:length(allSeq)
    % get Id 
    Id = str2double(allSeq{i}(length('/n/fs/sun3d/data/NYUv2images/NYU')+1:end-1));
    % put the depth map into folder  
    depthFilename = fullfile(outFolder,sprintf('NYU%04d_0000.png',Id));
    HHAFilename   = [depthFilename(1:(end-4)),'_hha.png'];
    LabelFilename = [depthFilename(1:(end-4)),'_label.png'];
    if savemat
        sceneVoxFilename_mat = fullfile(outputmatPath,sprintf('NYU%04d_0000_gt.mat',Id));
    end 
     
    sceneVoxFilename = [depthFilename(1:(end-4)),'.bin'];
    
    fprintf('%s\n',depthFilename);
    
    %/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001/depth/NYU0001.png
    depthfilename =  sprintf('/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU%04d/fullres/NYU%04d.png',Id,Id);
    depthInpaint = readDepth(depthfilename);
    saveDepth (depthInpaint,depthFilename);

    HHA = Depth2HHA( depthInpaint, camK);
    imwrite(HHA, HHAFilename);
    
    
    
    load(sprintf('/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU%04d/seg',Id));
    [~,mapNYU894To40Ids] = ismember(mapNYU894To40,nyu40class);
    [~,mapNYU40to36Ids] = ismember(mapNYU40to36,p5d36class);
    
    mapNYU894To40Ids =[0,mapNYU894To40Ids];
    mapNYU40to36Ids =[0,mapNYU40to36Ids];
    
    segcrop = mapNYU40to36Ids(mapNYU894To40Ids(seglabel+1)+1);%segmentation_class_map(cls_mapping(seglabel+1)+1);
    seg = zeros(480,640);
    seg(36:36+size(segcrop,1)-1,32:32+size(segcrop,2)-1) = segcrop;
    imwrite(uint8(seg),LabelFilename);
    
    
    % caculate transform to make floor be [0,0,0]
    matdata_name = allannote(find(count ==Id)).name;
    load(fullfile('/n/fs/modelnet/SUNCG/code/voxelComplete/voxlet/CAD_3D/mat/',matdata_name))
    floorId = 0;
    for objid = 1:length(model.objects)
        if strcmp(model.objects{objid}.model.label,'floor');
            floorId = objid;
            break;
        end
    end
    floorHeight = model.objects{floorId}.model.surfaces{1}.polygon.pts{1}.y;
    % change it from y up to z up
    transform = [0,0,-1*floorHeight];
    extCam2World = [[1 0 0; 0 -1 0;0 0 1]*[1 0 0; 0 0 1; 0 1 0]*model.camera.R'*[1 0 0; 0 -1 0;0 0 -1] transform'];
    
    %{
    point_t = extCam2World(1:3,1:3)*points3d' + repmat(extCam2World(1:3,4)',[length(points3d),1])';
    vis_point_cloud(point_t'); xlabel('x');ylabel('y');zlabel('z');
    %}
    
    voxRangeExtremesCam = [[-voxSizeCam(1:2).*voxUnit/2;0],[-voxSizeCam(1:2).*voxUnit/2;2]+voxSizeCam.*voxUnit];
    voxOriginCam = mean(voxRangeExtremesCam,2);
    
    
    camPoseArr = [extCam2World', [0;0;0;1]];  
    camPoseArr = camPoseArr(:);
    
    
    % genreating volumne
    
    
    % get world voxels preject to depth map and label to get voxel labels
    voxOriginWorld = extCam2World(1:3,1:3)*voxOriginCam + extCam2World(1:3,4) - [voxSize(1)/2*voxUnit;voxSize(2)/2*voxUnit;voxSize(3)/2*voxUnit];
    voxOriginWorld(3) = height_belowfloor;% -0.2;
    [gridPtsWorldX,gridPtsWorldY,gridPtsWorldZ] = ndgrid(voxOriginWorld(1):voxUnit:(voxOriginWorld(1)+(voxSize(1)-1)*voxUnit), ...
                                                         voxOriginWorld(2):voxUnit:(voxOriginWorld(2)+(voxSize(2)-1)*voxUnit), ...
                                                         voxOriginWorld(3):voxUnit:(voxOriginWorld(3)+(voxSize(3)-1)*voxUnit));
    gridPtsWorld = [gridPtsWorldX(:),gridPtsWorldY(:),gridPtsWorldZ(:)]';
    
    
    extWorld2Cam = inv([extCam2World;[0,0,0,1]]);
    gridPtsCam = extWorld2Cam(1:3,1:3)*gridPtsWorld + repmat(extWorld2Cam(1:3,4),1,size(gridPtsWorld,2));
    gridPtsPixX = round(gridPtsCam(1,:).*(camK(1,1))./gridPtsCam(3,:)+camK(1,3));
    gridPtsPixY = round(gridPtsCam(2,:).*(camK(2,2))./gridPtsCam(3,:)+camK(2,3));
    
    invalidPixInd = gridPtsPixX <= 1 | gridPtsPixX >= size(depthInpaint,2) | ...
                    gridPtsPixY <= 1 | gridPtsPixY >= size(depthInpaint,1);
    
    [rgb,points3d]=read_3d_pts_general(depthInpaint,camK,size(depthInpaint),[]);
    depthRead = points3d(sub2ind(size(depthInpaint),gridPtsPixY(~invalidPixInd),gridPtsPixX(~invalidPixInd)),:);
    
    dis = depthRead - [gridPtsCam(:,~invalidPixInd)]';
    distosurface = sqrt(sum(dis.*dis,2));
    onSurface = distosurface<=voxUnit*voxMargin;
    
    validVox = find(~invalidPixInd);
    validVox = validVox(onSurface);
    
   
    
    gridPtsLabel = zeros(1,size(gridPtsWorld,2));
    gridPtsLabel(validVox) = seg(sub2ind(size(seg),gridPtsPixY(validVox),gridPtsPixX(validVox)));
    %cls_color = [0,0,0;0,1,0;1,0,0;0,0,1;rand(35-3,3)];
    %scatter3(gridPtsWorld(1,gridPtsLabel>0),gridPtsWorld(2,gridPtsLabel>0),gridPtsWorld(3,gridPtsLabel>0), ones(1,sum(gridPtsLabel>0)),cls_color(gridPtsLabel(gridPtsLabel>0)+1,:));
                 
    labelGridPts = gridPtsWorld(:,find(gridPtsLabel > 0));
    pcwrite(pointCloud(labelGridPts','Color',cls_color(gridPtsLabel(find(gridPtsLabel > 0))+1,:)),fullfile(outFolder,sprintf('NYU%04d.ply',Id)),'PLYFormat','binary');

     
    extSwap = [0,1,0;0,0,1;1,0,0];
    [gridPtsX,gridPtsY,gridPtsZ] = ind2sub(voxSize,1:size(gridPtsLabel,2));
    gridPts = [gridPtsX(:),gridPtsY(:),gridPtsZ(:)]';
    gridPts = extSwap(1:3,1:3) * gridPts;
    gridPtsLabel(sub2ind(voxSizeTarget,gridPts(1,:)',gridPts(2,:)',gridPts(3,:)')) = gridPtsLabel;
    % Save the volume
    sceneVox = reshape(gridPtsLabel,voxSizeTarget');
    
    if savemat
       fprintf('sceneVoxFilename_mat: %s\n',sceneVoxFilename_mat);
       camPose = extCam2World;
       save(sceneVoxFilename_mat,'sceneVox','camPose','voxOriginWorld','-v7.3'); 
       sceneVox_ds = downSample(sceneVox,4);
       save(fullfile(outputmatPath,sprintf('NYU%04d_0000_gt_d4.mat',Id)),'sceneVox_ds','camPose','voxOriginWorld','-v7.3'); 
    end 
        
    % Save vox origin in world coordinates as first three values
    sceneVoxArr = sceneVox(:);
    sceneVoxGrad = sceneVoxArr(2:end)-sceneVoxArr(1:(end-1));
    sceneVoxPeaks = find(abs(sceneVoxGrad) > 0);
    sceneVoxRLE = [sceneVox(sceneVoxPeaks(2:end))';(sceneVoxPeaks(2:end)-sceneVoxPeaks(1:(end-1)))'];
    sceneVoxRLE = [sceneVox(sceneVoxPeaks(1)),sceneVoxPeaks(1), ...
                   sceneVoxRLE(:)', ...
                   sceneVoxArr(sceneVoxPeaks(end)+1),length(sceneVoxArr)-sceneVoxPeaks(end)];

    % write camera to world transformation 
    
    fileID = fopen(sceneVoxFilename,'w');
    fwrite(fileID,single(voxOriginWorld),'single');
    fwrite(fileID,single(camPoseArr),'single');
    fwrite(fileID,uint32(sceneVoxRLE),'uint32');
    fclose(fileID);
end