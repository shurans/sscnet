function [sceneVox, voxOriginWorld] = getSceneVoxSUNCG(pathToData,sceneId,floorId,roomId,extCam2World)
% Notes: grid is Z up while the The loaded houses are y up 
%{
pathToData = '/n/fs/suncg/data/planner5d/';
sceneId = '000514ade3bcc292a613a4c2755a5050';
floorId = 1;
roomId = 1;
cameraPose = [43.9162 1.64774 50.0449  0.0417627 -0.196116 -0.979691  0.00835255 0.980581 -0.195938  0.55 0.430998  17.8815];
extCam2World = camPose2Extrinsics(cameraPose);
extCam2World = [[1 0 0; 0 0 1; 0 1 0]*extCam2World(1:3,1:3)*[1 0 0; 0 0 1; 0 1 0] extCam2World([1,3,2],4)];
%}

load('suncgObjcategory.mat')
volume_param;
% Compute voxel range in world coordinates
voxRangeExtremesCam = [[-voxSizeCam(1:2).*voxUnit/2;0],[-voxSizeCam(1:2).*voxUnit/2;2]+voxSizeCam.*voxUnit];
voxOriginCam = mean(voxRangeExtremesCam,2);

% Compute voxel grid centers in world coordinates
voxOriginWorld = extCam2World(1:3,1:3)*voxOriginCam + extCam2World(1:3,4) - [voxSize(1)/2*voxUnit;voxSize(2)/2*voxUnit;voxSize(3)/2*voxUnit];
voxOriginWorld(3) = height_belowfloor;
[gridPtsWorldX,gridPtsWorldY,gridPtsWorldZ] = ndgrid(voxOriginWorld(1):voxUnit:(voxOriginWorld(1)+(voxSize(1)-1)*voxUnit), ...
                                                     voxOriginWorld(2):voxUnit:(voxOriginWorld(2)+(voxSize(2)-1)*voxUnit), ...
                                                     voxOriginWorld(3):voxUnit:(voxOriginWorld(3)+(voxSize(3)-1)*voxUnit));
gridPtsWorld = [gridPtsWorldX(:),gridPtsWorldY(:),gridPtsWorldZ(:)]';
gridPtsLabel = zeros(1,size(gridPtsWorld,2));

house = loadjson(fullfile(pathToData,'house', sceneId,'house.json'));
roomStruct = house.levels{floorId}.nodes{roomId};
floorStruct = house.levels{floorId};

% find all grid in the room 
floorObj = read_wobj([fullfile(pathToData,'room',sceneId,roomStruct.modelId) 'f.obj']);
inRoom = zeros(size(gridPtsWorldX));
for i = 1:length(floorObj.objects(3).data.vertices)
    faceId = floorObj.objects(3).data.vertices(i,:);
    floorP = floorObj.vertices(faceId,[1,3])';
    inRoom = inRoom|inpolygon(gridPtsWorldX,gridPtsWorldY,floorP(1,:),floorP(2,:));
end

% find floor 
floorZ = mean(floorObj.vertices(:,2));
gridPtsObjWorldInd = inRoom(:)'&(abs(gridPtsWorld(3,:)-floorZ) <= voxUnit/2);
[~,classRootId] = getobjclassSUNCG('floor',objcategory);
gridPtsLabel(gridPtsObjWorldInd) = classRootId;  

% find ceiling 
ceilObj = read_wobj([fullfile(pathToData,'room',sceneId,roomStruct.modelId) 'c.obj']);
ceilZ = mean(ceilObj.vertices(:,2));
gridPtsObjWorldInd = inRoom(:)'&abs(gridPtsWorld(3,:)-ceilZ) <= voxUnit/2;
[~,classRootId] = getobjclassSUNCG('ceiling',objcategory);
gridPtsLabel(gridPtsObjWorldInd) = classRootId;  

% Load walls
WallObj = read_wobj([fullfile(pathToData,'room',sceneId,roomStruct.modelId) 'w.obj']);
inWall = zeros(size(gridPtsWorldX));
for oi = 1:length(WallObj.objects)
    if WallObj.objects(oi).type == 'f'
        for i = 1:length(WallObj.objects(oi).data.vertices)
            faceId = WallObj.objects(oi).data.vertices(i,:);
            floorP = WallObj.vertices(faceId,[1,3])';
            inWall = inWall|inpolygon(gridPtsWorldX,gridPtsWorldY,floorP(1,:),floorP(2,:));
        end
    end
end
gridPtsObjWorldInd = inWall(:)'&(gridPtsWorld(3,:)<ceilZ-voxUnit/2)&(gridPtsWorld(3,:)>floorZ+voxUnit/2);
[~,classRootId] = getobjclassSUNCG('wall',objcategory);
gridPtsLabel(gridPtsObjWorldInd) = classRootId;     



% Loop through each object and set voxels to class ID
for objId = roomStruct.nodeIndices
    object_struct = floorStruct.nodes{objId+1};
    if isfield(object_struct, 'modelId')
        % Set segmentation class ID
        [classRootName,classRootId] = getobjclassSUNCG(strrep(object_struct.modelId,'/','__'),objcategory);

        % Compute object bbox in world coordinates
        objBbox = [object_struct.bbox.min([1,3,2])',object_struct.bbox.max([1,3,2])'];

        % Load segmentation of object in object coordinates
        filename= fullfile(pathToData,'object_vox/object_vox_data/',strrep(object_struct.modelId,'/','__'), [strrep(object_struct.modelId,'/','__'), '.binvox']);
        [voxels,scale,translate] = read_binvox(filename);
        [x,y,z] = ind2sub(size(voxels),find(voxels(:)>0));   
        objSegPts = bsxfun(@plus,[x,y,z]*scale,translate');

        % Convert object to world coordinates
        extObj2World_yup = reshape(object_struct.transform,[4,4]);
        objSegPts = extObj2World_yup*[objSegPts(:,[1,3,2])';ones(1,size(x,1))];
        objSegPts = objSegPts([1,3,2],:);

        % Get all grid points within the object bbox in world coordinates
        gridPtsObjWorldInd =      gridPtsWorld(1,:) >= objBbox(1,1) - voxUnit & gridPtsWorld(1,:) <= objBbox(1,2) + voxUnit & ...
                                  gridPtsWorld(2,:) >= objBbox(2,1) - voxUnit & gridPtsWorld(2,:) <= objBbox(2,2) + voxUnit & ...
                                  gridPtsWorld(3,:) >= objBbox(3,1) - voxUnit & gridPtsWorld(3,:) <= objBbox(3,2) + voxUnit;
        gridPtsObjWorld = gridPtsWorld(:,find(gridPtsObjWorldInd));


        % If object is a window or door, clear voxels in object bbox
        [~,wallId] = getobjclassSUNCG('wall',objcategory); 
        if classRootId == 4 || classRootId == 5
           gridPtsObjClearInd = gridPtsObjWorldInd&gridPtsLabel==wallId;
           gridPtsLabel(gridPtsObjClearInd) = 0;
        end

        % Apply segmentation to grid points of object
        [indices, dists] = multiQueryKNNSearchImpl(pointCloud(objSegPts'), gridPtsObjWorld',1);
        objOccInd = find(sqrt(dists) <= (sqrt(3)/2)*scale);
        gridPtsObjWorldLinearIdx = find(gridPtsObjWorldInd);
        gridPtsLabel(gridPtsObjWorldLinearIdx(objOccInd)) = classRootId;
    end
end

% Remove grid points not in field of view
extWorld2Cam = inv([extCam2World;[0,0,0,1]]);
gridPtsCam = extWorld2Cam(1:3,1:3)*gridPtsWorld + repmat(extWorld2Cam(1:3,4),1,size(gridPtsWorld,2));
gridPtsPixX = gridPtsCam(1,:).*(camK(1,1))./gridPtsCam(3,:)+camK(1,3);
gridPtsPixY = gridPtsCam(2,:).*(camK(2,2))./gridPtsCam(3,:)+camK(2,3);
invalidPixInd = (gridPtsPixX < 0 | gridPtsPixX >= 640 | gridPtsPixY < 0 | gridPtsPixY >= 480);
gridPtsLabel(find(invalidPixInd)) = 0;

% Remove grid points not in the room
gridPtsLabel(~inRoom(:)&gridPtsLabel(:)==0) = 255;

% Change coordinate axes XYZ -> YZX
extSwap = [0,1,0;0,0,1;1,0,0];
[gridPtsX,gridPtsY,gridPtsZ] = ind2sub(voxSize,1:size(gridPtsLabel,2));
gridPts = [gridPtsX(:),gridPtsY(:),gridPtsZ(:)]';
gridPts = extSwap(1:3,1:3) * gridPts;
gridPtsLabel(sub2ind(voxSizeTarget,gridPts(1,:)',gridPts(2,:)',gridPts(3,:)')) = gridPtsLabel;

% Save the volume
sceneVox = reshape(gridPtsLabel,voxSizeTarget');
