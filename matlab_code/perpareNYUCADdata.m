% perpareNYUCADdata
function perpareNYUCADdata(cluster_Idx)
% cd /n/fs/modelnet/SUNCG/code/voxelComplete/matlab
% /n/fs/vision/ionicNew/starter.sh perpareNYUCADdata 4000mb 100:00:00 1 500 1 

initPath();
load('cls_color.mat');
load('ClassMapping.mat');
cls_color = [cls_color;[0,1,0.3]];
[~,mapNYU894To40Ids] = ismember(mapNYU894To40,nyu40class);
[~,mapNYU40to36Ids] = ismember(mapNYU40to36,p5d36class);

mapNYU894To40Ids =[0,mapNYU894To40Ids];
mapNYU40to36Ids =[0,mapNYU40to36Ids];
dataRootfolder = '/n/fs/suncg/voxelLabelComplete/data';
savemat = 1; 
test = 1;
if exist('cluster_Idx','var')
    gap = 500;
else
    gap = 1;
    cluster_Idx =1;
end
if test
    load('test_NYUv2images.mat')
    allSeq = testSeq;
    outFolder = fullfile(dataRootfolder,'/NYUCADtest/');
    mkdir(outFolder);
else
    load('train_NYUv2images.mat')
    allSeq = trainSeq;
    outFolder = fullfile(dataRootfolder,'/NYUCADtrain/');
    mkdir(outFolder);
end
if savemat
    if test
       outputmatPath = fullfile('/n/fs/suncg/voxelLabelComplete/','mat','NYUCADtest');
    else
       outputmatPath = fullfile('/n/fs/suncg/voxelLabelComplete/','mat','NYUCADtrain');
    end
    mkdir(outputmatPath);
end


allannote = dir('/n/fs/modelnet/SUNCG/code/voxelComplete/voxlet/CAD_3D/mat/');
allannote = allannote(4:end);
for i = 1:length(allannote)
    ind = find(allannote(i).name == '_');
    count(i) = str2double(allannote(i).name(1:ind(1)-1));
end


volume_param;

% load 
for i = cluster_Idx:gap:length(allSeq)
    % get Id 
    Id = str2double(allSeq{i}(length('/n/fs/sun3d/data/NYUv2images/NYU')+1:end-1)); 
    depthFilename = fullfile(outFolder,sprintf('NYU%04d_0000.png',Id));
    sceneVoxFilename_mat = fullfile(outputmatPath,sprintf('NYU%04d_0000_gt.mat',Id));
    sceneVoxFilename = [depthFilename(1:(end-4)),'.bin'];
    objFilename = fullfile(outFolder,sprintf('NYU%04d_0000',Id));
    
    
    HHAFilename = [depthFilename(1:(end-4)),'_hha.png'];
    segFilename = [depthFilename(1:(end-4)),'_seg.png'];
    
    matdata_name = allannote(find(count ==Id)).name;
    %depthfilename = sprintf('/n/fs/modelnet/SUNCG/code/voxelComplete/voxlet/renders_for_release/%s/images/depth8.png',matdata_name(1:end-4));
        
    if exist(sceneVoxFilename,'file')%(exist(sceneVoxFilename,'file')&&exist(HHAFilename,'file'))||~exist(depthfilename,'file')
        fprintf('skipping %s\n',sceneVoxFilename);
        continue;
    else
        % save voxel label
        load(fullfile('/n/fs/modelnet/SUNCG/code/voxelComplete/voxlet/CAD_3D/mat/',matdata_name))

        floorId = 0;
        for objid = 1:length(model.objects)
            if strcmp(model.objects{objid}.model.label,'floor');
                floorId = objid;
                break;
            end
        end
        floorHeight = model.objects{floorId}.model.surfaces{1}.polygon.pts{1}.y;

        transform = [0,0,-1*floorHeight];

        extCam2World = [[1 0 0; 0 -1 0;0 0 1]*[1 0 0; 0 0 1; 0 1 0]*model.camera.R'*[1 0 0; 0 -1 0;0 0 -1] transform'];

        voxRangeExtremesCam = [[-voxSizeCam(1:2).*voxUnit/2;0],[-voxSizeCam(1:2).*voxUnit/2;2]+voxSizeCam.*voxUnit];
        voxOriginCam = mean(voxRangeExtremesCam,2);
        camPoseArr = [extCam2World', [0;0;0;1]];  
        camPoseArr = camPoseArr(:);

        voxOriginWorld = extCam2World(1:3,1:3)*voxOriginCam + extCam2World(1:3,4) - [voxSize(1)/2*voxUnit;voxSize(2)/2*voxUnit;voxSize(3)/2*voxUnit];
        voxOriginWorld(3) = height_belowfloor;
        [gridPtsWorldX,gridPtsWorldY,gridPtsWorldZ] = ndgrid(voxOriginWorld(1):voxUnit:(voxOriginWorld(1)+(voxSize(1)-1)*voxUnit), ...
                                                             voxOriginWorld(2):voxUnit:(voxOriginWorld(2)+(voxSize(2)-1)*voxUnit), ...
                                                             voxOriginWorld(3):voxUnit:(voxOriginWorld(3)+(voxSize(3)-1)*voxUnit));
        gridPtsWorld = [gridPtsWorldX(:),gridPtsWorldY(:),gridPtsWorldZ(:)]'; 
        
        % in Room       
        inRoom = zeros(size(gridPtsWorldX));
        for sid = 1:length(model.objects{floorId}.model.surfaces) 
            floorStruct = cell2mat(model.objects{floorId}.model.surfaces{sid}.polygon.pts);
            floorP = [[floorStruct.x];-1*[floorStruct.z];[floorStruct.y]];
            inRoom = inRoom|inpolygon(gridPtsWorldX,gridPtsWorldY,floorP(1,:),floorP(2,:));
        end
        
        

        gridPtsLabel = zeros(1,size(gridPtsWorld,2));
        %{
        for i_obj = 1:length(model.objects)
            % labeling the world coordinate
            o = model.objects{i_obj};
            if isfield(o, 'mesh'),
                for i_comp = 1:length(o.mesh.comp)
                hold on; patch('vertices', o.mesh.comp{i_comp}.vertices, 'faces', o.mesh.comp{i_comp}.faces,'facecolor' , [o.mesh.app.color.x o.mesh.app.color.y o.mesh.app.color.z]);
                end
                pause
            end
        end
        %}
        
        clear meshobjArr;        
        for i_obj = 1:length(model.objects)
            % labeling the world coordinate
            o = model.objects{i_obj};
            if isfield(o, 'mesh'),
                [~,nyu894classId] = ismember(o.model.label,nyu894class);
                
                cls_mapping = mapNYU40to36Ids(mapNYU894To40Ids(nyu894classId+1)+1);
                
                
                vertices =[];
                faces =[];
                
                for i_comp = 1:length(o.mesh.comp)
                    faces = [faces;o.mesh.comp{i_comp}.faces+size(vertices,1)];
                    vertices = [vertices;o.mesh.comp{i_comp}.vertices];
                end
                vertices(:,2) = vertices(:,2) - floorHeight;
                vertices(:,[1,3,2]) = vertices(:,[1,2,3]);
                vertices(:,2) = -vertices(:,2);
                if ~isempty(faces)
                    meshobjArr(i_obj).faces = mat2cell(faces-1,ones(size(faces,1),1),3);
                else   
                    meshobjArr(i_obj).faces =[];
                end
                meshobjArr(i_obj).vertices = vertices';
                meshobjArr(i_obj).obj_name = sprintf('%d %s %d %s %d',o.model.label,'wall',0,'wall',mapNYU894To40Ids(nyu894classId+1));
                meshobjArr(i_obj).materials =[];
                
                %% get the voxel label
                if size(faces,1)>=12&&~strcmp(o.model.label,'wall')
                    fprintf('o %s,%d\n', o.model.label,cls_mapping)
                    meshObj.faces = faces;
                    meshObj.vertices = vertices;

                    objBbox = [min(meshObj.vertices)',max(meshObj.vertices)'];
                    objBbox_grid = round((objBbox-[voxOriginWorld,voxOriginWorld])/voxUnit)+1;
                    %{
                    objBbox_grid(:,1) = max(objBbox_grid(:,1),1);
                    objBbox_grid(:,2) = min([objBbox_grid(:,2)+1,size(gridPtsWorldX)'],[],2);
                    [xg,yg,zg]=ndgrid(objBbox_grid(1,1):objBbox_grid(1,2),objBbox_grid(2,1):objBbox_grid(2,2),objBbox_grid(3,1):objBbox_grid(3,2));
                    %}
                    VolumeFullsize = [objBbox_grid(1,2)-objBbox_grid(1,1)+2,objBbox_grid(2,2)-objBbox_grid(2,1)+2,objBbox_grid(3,2)-objBbox_grid(3,1)+2];
                    if ~sum((objBbox_grid(:,1)-size(gridPtsWorldX)')>0)
                        Volume=polygon2voxel(meshObj,VolumeFullsize,'au',0,0,1/voxUnit);

                        minbound = max(1,ones(3,1)-objBbox_grid(:,1)+1);
                        maxbound = max(0,objBbox_grid(:,2)+1 - size(gridPtsWorldX)');
                        Volume = Volume(minbound(1):end-maxbound(1),minbound(2):end-maxbound(2),minbound(3):end-maxbound(3));
                        
                        objBbox_grid_bound(:,1) = max(objBbox_grid(:,1),1);
                        objBbox_grid_bound(:,2) = min([objBbox_grid(:,2)+1,size(gridPtsWorldX)'],[],2);
                        [xg,yg,zg]=ndgrid(objBbox_grid_bound(1,1):objBbox_grid_bound(1,2),objBbox_grid_bound(2,1):objBbox_grid_bound(2,2),objBbox_grid_bound(3,1):objBbox_grid_bound(3,2));
                        if ~any(size(xg) == 0)
                            gridPtsObjWorldInd = sub2ind(size(gridPtsWorldX),xg,yg,zg);
                            gridPtsLabel(gridPtsObjWorldInd(find(Volume>0))) = cls_mapping;
                        end

                        %hold on; patch('vertices', vertices, 'faces', faces,'facecolor' , [o.mesh.app.color.x o.mesh.app.color.y o.mesh.app.color.z]);
                    end
                else
                    for i_comp = 1:length(o.mesh.comp)
                        faces = o.mesh.comp{i_comp}.faces;
                        if ~isempty(faces)
                            vertices = o.mesh.comp{i_comp}.vertices;
                            vertices(:,2) = vertices(:,2) - floorHeight;
                            vertices(:,[1,3,2]) = vertices(:,[1,2,3]);
                            vertices(:,2) = -vertices(:,2);
                            bottom = unique(vertices(vertices(:,3)<= (mean(vertices(:,3))+0.0001),:),'rows');
                            succ = 0;
                            try
                                ind = convhull(bottom(:,1),bottom(:,2));
                                succ =1;
                            end
                            if succ&&(max(vertices(:,3))-min(vertices(:,3))<0.2)
                               fprintf('f %s,%d\n', o.model.label,cls_mapping)
                               % box object or floor or ceiling
                               in = inpolygon(gridPtsWorld(1,:),gridPtsWorld(2,:), bottom(ind,1),bottom(ind,2));
                               in = in & gridPtsWorld(3,:) < (max(vertices(:,3)) + voxUnit) & gridPtsWorld(3,:) > (min(vertices(:,3)) - voxUnit);
                               gridPtsLabel(in) = cls_mapping;
                            else
                               fprintf('w %s,%d\n', o.model.label,cls_mapping)
                               for fid = 1:size(o.mesh.comp{i_comp}.faces,1)
                                   vid = o.mesh.comp{i_comp}.faces(fid,:);
                                   bottom = vertices(vertices(vid,3)<= (mean(vertices(vid,3))+0.0001),:);
                                   if size(bottom,1)==2
                                       d = (bottom(2,1:2)-bottom(1,1:2))/norm(bottom(2,1:2)-bottom(1,1:2));
                                       d = [d(2),-d(1)];
                                       v = [bottom(:,1:2)- voxUnit*repmat(d,[size(bottom,1),1]); bottom(end:-1:1,1:2)+voxUnit*repmat(d,[size(bottom,1),1])];
                                       in = inpolygon(gridPtsWorld(1,:),gridPtsWorld(2,:), v([1:end,1],1),v([1:end,1],2));
                                       in = in & gridPtsWorld(3,:) <= max(vertices(:,3)) + voxUnit & gridPtsWorld(3,:) >= min(vertices(:,3)) - voxUnit;
                                       gridPtsLabel(in) = cls_mapping;
                                   end
                               end
                            end
                        end
                    end                        
                end
            else
                fprintf('skip %s,%d\n', o.model.label,cls_mapping)
            end
        end
        % save mesh 
        mesh2obj(objFilename,meshobjArr,[]);
        % render Depth
        [depthInpaint,segLabel] = renderView(objFilename,extrinsics2camPose(extCam2World));
        %save('debug','depthInpaint','segLabel');

        saveDepth (depthInpaint,depthFilename);
        % save HHA
        HHA = Depth2HHA(depthInpaint, camK);
        imwrite(HHA, HHAFilename);
        
        % in FOV
        extWorld2Cam = inv([extCam2World;[0,0,0,1]]);
        gridPtsCam = extWorld2Cam(1:3,1:3)*gridPtsWorld + repmat(extWorld2Cam(1:3,4),1,size(gridPtsWorld,2));
        gridPtsPixX = round(gridPtsCam(1,:).*(camK(1,1))./gridPtsCam(3,:)+camK(1,3));
        gridPtsPixY = round(gridPtsCam(2,:).*(camK(2,2))./gridPtsCam(3,:)+camK(2,3));
        
        invalidPixInd = gridPtsPixX <= 1 | gridPtsPixX > size(depthInpaint,2) | ...
                        gridPtsPixY <= 1 | gridPtsPixY > size(depthInpaint,1);
        %{            
        validPix = find(gridPtsPixX > 1 & gridPtsPixX <= size(depthInpaint,2) & ...
                        gridPtsPixY > 1 & gridPtsPixY <= size(depthInpaint,1));
        gridPtsPixDepth = depthInpaint(sub2ind(size(depthInpaint),gridPtsPixY(validPix)',gridPtsPixX(validPix)'))';
        gridPtsPixSeg = segLabel(sub2ind(size(depthInpaint),gridPtsPixY(validPix)',gridPtsPixX(validPix)'))';
        ptDist = (gridPtsPixDepth-gridPtsCam(3,validPix)).*sqrt(1+(gridPtsCam(1,validPix)./gridPtsCam(3,validPix)).^2+(gridPtsCam(2,validPix)./gridPtsCam(3,validPix)).^2);
        outWall = validPix(gridPtsPixSeg==3&ptDist>0.1);
        gridPtsLabel(outWall) = 255;
                        
                        %}        
                   
        
        gridPtsLabel(invalidPixInd) = 0; 
        gridPtsLabel(~inRoom(:)&gridPtsLabel(:)==0) = 255;
        
        
        %% gen ply for checking 
        labelGridPts = gridPtsWorld(:,find(gridPtsLabel > 0 & gridPtsLabel < 255));
        pcwrite(pointCloud(labelGridPts','Color',cls_color(gridPtsLabel(find(gridPtsLabel > 0&gridPtsLabel < 255))+1,:)),fullfile(outFolder,sprintf('NYU%04d.ply',Id)),'PLYFormat','binary');

        extSwap = [0,1,0;0,0,1;1,0,0];
        [gridPtsX,gridPtsY,gridPtsZ] = ind2sub(voxSize,1:size(gridPtsLabel,2));
        gridPts = [gridPtsX(:),gridPtsY(:),gridPtsZ(:)]';
        gridPts = extSwap(1:3,1:3) * gridPts;
        gridPtsLabel(sub2ind(voxSizeTarget,gridPts(1,:)',gridPts(2,:)',gridPts(3,:)')) = gridPtsLabel;
        sceneVox = reshape(gridPtsLabel,voxSizeTarget');
        %clf;PATCH_3Darray(sceneVox,'col',cls_color);
        %%
       
        fprintf('sceneVoxFilename: %s\n',sceneVoxFilename);
        if savemat
           fprintf('sceneVoxFilename_mat: %s\n',sceneVoxFilename_mat);
           camPose = extCam2World;
           save(sceneVoxFilename_mat,'sceneVox','camPose','voxOriginWorld','-v7.3'); 
           sceneVox_ds = downSample(sceneVox,4);
           save(fullfile(outputmatPath,sprintf('NYU%04d_0000_gt_d4.mat',Id)),'sceneVox_ds','camPose','voxOriginWorld','-v7.3'); 
        end 
        sceneVoxArr = sceneVox(:);
        sceneVoxGrad = sceneVoxArr(2:end)-sceneVoxArr(1:(end-1));
        sceneVoxPeaks = find(abs(sceneVoxGrad) > 0);
        sceneVoxRLE = [sceneVox(sceneVoxPeaks(2:end))';(sceneVoxPeaks(2:end)-sceneVoxPeaks(1:(end-1)))'];
        sceneVoxRLE = [sceneVox(sceneVoxPeaks(1)),sceneVoxPeaks(1), ...
                        sceneVoxRLE(:)', ...
                       sceneVoxArr(sceneVoxPeaks(end)+1),length(sceneVoxArr)-sceneVoxPeaks(end)];
        fileID = fopen(sceneVoxFilename,'w');
        fwrite(fileID,single(voxOriginWorld),'single');
        fwrite(fileID,single(camPoseArr),'single');
        fwrite(fileID,uint32(sceneVoxRLE),'uint32');
        fclose(fileID);
        %}
    end
       
end