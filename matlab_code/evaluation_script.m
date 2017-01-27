function evaluation_script(OutputPath,benchmark)
% example: 
% evluation_script('../results/','nyu')
% evluation_script('../results/','nyucad')
% evluation_script('../results/','suncg')

dataRootfolder = '../data/';
dbstop if error;
hdf5Path = OutputPath;
downscale = 4;
addpath('./utils'); addpath('./benchmark');
load('ClassMapping.mat');load('voxletTest');
%% get mapping to 11 object class: 
obj_class = {'empty','ceiling','floor','wall','window','chair','bed','sofa','table','tvs','furn','objs'};
[~,mapIds] = ismember(map36to11,elevenClass);
mapIds = [0,mapIds];
classtoEvaluate = [1:11];

if strcmp(benchmark,'suncg')
   groundtruth_path = fullfile(dataRootfolder,'eval','SUNCGtest');
   evalvol_path = groundtruth_path;
   load(fullfile(fullfile(dataRootfolder,'depthbin','SUNCGtest','camera_list_train.mat')));
   for dataIdx = 1:length(dataList)
       Filename{dataIdx} = sprintf('%08d_%s_fl%03d_rm%04d_0000',dataIdx-1,dataList(dataIdx).sceneId,dataList(dataIdx).floorId,dataList(dataIdx).roomId);
   end
   numoffiles = 470;
   result_filename  ='result_suncg.hdf5';
else
   groundtruth_path = fullfile(dataRootfolder,'eval','NYUCADtest');
   load('./benchmark/test_NYUv2Ids.mat')
   for dataIdx = 1:length(testSeqId)
       Id = testSeqId(dataIdx); 
       Filename{dataIdx} = sprintf('NYU%04d_0000',Id);
   end
   if strcmp(benchmark,'nyucad')
       evalvol_path = fullfile(dataRootfolder,'eval','NYUCADtest');
       result_filename  ='result_nyucad.hdf5';
    elseif strcmp(benchmark,'nyu')
        evalvol_path = fullfile(dataRootfolder,'eval','NYUtest');
        result_filename  ='result_nyu.hdf5';
    else
        fprintf('unkown benchmark ... \n')
    end
end

fprintf('loading files %s \n', fullfile(hdf5Path,result_filename))
predobjTensor = hdf5read(fullfile(hdf5Path,result_filename),'/result');
numoffiles = size(predobjTensor,5);


%% evaluation
fprintf('evaluation ... \n')
mkdir(fullfile(OutputPath,benchmark));
for batchId = 1:numoffiles
    % get groundtruth  
    ld = load(fullfile(groundtruth_path,[Filename{batchId} '_gt_d4.mat'])); 
    sceneVox =ld.sceneVox_ds;
    ld = load(fullfile(evalvol_path,[Filename{batchId} '_vol_d4.mat']));
    vol =ld.flipVol_ds;
    sceneVox(sceneVox==255|isnan(sceneVox)) = 0;
    labelobj = mapIds(sceneVox+1);
    % get prediction 
    predobj_conf = predobjTensor(:,:,:,:,batchId);
    [~,predobj] = max(predobj_conf,[],4);
    predobj = predobj-1;
    
    nonfree_voxels_to_evaluate = abs(vol)<1|vol==-1;
    resultsFullSeg(batchId) = evaluate_prediction(labelobj, predobj, nonfree_voxels_to_evaluate,classtoEvaluate);
    resultsFullOcc(batchId) = evaluate_completion(labelobj,predobj, predobj_conf(:,:,:,1), vol,[]);
end


tp = sum(reshape([resultsFullSeg.tp],length(classtoEvaluate),numoffiles)',1);
fp = sum(reshape([resultsFullSeg.fp],length(classtoEvaluate),numoffiles)',1);
fn = sum(reshape([resultsFullSeg.fn],length(classtoEvaluate),numoffiles)',1);
full_precision = tp ./ (tp + fp);
full_recall = tp ./ (tp + fn);
full_iou = tp ./ (tp + fn + fp);

fprintf('Semantic Scene Compeltion:\nprec ,recall , IoU\n mean: %f,%f,%f\n',mean(full_precision), mean(full_recall),  mean(full_iou));
for i = 1:length(full_recall)
    fprintf(' %s %f %f %f\n', obj_class{i+1}, full_precision(i), full_recall(i), full_iou(i));
end

%% compeltion 
occ_precision = mean([resultsFullOcc.precision]);
occ_recall = mean([resultsFullOcc.recall]);
occ_iou = mean([resultsFullOcc.iou]);
fprintf('Scene Compeltion:\nprec ,recall , IoU\n %f,%f,%f\n',occ_precision,occ_recall, occ_iou);

% % evaluate on voxlet subset 
% if strcmp(benchmark,'nyucad')||strcmp(benchmark,'nyu')
%    occ_precision = mean([resultsFullOcc(voxletTest).precision]);
%    occ_recall = mean([resultsFullOcc(voxletTest).recall]);
%    occ_iou = mean([resultsFullOcc(voxletTest).iou]);
%    fprintf('voxlet testset: IOU: %f,%f,%f\n',occ_precision,occ_recall, occ_iou);
% end


%% Visualization
fprintf('Visualization saved in :%s \n',fullfile(OutputPath,benchmark));
colorPalette = [ 0.09 0.75 0.81; 0.84 0.15 0.16 ;0.17 0.63 0.17 ;0.62 0.85 0.90 ;0.45 0.62 0.81;
                0.80 0.80 0.36;1.00 0.73 0.47 ;0.58 0.40 0.74; 0.12 0.47 0.71 ;
                0.74 0.74 0.13;1.00 0.50 0.05; 0.77 0.69 0.84 ;0.6  0.6  0.6];
for batchId = 1:50:size(predobjTensor,5);
    ld = load(fullfile(groundtruth_path,[Filename{batchId} '_gt_d4.mat'])); 
    sceneVox =ld.sceneVox_ds;
    ld = load(fullfile(evalvol_path,[Filename{batchId} '_vol_d4.mat']));
    vol =ld.flipVol_ds;

    sceneVox(sceneVox==255) = 0;
    labelobj = mapIds(sceneVox+1);
    nonfree_voxels_to_evaluate = abs(vol)<1|vol==-1;

    %% groundtruth 
    [gridPtsX,gridPtsY,gridPtsZ] = ndgrid(downscale/2:downscale:size(vol,3)*downscale-downscale/2,...
                                          downscale/2:downscale:size(vol,2)*downscale-downscale/2,...
                                          downscale/2:downscale:size(vol,1)*downscale-downscale/2);
    gridPts = [gridPtsX(:),gridPtsY(:),gridPtsZ(:)];    
    occSegLabelVal = labelobj(find(labelobj > 0));
    segGridPts = gridPts(find(labelobj > 0),:);
    pcwrite(pointCloud(segGridPts,'Color',colorPalette(occSegLabelVal+1,:)),fullfile(OutputPath,benchmark,['label_' num2str(batchId)]),'PLYFormat','binary');

    %% segmentation predictions
    predobj_conf = predobjTensor(:,:,:,:,batchId);
    [~,predobj] = max(predobj_conf,[],4);
    predobj = predobj-1;
    occSegPredVal = predobj(find(nonfree_voxels_to_evaluate&predobj>0));
    predGridPts = gridPts(find(nonfree_voxels_to_evaluate&predobj>0),:);
    pcwrite(pointCloud(predGridPts,'Color',colorPalette(occSegPredVal+1,:)),fullfile(OutputPath,benchmark,['pred_' num2str(batchId)]),'PLYFormat','binary');
end
