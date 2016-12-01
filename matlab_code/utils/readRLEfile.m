function checkVox = readRLEfile(sceneVoxFilename)
fileID = fopen(sceneVoxFilename,'r');  
voxOriginWorld = fread(fileID,3,'single');
camPoseArr = fread(fileID,16,'single');
checkVoxRLE = fread(fileID,'uint32');
fclose(fileID);
checkVox = [];
parfor RLEIdx = 1:(length(checkVoxRLE)/2)
    fprintf('Checking RLE: %d/%d\n',RLEIdx,length(checkVoxRLE)/2);
    checkVoxVal  = checkVoxRLE(RLEIdx*2-1);
    checkVoxIter = checkVoxRLE(RLEIdx*2);
    checkVox = [checkVox,repmat(checkVoxVal,1,checkVoxIter)];
end
checkVox = reshape(checkVox,[240,144,240]);
% [gridPtsX,gridPtsY,gridPtsZ] = ndgrid(1:240,1:144,1:240);
% gridPts = [gridPtsX(:),gridPtsY(:),gridPtsZ(:)]';
% gridPts = gridPts(:,find(checkVox > 0))';
end