function [rgb,points3d]=read_3d_pts_general(depthInpaint,K,depthInpaintsize,imageName)
    % K is [fx 0 cx; 0 fy cy; 0 0 1];  
    % for uncrop image crop =[1,1];
    % imageName is the full path to image
    cx = K(1,3); cy = K(2,3);  
    fx = K(1,1); fy = K(2,2); 
    invalid = depthInpaint==0;
    if ~isempty(imageName)
        if ischar(imageName)
            im = imread(imageName);
            rgb = im2double(im); 
        else
            rgb = imageName;
        end
    else
        rgb =double(cat(3,zeros(depthInpaintsize(1),depthInpaintsize(2)),...
                        ones(depthInpaintsize(1),depthInpaintsize(2)),...
                        zeros(depthInpaintsize(1),depthInpaintsize(2))));
    end
    rgb = reshape(rgb, [], 3);
    %3D points
    [x,y] = meshgrid(1:depthInpaintsize(2), 1:depthInpaintsize(1));   
    x3 = (x-cx).*depthInpaint*1/fx;  
    y3 = (y-cy).*depthInpaint*1/fy;  
    z3 = depthInpaint;  
    points3d = [x3(:) y3(:) z3(:)];
    points3d(invalid(:),:) =NaN;
end