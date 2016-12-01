function outV = downSample(V,downScale,sample_mode)
if ~exist('sample_mode','var')
    sample_mode = 'label';
end
if strcmp(sample_mode,'label')
    outSize = round(size(V)/downScale);
    threshold = 0.95 * downScale * downScale * downScale;
    [x_i,y_i,z_i]=ndgrid(1:outSize(1),1:outSize(2),1:outSize(3));
    for id = 1:length(x_i(:))
        subvolume = V((x_i(id)-1)*downScale+1:x_i(id)*downScale,...
                      (y_i(id)-1)*downScale+1:y_i(id)*downScale,...
                      (z_i(id)-1)*downScale+1:z_i(id)*downScale);
       if sum(subvolume(:)==0|subvolume(:)==255) > threshold
          outV(x_i(id),y_i(id),z_i(id)) =  mode(subvolume(:));
       else
          outV(x_i(id),y_i(id),z_i(id)) =  mode(subvolume(subvolume>0&subvolume<255));
       end
    end 
else
    outSize = round(size(V)/downScale);
    threshold = 0.75 * downScale * downScale * downScale;
    
    [x_i,y_i,z_i]=ndgrid(1:outSize(1),1:outSize(2),1:outSize(3));
    for id = 1:length(x_i(:))
        subvolume = V((x_i(id)-1)*downScale+1:x_i(id)*downScale,...
                      (y_i(id)-1)*downScale+1:y_i(id)*downScale,...
                      (z_i(id)-1)*downScale+1:z_i(id)*downScale);
       if sum(abs(subvolume(:))>=1) > threshold
          outV(x_i(id),y_i(id),z_i(id)) =  mode(subvolume(:));
       else
          outV(x_i(id),y_i(id),z_i(id)) =  mean(subvolume(:));
       end
    end 
end
end

% function val = modeLargerZero(id)
%     global Vol x_i y_i z_i ds
%     subvolume = Vol((x_i(id)-1)*ds+1:x_i(id)*ds,...
%                   (y_i(id)-1)*ds+1:y_i(id)*ds,...
%                   (z_i(id)-1)*ds+1:z_i(id)*ds);
%    if sum(subvolume(:)==0) < 0.95 * ds * ds * ds
%       val =  mode(subvolume(subvolume>0));
%    else
%       val =  0;
%    end
% end