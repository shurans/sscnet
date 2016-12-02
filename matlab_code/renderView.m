function [depth,label] = renderView(file_in,cameraPoses)
         
         file_out = tempname; %['tmp' num2str(round(rand(1)*1000))];
         
         cmd = sprintf('../render/renderDepth  %s %s %f %f %f %f %f %f %f %f %f %f %f %f',file_in,file_out,cameraPoses);
         fprintf('%s\n',cmd);
         system(horzcat(['unset LD_LIBRARY_PATH; ', cmd]));
         
         % read in depth and label
         %A = fread(fileID,sizeA,precision)
         im_w = 640;
         im_h = 480;
         
         fid = fopen([file_out '.depth']);
         depth = fread(fid,im_w*im_h,'float');
         depth = reshape(depth,im_w,im_h)';
         depth = depth(end:-1:1,:);
         fclose(fid);
         
         fid = fopen([file_out '.label']);
         label = fread(fid,im_w*im_h,'uint32');
         label = reshape(label,im_w,im_h)';
         label = label(end:-1:1,:);
         fclose(fid);
         
         delete([file_out '.label']);
         delete([file_out '.depth']);
         %imagesc(label');imagesc(depth');
         %pause;
         %system(horzcat(['unset LD_LIBRARY_PATH; ','../render/renderDepth  01d4be86806197794c9333540d5bf77c/fl001_rm0007 tmp 35.230683 39.174732 1.295105 -0.833107 0.534932 0.140642 0.118346 -0.075989 0.990060 0.540302 0.841471 0.000000']));
end

