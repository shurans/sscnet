function [voxels,scale,translate] = read_binvox(filename)
         fid = fopen(filename,'r');
         % read header
         tline = fscanf(fid,'%s',1);
         if ~strcmp(tline,'#binvox')
             fprintf(['Error: first line reads [' tline '] instead of [#binvox]\n']);
         end
         version = fscanf(fid,'%d',1); 
%          fprintf('reading binvox version %d\n',version);
         depth = -1;
         done = 0;
         while (~feof(fid)&&~done)
            tline = fscanf(fid,'%s',1);
            if strcmp(tline,'data')
                done = 1;
            elseif strcmp(tline,'dim')
                A=fscanf(fid,'%d',3);
                width = A(1); 
                height= A(2); 
                depth = A(3);
            elseif strcmp(tline,'translate')
                translate = fscanf(fid,'%f',3);
            elseif strcmp(tline,'scale')
                scale =fscanf(fid,'%f',1);
            end
         end
         
         scale = scale/width;
         totalsize = width * height * depth;
         voxels = zeros(width, height, depth);
         % read voxel data
         index = 0;end_index = 0;nr_voxels = 0;
         
         
         B = fread(fid,1,'uint8'); %read the linefeed char
         while((end_index < totalsize) && ~feof(fid)) 
              B = fread(fid,2,'uint8');
              value = B(1);
              count = B(2);
              end_index = index + count;
              
              if (end_index >=totalsize) 
                  break;
              end
              
              for i=index:min(totalsize,end_index)
                  voxels(i+1) = value; % matlab index starts 1
              end
              
              if (value) 
                  nr_voxels = nr_voxels+count;
              end
              index = end_index;
         end
         % matlab's strange ordering 
         voxels = permute(voxels,[3,1,2]);
         fclose(fid);
end