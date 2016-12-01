function write_binvox(filename,voxels,translate,scale)
         
         fid = fopen(filename,'w'); 
         % write header 
         depth   = size(voxels,1);
         width   = size(voxels,2);
         height  = size(voxels,3);
         wxh     = width * height; 
         totalsize = wxh * depth;
         % matlab's strange ordering 
         voxels = permute(voxels,[2,3,1]);
         
         fprintf(fid,'#binvox 1\n');
         fprintf(fid,'dim %d %d %d\n',depth,width,height);
         fprintf(fid,'translate %f %f %f\n',translate(1),translate(2),translate(3));
         fprintf(fid,'scale %f\n',scale);
         fprintf(fid,'data\n');
         
         
         % write voxel data
         bytes_written = 0;
         total_ones = 0;
         index = 0;
         
          while (index < totalsize)
               value = voxels(index+1);
               if value > 255
                   error('value > 255 byte precision is not enough ! \n');
               end
               
               count = 0;
               while ((index < totalsize) && (count < 255) && (value == voxels(index+1)))
                    index = index +1;
                    count = count+1;
               end
               if (value>0)
                    total_ones = total_ones+count;
               end
               fwrite(fid,uint8([value,count]),'uint8');
               bytes_written = bytes_written + 2;
          end
         
         
         fclose(fid);
end