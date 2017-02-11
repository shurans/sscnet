function cameraInfo = readCameraName(cameraInfofile)
cameraInfo =[];
fid = fopen(cameraInfofile,'r');
tline = fgets(fid);
cnt = 1;
while ischar(tline)
    %Room#0_0_3
   parseline = sscanf(tline, 'Room#%d_%d_%d');
   if isempty(parseline)
       break
   end
   cameraInfo(cnt).floorId = parseline(1);
   cameraInfo(cnt).roomId  = parseline(2);
   cnt = cnt+1;
   tline = fgets(fid);
end
fclose(fid);
end