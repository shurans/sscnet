function cameraPoses = readCameraPose(camereafile)
cameraPoses =[];
fid = fopen(camereafile,'r');
tline = fgets(fid);
while ischar(tline)
    parseline = sscanf(tline, '%f');
    cameraPoses = [cameraPoses;parseline'];
    tline = fgets(fid);
end
fclose(fid);
end