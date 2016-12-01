function mesh2off(OFFfilename, faces,vertices)
% output a mesh into an OFF file
% input:
% OFFfilename is the file name for the off file
% faces is Nx3 matrix for vertex index
% vertices is KxD matrix. D is the number of polygon size. =3 if triangles


file = fopen(OFFfilename,'w');
fprintf (file, 'OFF %d %d 0\n', size(vertices,1), size(faces,1));
for v=1:size(vertices,1)
    fprintf(file, '%f %f %f\n', vertices(v,1), vertices(v,2), vertices(v,3));
end

faces = faces -1; % matlab starts from 1, office starts from 0

for f=1:size(faces,1)
    fprintf(file, '%d', size(faces,2));
    for i=1:size(faces,2)
        fprintf(file, ' %f', faces(f,i));
    end
    fprintf(file, '\n');
end

fclose(file);