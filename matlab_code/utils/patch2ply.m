function patch2ply(PLYfilename, coordinate, faces, rgb,alpha)
    % coordinate is 3 * n single matrix for n points
    % faces      is 3 * m matrix for m triangles
    % rgb        is 3 * n uint8  matrix for n points range [0, 255]

    if size(coordinate,2)==3 && size(coordinate,1)~=3
        coordinate = coordinate';
    end
    
    isValid = (~isnan(coordinate(1,:))) & (~isnan(coordinate(2,:))) & (~isnan(coordinate(3,:)));
    coordinate = coordinate(:,isValid);

    data = reshape(typecast(reshape(single(coordinate),1,[]),'uint8'),3*4,[]);

    if exist('rgb','var')
        if size(rgb,2)==3 && size(rgb,1)~=3
            rgb = rgb';
        end
        
        if ~isa(rgb,'uint8')
            if max(rgb(:))<=1
                rgb = rgb * 255;
            end
        end
        
        if isa(rgb,'double')
            rgb = uint8(rgb);
        end
        
        rgb = rgb(:,isValid);
        data = [data; rgb];
    end
    
    if exist('alpha','var')
        alpha = alpha(:)';
        if max(alpha(:))<=1
           alpha = alpha*255;
        end
        if isa(alpha,'double')
            alpha = uint8(alpha);
        end
        alpha = alpha(isValid);
        data = [data; alpha];
    end

    file = fopen(PLYfilename,'w');
    fprintf (file, 'ply\n');
    fprintf (file, 'format binary_little_endian 1.0\n');
    fprintf (file, 'element vertex %d\n', size(data,2));
    fprintf (file, 'property float x\n');
    fprintf (file, 'property float y\n');
    fprintf (file, 'property float z\n');
    if exist('rgb','var')
        fprintf (file, 'property uchar red\n');
        fprintf (file, 'property uchar green\n');
        fprintf (file, 'property uchar blue\n');
    end
    
    if exist('alpha','var')
       fprintf (file, 'property uchar alpha\n');
    end
    
    fprintf (file, 'element face %d\n', size(faces,2));
    fprintf (file, 'property list uchar int vertex_index\n');
    fprintf (file, 'end_header\n');
    
    % write vertices data
    fwrite(file, data,'uint8');
    
    
    % write faces data
    if (size(faces,2) > 0)
        %faces = faces([3 2 1],:); % reverse the order to get a better normal    
        faces_data = int32(faces-1);
        faces_data = reshape(typecast(reshape(faces_data,1,[]),'uint8'),size(faces,1)*4,[]);
        faces_data = [uint32(ones(1,size(faces,2))*size(faces,1)); faces_data];        
        fwrite(file, faces_data,'uint8');
    end
    
    fclose(file);

end