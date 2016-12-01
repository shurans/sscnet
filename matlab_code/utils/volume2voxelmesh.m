function mesh = volume2voxelmesh(labelobj,gap,colorPalette)
        
        if ~exist('colorPalette','var')
            colorPalette = [  0.09 0.75 0.81   
                              0.84 0.15 0.16 
                              0.17 0.63 0.17 
                              0.62 0.85 0.90 
                              0.45 0.62 0.81
                              0.80 0.80 0.36 
                              1.00 0.73 0.47 
                              0.58 0.40 0.74   
                              0.12 0.47 0.71 
                              0.74 0.74 0.13 
                              1.00 0.50 0.05
                              0.77 0.69 0.84 
                              0.6  0.6  0.6];
            colorPalette = colorPalette + 0.05;
            colorPalette(colorPalette>1) =1;
        end
        
        label_color_all = colorPalette(labelobj+1,:);
        
        [gridPtsX,gridPtsY,gridPtsZ] = ndgrid(1:gap:gap*size(labelobj,3),1:gap:gap*size(labelobj,2),1:gap:gap*size(labelobj,1));
        gridPts = [gridPtsX(:),gridPtsY(:),gridPtsZ(:)];
        label_color = label_color_all(find(labelobj > 0),:);
        points = gridPts(find(labelobj > 0),:);
        vox_size = 0.4*gap;
                                      
         numPoints = size(points,1);
         mesh.vertex = zeros(8*numPoints,3);
         mesh.vertex_color = zeros(8*numPoints,3);
         mesh.faces = zeros(6*numPoints,4);
        coeff = [-1,-1,-1;
                  1,-1,-1;
                  1, 1,-1;
                 -1, 1,-1;
                 -1,-1, 1;
                  1,-1, 1;
                  1, 1, 1;
                 -1, 1, 1;];
         for i =1:8
             mesh.vertex(i:8:end,:)=bsxfun(@plus,points,coeff(i,:)*vox_size);
             mesh.vertex_color(i:8:end,:) = label_color;
         end
         cubeface = [4,3,2,1;
                     6,7,8,5;
                     2,3,7,6;
                     1,5,8,4;
                     1,2,6,5;
                     3,4,8,7];
         for i = 1:numPoints
             mesh.faces(6*(i-1)+1:6*(i-1)+6,:) = cubeface + 8*(i -1);
         end
         
end