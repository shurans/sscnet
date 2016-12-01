function show3DModel(faces,vertices,showNormal)
%
% this function draws a 3D CG model with the normal of their faces
% the normal is computed by left-hand-rule
% http://stackoverflow.com/questions/1516296/find-the-normal-angle-of-the-face-of-a-triangle-in-3d-given-the-co-ordinated-of
% i.e. vertices are ordered clockwise with respect to its outward normal
% 
% input:
% faces is Nx3 matrix for vertex index
% vertices is Kx3 matrix
%
% demo:
% load model;
% show3DModel(faces,vertices);

X = reshape(vertices(faces(:),1),size(faces))';
Y = reshape(vertices(faces(:),2),size(faces))';
Z = reshape(vertices(faces(:),3),size(faces))';

fill3(X,Y,Z,'y');

cX = mean(X(1:3,:));
cY = mean(Y(1:3,:));
cZ = mean(Z(1:3,:));

p21= [X(2,:) - X(1,:); Y(2,:) - Y(1,:); Z(2,:) - Z(1,:)];
p31= [X(3,:) - X(1,:); Y(3,:) - Y(1,:); Z(3,:) - Z(1,:)];

nXYZ = cross(p21,p31);
lXYZ = sqrt(sum(nXYZ.^2,1));
nXYZ = nXYZ ./ repmat(lXYZ,3,1);

hold on;
if ~exist('showNormal','var') || showNormal
    quiver3(cX,cY,cZ,nXYZ(1,:),nXYZ(2,:),nXYZ(3,:));
end
axis equal
