function [Rtilt,R] = rectify(XYZ)   
    %% XYZ is HxWx3 matrix
    % X = XYZ(:,:,1);Y = XYZ(:,:,2);Z = XYZ(:,:,3);
    % XYZnew = Rtilt*[X(:),Y(:),Z(:)]'
    [Rtilt,R,world_center] = dominantAxes([eye(3) zeros(3,1)],XYZ);


                   


function [Rtilt,R,world_center] = dominantAxes(cameraRt, pts)

%XYZ = pts; 
%points = [reshape(XYZ(:,:,1),1,[]);reshape(XYZ(:,:,2),1,[]);reshape(XYZ(:,:,3),1,[])];

S = 10;
pointsOK = pts(:,sum(isnan(pts),1)==0);
pointsOK = pointsOK(:,sum(pointsOK,1)>0);
pointsOK = pointsOK(:,1:S:end);
%tic;normals = points2normals_radius(pointsOK);toc;
normals = points2normals(pointsOK);

%{
figure,
s =1;
quiver3(pointsOK(1,1:S*s:end),pointsOK(2,1:S*s:end),pointsOK(3,1:S*s:end),normals(1,1:S*s:end),normals(2,1:S*s:end),normals(3,1:S*s:end)); 

figure,
indxxx = B == b;
pointsOKxxx = pointsOK(:,1:S:end);
quiver3(pointsOK(1,indxxx),pointsOK(2,indxxx),pointsOK(3,indxxx),normals(1,indxxx),normals(2,indxxx),normals(3,indxxx)); 
hold on
quiver3(pointsOK(1,1:S*s:end),pointsOK(2,1:S*s:end),pointsOK(3,1:S*s:end),nrm(1,1:s:end),nrm(2,1:s:end),nrm(3,1:s:end),'-.r'); 

figure,
plot3(bins(1,:),bins(2,:),bins(3,:),'.')

%}



% approximately 1313 bins
sphere = icosahedron2sphere(4)';
bins = sphere(:, sphere(1, :) >= 0);

%NSAMPLE = 1e5;
%sampleind = randsample(1 : size(normals, 2), min(size(normals, 2), NSAMPLE));
%normals = normals(:,sampleind );
[D, B] = max(abs(bins' * normals), [], 1);

H = accumarray(cat(2, B', repmat(1, [length(B) 1])), repmat(1, [length(B) 1]));

A = eye(3);
[~, I] = sort(-H);
for j = 1 : 3
    if ~isempty(I)
          b = I(1);
          % choose mean normal that falls into the biggest bin
          in_bin = normals(:, B == b);
          % flip mirrored normals
          dots = sum(in_bin .* repmat(bins(:, b), [1 size(in_bin, 2)]), 1);
          in_bin(:, (dots < 0)) = -in_bin(:, (dots < 0));
          v = mean(in_bin, 2);
          v = v / norm(v);

          A(:, j) = v;

          fprintf('Bin: %d, Normal: %f %f %f. Contains %d points. Mean vector: %f %f %f\n', b, bins(:, b), H(b), v);
          % remove bins that are not ~90 degrees away
          dots = sum(bins(:, I) .* repmat(v, [1 length(I)]), 1);
          I = I((dots >= cos(deg2rad(110))) & (dots <= cos(deg2rad(70))));
    end
end

axisI = A(:,1);
axisII = A(:,2);
axisII = axisII - (axisI'*axisII)*axisI;
axisII = axisII/norm(axisII);
axisIII = cross(axisI,axisII);

AA =[axisI,axisII,axisIII -1*[axisI,axisII,axisIII]];
[~, zi] = max(squeeze(cameraRt(1:3, 3, :))'*AA);
ZZ = AA(:, zi);
[~, xi] = max(squeeze(cameraRt(1:3, 1, :))'*AA);
XX = AA(:, xi);
[~, yi] = max(squeeze(cameraRt(1:3, 2, :))'*AA);
YY = AA(:, yi);
%{
for i =1:3,
    hold on;
    quiver3(1,1,1,AA(1,i),AA(2,i),AA(3,i));
quiver3(0,0,0,A(1,i),A(2,i),A(3,i));
pause;
end
axis tight;
%}
R = [XX YY ZZ]';
q = quaternion.rotateutov(ZZ, [0;0;1]);
Rtilt = RotationMatrix(q);
world_center = nanmean(reshape(pts,3,[]),2);







function rad = deg2rad(deg)

rad = deg*pi/180;

return;

function [coor,tri] = icosahedron2sphere(level)

% copyright by Jianxiong Xiao http://mit.edu/jxiao
% this function use a icosahedron to sample uniformly on a sphere
%{
Please cite this paper if you use this code in your publication:
J. Xiao, T. Fang, P. Zhao, M. Lhuillier, and L. Quan
Image-based Street-side City Modeling
ACM Transaction on Graphics (TOG), Volume 28, Number 5
Proceedings of ACM SIGGRAPH Asia 2009
%}


a= 2/(1+sqrt(5));
M=[
    0 a -1 a 1 0 -a 1 0
    0 a 1 -a 1 0 a 1 0
    0 a 1 0 -a 1 -1 0 a
    0 a 1 1 0 a 0 -a 1
    0 a -1 0 -a -1 1 0 -a
    0 a -1 -1 0 -a 0 -a -1
    0 -a 1 a -1 0 -a -1 0
    0 -a -1 -a -1 0 a -1 0
    -a 1 0 -1 0 a -1 0 -a
    -a -1 0 -1 0 -a -1 0 a
    a 1 0 1 0 -a 1 0 a
    a -1 0 1 0 a 1 0 -a
    0 a 1 -1 0 a -a 1 0
    0 a 1 a 1 0 1 0 a
    0 a -1 -a 1 0 -1 0 -a
    0 a -1 1 0 -a a 1 0
    0 -a -1 -1 0 -a -a -1 0
    0 -a -1 a -1 0 1 0 -a
    0 -a 1 -a -1 0 -1 0 a
    0 -a 1 1 0 a a -1 0
    ];

coor = reshape(M',3,60)';
%[M(:,[1 2 3]); M(:,[4 5 6]); M(:,[7 8 9])];


[coor, ~, idx] = unique(coor,'rows');

tri = reshape(idx,3,20)';

%{
for i=1:size(tri,1)
    x(1)=coor(tri(i,1),1);
    x(2)=coor(tri(i,2),1);
    x(3)=coor(tri(i,3),1);
    y(1)=coor(tri(i,1),2);
    y(2)=coor(tri(i,2),2);
    y(3)=coor(tri(i,3),2);
    z(1)=coor(tri(i,1),3);
    z(2)=coor(tri(i,2),3);
    z(3)=coor(tri(i,3),3);
    patch(x,y,z,'r');
end

axis equal
axis tight
%}

% extrude
coor = coor ./ repmat(sqrt(sum(coor .* coor,2)),1, 3);

for i=1:level
    m = 0;
    for t=1:size(tri,1)
        n = size(coor,1);
        coor(n+1,:) = ( coor(tri(t,1),:) + coor(tri(t,2),:) ) / 2;
        coor(n+2,:) = ( coor(tri(t,2),:) + coor(tri(t,3),:) ) / 2;
        coor(n+3,:) = ( coor(tri(t,3),:) + coor(tri(t,1),:) ) / 2;
        
        triN(m+1,:) = [n+1     tri(t,1)    n+3];
        triN(m+2,:) = [n+1     tri(t,2)    n+2];
        triN(m+3,:) = [n+2     tri(t,3)    n+3];
        triN(m+4,:) = [n+1     n+2         n+3];
        
        n = n+3;
        
        m = m+4;
        
    end
    tri = triN;
    
    % uniquefy
    [coor, ~, idx] = unique(coor,'rows');
    tri = idx(tri);
    
    % extrude
    coor = coor ./ repmat(sqrt(sum(coor .* coor,2)),1, 3);
end

% vertex number: 12  42  162  642

          