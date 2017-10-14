function normals = points2normals(points)
    % estimating a normal vector based on nearby 100 points
    % points is 3 * n matrix for n points

    if size(points,2)==3 && size(points,1)~=3
        points = points';
    end
    
    normals = lsqnormest(points, 50);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functions from http://www.mathworks.com/matlabcentral/fileexchange/27804-iterative-closest-point

% Least squares normal estimation from point clouds using PCA
%
% H. Hoppe, T. DeRose, T. Duchamp, J. McDonald, and W. Stuetzle. 
% Surface reconstruction from unorganized points. 
% In Proceedings of ACM Siggraph, pages 71:78, 1992.
%
% p should be a matrix containing the horizontally concatenated column
% vectors with points. k is a scalar indicating how many neighbors the
% normal estimation is based upon.
%
% Note that for large point sets, the function performs significantly
% faster if Statistics Toolbox >= v. 7.3 is installed.
%
% Jakob Wilm 2010

function n = lsqnormest(p, k)
m = size(p,2);
n = zeros(3,m);

v = ver('stats');
if str2double(v.Version) >= 7.5 
    neighbors = transpose(knnsearch(transpose(p), transpose(p), 'k', k+1));
else
    neighbors = k_nearest_neighbors(p, p, k+1);
end

for i = 1:m
    x = p(:,neighbors(2:end, i));
    p_bar = 1/k * sum(x,2);
    
    P = (x - repmat(p_bar,1,k)) * transpose(x - repmat(p_bar,1,k)); %spd matrix P
    %P = 2*cov(x);
    
    [V,D] = eig(P);
    
    [~, idx] = min(diag(D)); % choses the smallest eigenvalue
    
    n(:,i) = V(:,idx);   % returns the corresponding eigenvector    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Program to find the k - nearest neighbors (kNN) within a set of points. 
% Distance metric used: Euclidean distance
%
% Note that this function makes repetitive use of min(), which seems to be
% more efficient than sort() for k < 30.

function [neighborIds,neighborDistances] = k_nearest_neighbors(dataMatrix, queryMatrix, k)

numDataPoints = size(dataMatrix,2);
numQueryPoints = size(queryMatrix,2);

neighborIds = zeros(k,numQueryPoints);
neighborDistances = zeros(k,numQueryPoints);

D = size(dataMatrix, 1); %dimensionality of points

for i=1:numQueryPoints
    d=zeros(1,numDataPoints);
    for t=1:D % this is to avoid slow repmat()
        d=d+(dataMatrix(t,:)-queryMatrix(t,i)).^2;
    end
    for j=1:k
        [s,t] = min(d);
        neighborIds(j,i)=t;
        neighborDistances(j,i)=sqrt(s);
        d(t) = NaN; % remove found number from d
    end
end
    