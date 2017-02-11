function cameraPose = camExtrinsics2Pose(extrinsics)
    % cameraPose : &vx, &vy, &vz, &tx, &ty, &tz, &ux, &uy, &uz, &rx, &ry, &rz
    % extrinsics : camera to world
    org = extrinsics(:,4);
    tv = extrinsics(:,3);
    up = -extrinsics(:,2);
    rv = cross(tv,uv);
    
    %cameraPose = [extrinsics(:,4)',extrinsics(:,3)', extrinsics(:,2)', extrinsics(:,1)'];
    cameraPose = [org',tv',up',rv'];
end

