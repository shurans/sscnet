function cameraPose = extrinsics2camPose(extrinsics)
    % cameraPose : &vx, &vy, &vz, &tx, &ty, &tz, &ux, &uy, &uz, &rx, &ry, &rz
    % extrinsics : camera to world
    cameraPose = [extrinsics(:,4)',extrinsics(:,3)', extrinsics(:,2)', extrinsics(:,1)'];
end

