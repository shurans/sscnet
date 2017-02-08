function extrinsics = camPose2Extrinsics(cameraPose)
    % cameraPose : &vx, &vy, &vz, &tx, &ty, &tz, &ux, &uy, &uz, &rx, &ry, &rz
    % extrinsics : camera to world
    tv = cameraPose(4:6);
    uv = cameraPose(7:9);
    rv = cross(tv,uv);
    
    extrinsics = [rv',  -cameraPose(7:9)', cameraPose(4:6)', cameraPose(1:3)'];
end