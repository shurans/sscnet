function saveDepth (depth,filename)
        depth(isnan(depth)) =0;
        depth =single(depth)*1000;
        depthVis = uint16(depth);
        depthVis = bitor(bitshift(depthVis,3), bitshift(depthVis,3-16));
        imwrite(depthVis,filename);
end