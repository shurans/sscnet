function depthInpaint = readDepth(depthpath)         
    depthVis = imread(depthpath);
    depthInpaint = bitor(bitshift(depthVis,-3), bitshift(depthVis,16-3));
    depthInpaint = double(depthInpaint)/1000; 
end