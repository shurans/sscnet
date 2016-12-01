function results = evaluate_completion(GTV, V, empty_conf_V, vol,conf_threshold)
       voxels_to_evaluate = vol<0&vol>=-1;

       % getting the ground truth TSDF voxels
       gt = GTV(voxels_to_evaluate); 
       % Getting the relevant predictions
       prediction = V(voxels_to_evaluate);
       
       
       % now doing IOU
       union = sum(gt>0|prediction>0);
       intersection = sum(gt>0&prediction>0);


       results.iou = intersection/union;

       tp_occ = sum(gt > 0 & prediction > 0);
       fp_occ = sum(gt == 0 & prediction > 0);  
       fn_occ = sum(gt > 0 & prediction == 0);  
       
       results.tp_occ = tp_occ;
       results.fp_occ = fp_occ;
       results.fn_occ = fn_occ;
       if (tp_occ+tp_occ)>0
           results.precision = tp_occ/(tp_occ+fp_occ);
       else
           results.precision =0;
       end
       
       if (tp_occ+fn_occ)>0
           results.recall = tp_occ/(tp_occ+fn_occ);
       else
           results.recall =0;
       end
       
       if ~isempty(empty_conf_V)&&~isempty(conf_threshold)
           empty_conf = empty_conf_V(voxels_to_evaluate);
           for i = 1:length(conf_threshold)
               prediction_thre = ones(size(empty_conf));
               prediction_thre(empty_conf > conf_threshold(i)) = 0;
               results.tp_occ_th(i) = sum(gt >  0 & prediction_thre >  0);
               results.fp_occ_th(i) = sum(gt == 0 & prediction_thre >  0);
               results.fn_occ_th(i) = sum(gt >  0 & prediction_thre == 0);
           end
       end
       
      
end