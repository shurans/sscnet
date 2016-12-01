function results = evaluate_prediction(GTV, V, voxels_to_evaluate,classtoEvaluate)
   % evalutes a prediction grid, assuming to be the same size and position  as the ground truth grid...
   
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
   % pr
   for i = 1:length(classtoEvaluate)
       
       
       tp = sum(gt == classtoEvaluate(i) & prediction == classtoEvaluate(i));
       fp = sum(gt ~= classtoEvaluate(i) & prediction == classtoEvaluate(i));  
       fn = sum(gt == classtoEvaluate(i) & prediction ~= classtoEvaluate(i));  
       %tn = sum(gt ~= classtoEvaluate(i) & prediction ~= classtoEvaluate(i) );
       
       results.tp(i)  = tp;
       results.fp(i)  = fp;
       results.fn(i)  = fn;
   end   
end



