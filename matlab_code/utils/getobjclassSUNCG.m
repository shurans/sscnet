function [categoryRoot,classRootId, category,classId,insctancId] = getobjclassSUNCG(objname,objcategory)
            [~,insctancId] = ismember(objname,objcategory.all_labeled_obj);
            if (insctancId>0)
                classId = objcategory.classid(insctancId);
                category = objcategory.allcategories{objcategory.classid(insctancId)};
            else
                category = objname;
                classId = -1;
            end
            classRootId = length(objcategory.object_hierarchical)+1;
            categoryRoot = 'dont_care';
            for i = 1:length(objcategory.object_hierarchical)
                if ( ismember(category,objcategory.object_hierarchical{i}.clidern))
                    classRootId =i;
                    categoryRoot = objcategory.object_hierarchical{i}.categoryname;
                    break;
                end
            end
end


