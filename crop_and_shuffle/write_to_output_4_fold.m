function write_to_output_4_fold(fold) 

 output_path = ['4_fold_data/'];
 load([output_path 'shuffled_CAD_pretrain_' num2str(fold) '.mat'],'train_data','test_data');
 date = num2str(fold);
 
 file = fopen([output_path 'train_' date '.txt'],'w');
 framefile = fopen([output_path 'train_frame_' date '.txt'],'w');
 posefile = fopen([output_path 'train_pose_' date '.txt'],'w');
 
 % generate normal data
 ifvgg = true;
 ifcontext = true;
 ifdist = true;
 if ifvgg
     MAX_PEOPLE = 0;
     ifcontext = false;
     ifdist = false;
 else
     MAX_PEOPLE = 14;
 end
 
 data_dist = [];
 data_context = [];
 num_frame = 1;
 for i = 1:length(train_data)
     count = 0;
     [val, pos] = sort(train_data{i}.patch_label,'descend');
     for j = 1:length(train_data{i}.patch_idx)
         % fprintf(file,'%s\n',[num2str(train_data{i}.patch_idx(j)) '.jpg  '  num2str(train_data{i}.patch_label(j))]);
         fprintf(file,'%s\n',[num2str(train_data{i}.patch_idx(pos(j))) '.jpg  '  num2str(train_data{i}.patch_label(pos(j)))]);
         fprintf(posefile,'%s\n',[num2str(train_data{i}.patch_idx(pos(j))) '.jpg  '  num2str(train_data{i}.pose(pos(j))-1)]);
         assert(train_data{i}.pose(pos(j))-1>=0);
         assert(train_data{i}.patch_label(pos(j))>=0);
         count = count+1;
     end
     for j = count+1:MAX_PEOPLE
         fprintf(file,'%s\n',['meanimage.jpg  -1']);
         fprintf(posefile,'%s\n',['meanimage.jpg  -1']);
     end
     if ifdist
         % distance matrix
         ammend_matrix = zeros(MAX_PEOPLE);
         ammend_matrix(1:count,1:count) = train_data{i}.dis_matrix;
         data_dist = [data_dist;ammend_matrix];
     end
     if ifcontext 
         % context matrix
         context = train_data{i}.context;
         if size(context,1) ~= 1
             context = sum(context)/size(context,1);
         end
         data_context = [data_context; context];
     end
     % frame txt:
     imagename = train_data{i}.detect.imgname;
     idx = strfind(imagename,'zhiweid');
     s = imagename(idx(1):end);
%      patch_label = [];
%      for u = 1:length(train_data{i}.patch_label)
%          if train_data{i}.patch_label(u) ~= 0
%              patch_label = [patch_label train_data{i}.patch_label(u)];
%          end
%      end
     patch_label = train_data{i}.patch_label;
     label = mode(patch_label) - 0;

     fprintf(framefile,'%s\n',[s '  ' num2str(label)]);
 end
 if ifdist
     save([output_path 'spatialtrial_train_' date '_dist_fea.mat'],'data_dist');
 end
 if ifcontext 
    save([output_path 'spatialtrial_train_' date '_context_fea.mat'],'data_context');
 end
 fclose(file);
 fclose(posefile);
 
 
 file = fopen([output_path 'val_' date '.txt'],'w');
 framefile = fopen([output_path 'val_frame_' date '.txt'],'w');
 posefile = fopen([output_path 'val_pose_' date '.txt'],'w');
 data_dist = [];
 data_context = [];
 for i = 1:length(test_data)
     count = 0;
     [val, pos] = sort(test_data{i}.patch_label,'descend');
     for j = 1:length(test_data{i}.patch_idx)
%          fprintf(file,'%s\n',[num2str(test_data{i}.patch_idx(j)) '.jpg  '  num2str(test_data{i}.patch_label(j))]);
         fprintf(file,'%s\n',[num2str(test_data{i}.patch_idx(pos(j))) '.jpg  '  num2str(test_data{i}.patch_label(pos(j)))]);
         fprintf(posefile,'%s\n',[num2str(test_data{i}.patch_idx(pos(j))) '.jpg  '  num2str(test_data{i}.pose(pos(j))-1)]);
         assert(test_data{i}.pose(pos(j))-1>=0);
         assert(test_data{i}.patch_label(pos(j))>=0);
         count = count+1;
     end
     for j = count+1:MAX_PEOPLE
         fprintf(file,'%s\n',['meanimage.jpg  0']);
         fprintf(posefile,'%s\n',['meanimage.jpg  0']);
     end
     if ifdist
         % distance matrix
         ammend_matrix = zeros(MAX_PEOPLE);
         ammend_matrix(1:count,1:count) = test_data{i}.dis_matrix;
         data_dist = [data_dist;ammend_matrix];
     end
     if ifcontext 
         % context matrix
         context = test_data{i}.context;
         if size(context,1)~=1
             context = sum(context)/size(context,1);
         end
         data_context = [data_context; context];
     end
     % frame txt:
     imagename = test_data{i}.detect.imgname;
     idx = strfind(imagename,'zhiweid');
     s = imagename(idx(1):end);
     assert(length(test_data{i}.patch_label)>=0);
     label = mode(test_data{i}.patch_label) - 0;
     fprintf(framefile,'%s\n',[s '  ' num2str(label)]);
 end
 if ifdist
     save([output_path 'spatialtrial_val_' date '_dist_fea.mat'],'data_dist');
 end
 if ifcontext
    save([output_path 'spatialtrial_val_' date '_context_fea.mat'],'data_context');
 end
 fclose(file);
 fclose(posefile);

 
 