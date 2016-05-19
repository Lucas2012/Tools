function temporal_write_to_output 
 load('temporal_shuffled_CAD_pretrain.mat','train_data','test_data');
 load('temporal_pre_shuffled_CAD_pretrain.mat','train_pre','test_pre');
 load('temporal_after_shuffled_CAD_pretrain.mat','train_after','test_after');
 load('temporal_p_pre_shuffled_CAD_pretrain.mat','train_p_pre','test_p_pre');
 load('temporal_a_after_shuffled_CAD_pretrain.mat','train_a_after','test_a_after');
 date = '1011';
 
 file = fopen(['train_' date '.txt'],'w');
 framefile = fopen(['train_frame_' date '.txt'],'w');
 posefile = fopen(['train_pose_' date '.txt'],'w');
 
 % generate normal data
 ifvgg = false;
 ifcontext = false;
 ifdist = false;
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
%      count = 0;
%      [val, pos] = sort(train_data{i}.patch_label,'descend');
%      for j = 1:length(train_data{i}.patch_idx)
%          % fprintf(file,'%s\n',[num2str(train_data{i}.patch_idx(j)) '.jpg  '  num2str(train_data{i}.patch_label(j))]);
%          fprintf(file,'%s\n',[num2str(train_data{i}.patch_idx(pos(j))) '.jpg  '  num2str(train_data{i}.patch_label(pos(j)))]);
%          fprintf(posefile,'%s\n',[num2str(train_data{i}.patch_idx(pos(j))) '.jpg  '  num2str(train_data{i}.pose(pos(j))-1)]);
%          assert(train_data{i}.pose(pos(j))-1>=0);
%          assert(train_data{i}.patch_label(pos(j))>=0);
%          count = count+1;
%      end
%      for j = count+1:MAX_PEOPLE
%          fprintf(file,'%s\n',['meanimage.jpg  -1']);
%          fprintf(posefile,'%s\n',['meanimage.jpg  -1']);
%      end
     write_file(file,train_p_pre{i},MAX_PEOPLE);
     write_file(file,train_pre{i},MAX_PEOPLE);
     write_file(file,train_data{i},MAX_PEOPLE);
     write_file(file,train_after{i},MAX_PEOPLE);
     write_file(file,train_a_after{i},MAX_PEOPLE);
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
     imagename_p_pre = train_p_pre{i}.detect.imgname;
     imagename_pre = train_pre{i}.detect.imgname;
     imagename = train_data{i}.detect.imgname;
     imagename_after = train_after{i}.detect.imgname;
     imagename_a_after = train_a_after{i}.detect.imgname;
     idx_p_pre = strfind(imagename_p_pre,'zhiweid');
     idx_pre = strfind(imagename_pre,'zhiweid');
     idx = strfind(imagename,'zhiweid');
     idx_after = strfind(imagename_after,'zhiweid');
     idx_a_after = strfind(imagename_a_after,'zhiweid');
     s_p_pre = imagename_p_pre(idx_p_pre(1):end);
     s_pre = imagename_pre(idx_pre(1):end);
     s = imagename(idx(1):end);
     s_after = imagename_after(idx_after(1):end);
     s_a_after = imagename_a_after(idx_a_after(1):end);
%      patch_label = [];
%      for u = 1:length(train_data{i}.patch_label)
%          if train_data{i}.patch_label(u) ~= 0
%              patch_label = [patch_label train_data{i}.patch_label(u)];
%          end
%      end
%      if ifvgg
%          frame_path = ['/media/storage/' s];
%          im_frame = imread(frame_path);
%          im_frame(:,:,1) = im_frame(:,:,1) - 103.939;
%          im_frame(:,:,2) = im_frame(:,:,2) - 116.779;
%          im_frame(:,:,3) = im_frame(:,:,3) - 123.68;
%          imwrite(im_frame,['crop_1011/frames/' num2str(num_frame) '.jpg']);
%          s = [num2str(num_frame) '.jpg'];
%          num_frame = num_frame + 1;
%      end
     p_pre_patch = train_p_pre{i}.patch_label;
     pre_patch = train_pre{i}.patch_label;
     patch_label = train_data{i}.patch_label;
     after_patch = train_after{i}.patch_label;
     a_after_patch = train_a_after{i}.patch_label;
     label_p_pre = mode(p_pre_patch) - 0;
     label_pre = mode(pre_patch) - 0;
     label = mode(patch_label) - 0;
     label_after = mode(after_patch) - 0;
     label_a_after = mode(a_after_patch) - 0;
     fprintf(framefile,'%s\n',[s_p_pre '  ' num2str(label_p_pre)]);
     fprintf(framefile,'%s\n',[s_pre '  ' num2str(label_pre)]);
     fprintf(framefile,'%s\n',[s '  ' num2str(label)]);
     fprintf(framefile,'%s\n',[s_after '  ' num2str(label_after)]);
     fprintf(framefile,'%s\n',[s_a_after '  ' num2str(label_a_after)]);
 end
 if ifdist
     save(['spatialtrial_train_' date '_dist_fea.mat'],'data_dist');
 end
 if ifcontext 
    save(['spatialtrial_train_' date '_context_fea.mat'],'data_context');
 end
 fclose(file);
 fclose(posefile);
 
 
 file = fopen(['val_' date '.txt'],'w');
 framefile = fopen(['val_frame_' date '.txt'],'w');
 posefile = fopen(['val_pose_' date '.txt'],'w');
 data_dist = [];
 data_context = [];
 for i = 1:length(test_data)
%      count = 0;
%      [val, pos] = sort(test_data{i}.patch_label,'descend');
%      for j = 1:length(test_data{i}.patch_idx)
% %          fprintf(file,'%s\n',[num2str(test_data{i}.patch_idx(j)) '.jpg  '  num2str(test_data{i}.patch_label(j))]);
%          fprintf(file,'%s\n',[num2str(test_data{i}.patch_idx(pos(j))) '.jpg  '  num2str(test_data{i}.patch_label(pos(j)))]);
%          fprintf(posefile,'%s\n',[num2str(test_data{i}.patch_idx(pos(j))) '.jpg  '  num2str(test_data{i}.pose(pos(j))-1)]);
%          assert(test_data{i}.pose(pos(j))-1>=0);
%          assert(test_data{i}.patch_label(pos(j))>=0);
%          count = count+1;
%      end
%      for j = count+1:MAX_PEOPLE
%          fprintf(file,'%s\n',['meanimage.jpg  0']);
%          fprintf(posefile,'%s\n',['meanimage.jpg  0']);
%      end
     write_file(file,test_p_pre{i},MAX_PEOPLE);
     write_file(file,test_pre{i},MAX_PEOPLE);
     write_file(file,test_data{i},MAX_PEOPLE);
     write_file(file,test_after{i},MAX_PEOPLE);
     write_file(file,test_a_after{i},MAX_PEOPLE);
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
     imagename_p_pre = test_p_pre{i}.detect.imgname;
     imagename_pre = test_pre{i}.detect.imgname;
     imagename = test_data{i}.detect.imgname;
     imagename_after = test_after{i}.detect.imgname;
     imagename_a_after = test_a_after{i}.detect.imgname;
     idx_p_pre = strfind(imagename_p_pre,'zhiweid');
     idx_pre = strfind(imagename_pre,'zhiweid');
     idx = strfind(imagename,'zhiweid');
     idx_after = strfind(imagename_after,'zhiweid');
     idx_a_after = strfind(imagename_a_after,'zhiweid');
     s_p_pre = imagename_p_pre(idx_p_pre(1):end);
     s_pre = imagename_pre(idx_pre(1):end);
     s = imagename(idx(1):end);
     s_after = imagename_after(idx_after(1):end);
     s_a_after = imagename_a_after(idx_a_after(1):end);
%      if ifvgg
%          frame_path = ['/media/storage/' s];
%          im_frame = imread(frame_path);
%          im_frame(:,:,1) = im_frame(:,:,1) - 103.939;
%          im_frame(:,:,2) = im_frame(:,:,2) - 116.779;
%          im_frame(:,:,3) = im_frame(:,:,3) - 123.68;
%          imwrite(im_frame,['crop_1001/frames/' num2str(num_frame) '.jpg']);
%          s = [num2str(num_frame) '.jpg'];
%          num_frame = num_frame + 1;
%      end
     assert(length(test_data{i}.patch_label)>=0);
     p_pre_label = mode(test_p_pre{i}.patch_label) - 0;
     pre_label = mode(test_pre{i}.patch_label) - 0;
     label = mode(test_data{i}.patch_label) - 0;
     after_label = mode(test_after{i}.patch_label) - 0;
     a_after_label = mode(test_a_after{i}.patch_label) - 0;
     fprintf(framefile,'%s\n',[s_p_pre '  ' num2str(p_pre_label)]);
     fprintf(framefile,'%s\n',[s_pre '  ' num2str(pre_label)]);
     fprintf(framefile,'%s\n',[s '  ' num2str(label)]);
     fprintf(framefile,'%s\n',[s_after '  ' num2str(after_label)]);
     fprintf(framefile,'%s\n',[s_a_after '  ' num2str(a_after_label)]);
 end
 if ifdist
     save(['spatialtrial_val_' date '_dist_fea.mat'],'data_dist');
 end
 if ifcontext
    save(['spatialtrial_val_' date '_context_fea.mat'],'data_context');
 end
 fclose(file);
 fclose(posefile);

 function write_file(file,data,MAX_PEOPLE)
 count = 0;
 [val, pos] = sort(data.patch_label,'descend');
 for j = 1:length(data.patch_idx)
     % fprintf(file,'%s\n',[num2str(train_data{i}.patch_idx(j)) '.jpg  '  num2str(train_data{i}.patch_label(j))]);
     fprintf(file,'%s\n',[num2str(data.patch_idx(pos(j))) '.jpg  '  num2str(data.patch_label(pos(j)))]);
%      fprintf(posefile,'%s\n',[num2str(data{i}.patch_idx(pos(j))) '.jpg  '  num2str(data{i}.pose(pos(j))-1)]);
     assert(data.pose(pos(j))-1>=0);
     assert(data.patch_label(pos(j))>=0);
     count = count+1;
 end
 for j = count+1:MAX_PEOPLE
     fprintf(file,'%s\n',['meanimage.jpg  -1']);
%      fprintf(posefile,'%s\n',['meanimage.jpg  -1']);
 end
 
 
 