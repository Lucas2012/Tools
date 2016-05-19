 load('CAD_pretrain.mat');
 train_pre = {};
 train_p_pre = {};
 train_a_after = {};
 train_after = {};
 % train data
  p_pre_v = -1;
  pre_v = 0;
  for i = 1:length(train_data)
     imgname = train_data{i}.detect.imgname;
     idx = strfind(imgname,'seq');
     v = str2num(imgname(idx+3:idx+4));
     after_data = train_data(min(length(train_data),i+1));
     imgname_after = after_data{1}.detect.imgname;
     idx = strfind(imgname_after,'seq');
     after_v = str2num(imgname_after(idx+3:idx+4));
     
     a_after_data = train_data(min(length(train_data),i+2));
     imgname_a_after = a_after_data{1}.detect.imgname;
     idx = strfind(imgname_a_after,'seq');
     a_after_v = str2num(imgname_a_after(idx+3:idx+4));
    
     if p_pre_v ~= pre_v || p_pre_v ~= v
         p_pre_v = pre_v;
         b_before_frame = train_data{i};
     else
         b_before_frame = train_data{i-1};
     end
     if pre_v ~= v
         pre_v = v;
         before_frame = train_data{i};
     else
         before_frame = train_data{i-1};
     end
     
     if after_v ~= v || i == length(train_data);
         after_frame = train_data{i};
     else
         after_frame = train_data{i+1};
     end
     if a_after_v ~= v || i >= length(train_data) - 1;
         a_after_frame = train_data{i};
     else
         a_after_frame = train_data{i+2};
     end
     train_p_pre{i} = b_before_frame;
     train_pre{i} = before_frame;
     train_after{i} = after_frame;
     train_a_after{i} = a_after_frame;
 end
 
 % test data
 test_pre = {};
 test_p_pre = {};
 test_a_after = {};
 test_after = {};
 % test data
  p_pre_v = -1;
  pre_v = 0;
  for i = 1:length(test_data)
     imgname = test_data{i}.detect.imgname;
     idx = strfind(imgname,'seq');
     v = str2num(imgname(idx+3:idx+4));
     after_data = test_data(min(length(test_data),i+1));
     imgname_after = after_data{1}.detect.imgname;
     idx = strfind(imgname_after,'seq');
     after_v = str2num(imgname_after(idx+3:idx+4));
     
     a_after_data = test_data(min(length(test_data),i+2));
     imgname_a_after = a_after_data{1}.detect.imgname;
     idx = strfind(imgname_a_after,'seq');
     a_after_v = str2num(imgname_a_after(idx+3:idx+4));
    
     if p_pre_v ~= pre_v || p_pre_v ~= v
         p_pre_v = pre_v;
         b_before_frame = test_data{i};
     else
         b_before_frame = test_data{i-1};
     end
     if pre_v ~= v
         pre_v = v;
         before_frame = test_data{i};
     else
         before_frame = test_data{i-1};
     end
     
     if after_v ~= v || i == length(test_data);
         after_frame = test_data{i};
     else
         after_frame = test_data{i+1};
     end
     if a_after_v ~= v || i >= length(test_data)-1;
         a_after_frame = test_data{i};
     else
         a_after_frame = test_data{i+2};
     end
     test_p_pre{i} = b_before_frame;
     test_pre{i} = before_frame;
     test_after{i} = after_frame;
     test_a_after{i} = a_after_frame;
 end
 
 save('temporal_CAD_pretrain.mat','train_data','test_data');
 save('temporal_CAD_pretrain_pre.mat','train_pre','test_pre');
 save('temporal_CAD_pretrain_p_pre.mat','train_p_pre','test_p_pre');
 save('tempora_CAD_pretrain_after.mat','train_after','test_after');
 save('tempora_CAD_pretrain_a_after.mat','train_a_after','test_a_after');
 