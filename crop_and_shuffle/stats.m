 load('CAD_pretrain.mat')
 
 a = zeros(1,6);
 for i = 1:length(test_data)
     for j = 1:length(test_data{i}.action)
         id = test_data{i}.patch_label(j)+1;
         a(id) = a(id) + 1;
     end
 end
 a
 
 a = zeros(1,6);
 for i = 1:length(train_data)
     for j = 1:length(train_data{i}.action)
         id = train_data{i}.patch_label(j)+1;
         a(id) = a(id) + 1;
     end
 end
 a