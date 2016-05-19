function shuffle_CAD_pretrain_4_fold(fold)
% load train test data
output_path =  '4_fold_data';
load([output_path '/CAD_pretrain_' num2str(fold) '.mat']);

new_train = {};
index_all = Fisher_Yates(length(train_data),1000);
for i = 1:length(train_data)
    new_train{i} = train_data{index_all(i)};
end

new_test = {};
index_all = Fisher_Yates(length(test_data),1000);
for i = 1:length(test_data)
    new_test{i} = test_data{index_all(i)};
end

train_data = new_train;
test_data = new_test;

save([output_path '/shuffled_CAD_pretrain_' num2str(fold) '.mat'],'train_data','test_data');

