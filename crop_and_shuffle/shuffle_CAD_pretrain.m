% load train test data
load('CAD_pretrain.mat');

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

save('shuffled_CAD_pretrain.mat','train_data','test_data');

