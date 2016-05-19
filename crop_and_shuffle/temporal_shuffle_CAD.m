% load train test data
 load('temporal_CAD_pretrain.mat','train_data','test_data');
 load('temporal_CAD_pretrain_pre.mat','train_pre','test_pre');
 load('temporal_CAD_pretrain_p_pre.mat','train_p_pre','test_p_pre');
 load('tempora_CAD_pretrain_after.mat','train_after','test_after');
 load('tempora_CAD_pretrain_a_after.mat','train_a_after','test_a_after');

new_train = {};
new_train_pre = {};
new_train_after = {};
new_train_p_pre = {};
new_train_a_after = {};
index_all = Fisher_Yates(length(train_data),1000);
for i = 1:length(train_data)
    new_train{i} = train_data{index_all(i)};
    new_train_pre{i} = train_pre{index_all(i)};
    new_train_after{i} = train_after{index_all(i)};
    new_train_p_pre{i} = train_p_pre{index_all(i)};
    new_train_a_after{i} = train_a_after{index_all(i)};
end

new_test = {};
new_test_pre = {};
new_test_after = {};
new_test_p_pre = {};
new_test_a_after = {};
index_all = Fisher_Yates(length(test_data),1000);
for i = 1:length(test_data)
    new_test{i} = test_data{index_all(i)};
    new_test_pre{i} = test_pre{index_all(i)};
    new_test_after{i} = test_after{index_all(i)};
    new_test_p_pre{i} = test_p_pre{index_all(i)};
    new_test_a_after{i} = test_a_after{index_all(i)};
end

train_data = new_train;
train_pre = new_train_pre;
train_after = new_train_after;
train_p_pre = new_train_p_pre;
train_a_after = new_train_a_after;
test_data = new_test;
test_pre = new_test_pre;
test_after = new_test_after;
test_p_pre = new_test_p_pre;
test_a_after = new_test_a_after;

save('temporal_shuffled_CAD_pretrain.mat','train_data','test_data');
save('temporal_pre_shuffled_CAD_pretrain.mat','train_pre','test_pre');
save('temporal_after_shuffled_CAD_pretrain.mat','train_after','test_after');
save('temporal_p_pre_shuffled_CAD_pretrain.mat','train_p_pre','test_p_pre');
save('temporal_a_after_shuffled_CAD_pretrain.mat','train_a_after','test_a_after');

