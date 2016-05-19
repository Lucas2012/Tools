function collect_train_test_4_fold(fold)
%% Author @zhiweid, Jan13 2015
% Readme: data_label stores image name and bounding boxes

% file_train = fopen('/media/storage/mzhai/nursinghome_temp_round_1/val_nursing.txt','w');
% parameters
only_write = false;
write_train = true;
write_val = true;
write_test = false;

source_prefix = 'annotation_distance_4_fold/';
img_prefix = '/media/storage/zhiweid/CAD/ActivityDataset';
output_path =  '4_fold_data';


test_fold{1}.fold = [1     4     5     6     8     2     7    28    35    11    10    26];
test_fold{2}.fold = [9    15    19    20     3    16    36    38    12    40    41];
test_fold{3}.fold = [21    22    23    24    17    43    37    13    25    39];
test_fold{4}.fold = [30    32    33    42    44    18    31    29    34    14    27];

num = 0;
patches_path = ['/media/storage/zhiweid/CollectiveActivityDataset/crop_images/crop_4_fold/'];
videos = [1:44];

train_data = {};
test_data = {};

video_train = setdiff(videos,test_fold{fold}.fold);
video_val = test_fold{fold}.fold;

total_num = 1;
for k = video_train
    disp(['video:' num2str(k)]);
    %load(['/media/storage/UCLA_courtyard_Dataset/annotation/annotation10_',num2str(k),'.mat']);
    try
        load([source_prefix 'annot_video' num2str(k) '.mat']);
    catch
        disp(['not found' source_prefix 'annot_video' num2str(k) '.mat']);
        continue;
    end
    for i = 1:length(data_label)
        train_data{total_num} = data_label(i);
        total_num = total_num + 1;  
    end
end

total_num = 1;
for k = video_val
    disp(['video:' num2str(k)]);
    %load(['/media/storage/UCLA_courtyard_Dataset/annotation/annotation10_',num2str(k),'.mat']);
    try
        load([source_prefix 'annot_video' num2str(k) '.mat']);
    catch
        disp(['not found' source_prefix 'annot_video' num2str(k) '.mat']);
        continue;
    end
    for i = 1:length(data_label)
        test_data{total_num} = data_label(i);
        total_num = total_num + 1;  
    end
end

save([output_path '/CAD_pretrain_' num2str(fold) '.mat'],'train_data','test_data');
