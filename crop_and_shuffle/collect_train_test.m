%% Author @zhiweid, Jan13 2015
% Readme: data_label stores image name and bounding boxes

% file_train = fopen('/media/storage/mzhai/nursinghome_temp_round_1/val_nursing.txt','w');
clear all;
% parameters
only_write = false;
write_train = true;
write_val = true;
write_test = false;

source_prefix = 'annotation_distance/';
% img_prefix = '/media/storage/zhiweid/CAD/ActivityDataset';


video_train = [7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44];

video_val = [4, 5, 6, 8, 9, 28, 29, 10, 25, 1, 2, 3, 11];

video_train2 = [45:51 54:58 61:64 66 67 69 70];

video_val2 = setdiff([45:72],video_train2);

video_train = [video_train video_train2];
video_val = [video_val video_val2];


num = 0;
patches_path = ['/media/storage/zhiweid/CollectiveActivityDataset/crop_images/crop_1011/'];
videos = [video_train video_val];

train_data = {};
test_data = {};

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
        
        action = data_label(i).action;
        temp = [];
        temp2 = [];
        for kk = 1:length(action)
            if action(kk) == 1 || action(kk) == 5
                continue
            end
            temp = [temp action(kk)];
            temp2 = [temp2 data_label(i).pose(kk)];
        end
        if length(temp) == 0
            continue;
        end
        train_data{total_num} = data_label(i);
        train_data{total_num}.action = temp;
        train_data{total_num}.pose = temp2;
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
        action = data_label(i).action;
        temp = [];
        temp2 = [];
        for kk = 1:length(action)
            if action(kk) == 1 || action(kk) == 5
                continue
            end
            temp = [temp action(kk)];
            temp2 = [temp2 data_label(i).pose(kk)];
        end
        if length(temp) == 0
            continue;
        end
        test_data{total_num} = data_label(i);
        test_data{total_num}.action = temp;
        test_data{total_num}.pose = temp2;
        total_num = total_num + 1;   
    end
end

save('CAD_pretrain.mat','train_data','test_data');
