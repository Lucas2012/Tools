%% Author @zhiweid, Jan13 2015
% Readme: data_label stores image name and bounding boxes

% file_train = fopen('/media/storage/mzhai/nursinghome_temp_round_1/val_nursing.txt','w');
clear all;
% parameters
only_write = false;
write_train = true;
write_val = true;
write_test = false;
if_vgg = false;
if_clean = true;

source_prefix = '/media/storage/zhiweid/CollectiveActivityDataset/annote/every10frames_annote_context_clean/';
img_prefix = '/media/storage/zhiweid/CAD/ActivityDataset';


video_train = [7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44];

video_val = [4, 5, 6, 8, 9, 28, 29, 10, 25, 1, 2, 3, 11];


num = 0;
patches_path = ['/media/storage/zhiweid/CollectiveActivityDataset/crop_images/crop_4_fold/'];
videos = [video_train video_val];

for k = videos
    disp(['video:' num2str(k)]);
    %load(['/media/storage/UCLA_courtyard_Dataset/annotation/annotation10_',num2str(k),'.mat']);
    try
        load([source_prefix 'annot_video' num2str(k) '.mat']);
    catch
        disp(['not found' source_prefix 'annot_video' num2str(k) '.mat']);
        continue;
    end
    for i = 1:length(data_label)
        img_count = 0;
        imname = data_label(i).detect.imgname;
        idx_name = strfind(imname,'zhiweid');
        imagepath = imname(idx_name:end);
        imagepath = ['/media/storage/' imagepath];
        img = imread(imagepath);
        img_w = size(img,2);
        img_h = size(img,1);
        res = [size(img,1) size(img,2)];
        patch_idx = [];
        patch_label = [];
        
        % crop image patches:
        for j=1:size(data_label(i).bboxes_tracked,1)
            img_count = img_count + 1;
            bbox = data_label(i).bboxes_tracked(j,1:4);
            patch = img(max(1,bbox(1,2)): min(img_h, bbox(1,2)+bbox(1,4)), max(1,bbox(1,1)): min(img_w,bbox(1,1)+bbox(1,3)),:);
            patch = imresize(patch,[256 256]);
            if if_vgg
                patch(:,:,1) = patch(:,:,1)-103.939;
                patch(:,:,2) = patch(:,:,2)-116.779;
                patch(:,:,3) = patch(:,:,3)-123.68;
            end
            num = num+1;
            imname = sprintf('%d.jpg', num);
            label = data_label(i).action(j)-2;
            assert(label>=0);
            imwrite(patch,sprintf([patches_path 'patches/%d_%d_%d_%d.jpg'], k,i,j,label));
            patch_idx = [patch_idx num];
            patch_label = [patch_label label];            
        end
        data_label(i).patch_idx = patch_idx;
        data_label(i).patch_label = patch_label;
        
        % calculate distance matrix:
        dis_matrix = zeros(size(data_label(i).bboxes_tracked,1));
        for j = 1:size(data_label(i).bboxes_tracked,1)
            posj = data_label(i).bboxes_tracked(j,:);
            posj = [posj(1) + posj(3)/2, posj(2)]; 
            for m = j+1:size(data_label(i).bboxes_tracked,1)
                posk = data_label(i).bboxes_tracked(m,:);
                posk = [posk(1) + posk(3)/2, posk(2)]; 
                dis = sqrt(sum((posj-posk).^2));
                dis_matrix(j,m) = dis;
            end
        end
        dis_matrix = dis_matrix + dis_matrix';
        data_label(i).dis_matrix = dis_matrix;   
    end
    save(['annotation_distance_4_fold/annot_video' num2str(k) '.mat'],'data_label');
end
