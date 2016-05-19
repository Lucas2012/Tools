%% Author @zhiweid, Jan13 2015
% Readme: data_label stores image name and bounding boxes

% file_train = fopen('/media/storage/mzhai/nursinghome_temp_round_1/val_nursing.txt','w');
clear all;
% parameters
only_write = false;
write_train = true;
write_val = true;
write_test = false;

source_prefix = '/media/storage/zhiweid/CollectiveActivityDataset/annote/Annotation/';
img_prefix = '/media/storage/zhiweid/CAD/ActivityDataset';


video_train = [7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44];

video_val = [4, 5, 6, 8, 9, 28, 29, 10, 25, 1, 2, 3, 11];

num = 1;
cropped_file = {};
val_file = {};
patches_path = ['/media/storage/zhiweid/CollectiveActivityDataset/crop_images/'];
videos = [video_train video_val];

video_patches_indice = {};
video_patches_start = {};
video_patches_in = zeros(1,length(video_train));
video_patches_st = zeros(1,length(video_train));

if ~only_write
    img_all_count = 0;
for k = videos
    disp(['video:' num2str(k)]);
    %load(['/media/storage/UCLA_courtyard_Dataset/annotation/annotation10_',num2str(k),'.mat']);
    try
        load([source_prefix 'annot_video' num2str(k) '.mat']);
    catch
        'not found'
        [source_prefix 'annot_video' num2str(k) '.mat']
        continue;
    end
    if num == 800 || num > 800
        num = num;
    end
    video_patches_start{k}.index = num+1;
    for i = 1:length(data_label)
        img_count = 0;
        imname = data_label(i).detect.imgname;
%         ed = length(imname);
%         st = ed - 15 + 1;
%         imname = [img_prefix 'Frames/' imname(1,st:ed)];
%         ind = strfind(imname, 'v');
%         imname = [img_prefix 'Frames/' imname(ind(end):end)];
%         frameNum = str2num(imname(1,st+7:ed-2));
        img = imread(imname);
        img_w = size(img,2);
        img_h = size(img,1);
        res = [size(img,1) size(img,2)];
        for j=1:size(data_label(i).bboxes_tracked,1)
            img_count = img_count + 1;
%             if img_count > MAXIMUM-2
%                 break;
%             end
            bbox = data_label(i).bboxes_tracked(j,1:4);
            patch = img(max(1,bbox(1,2)): min(img_h, bbox(1,2)+bbox(1,4)), max(1,bbox(1,1)): min(img_w,bbox(1,1)+bbox(1,3)),:);
            patch = imresize(patch,[256 256]);
            num = num+1;
            imname = sprintf('%d.jpg', num);
            label = data_label(i).action(j)-1;
            imwrite(patch,sprintf([patches_path 'patches/%d.jpg'], num));
%             end_t(1,img_count) = end_t(1,img_count)  + 1;
%             train_file{1,img_count}{1,end_t(1,img_count) } =  sprintf('%s %d\n',imname,label);
%             img_count_all = img_count_all+1;
            cropped_file{num}.string = sprintf('%s %d\n',imname,label); 
%             cropped_file{num}.framenum = frameNum; 
            cropped_file{num}.actionLabel = label; 
            a=1;
        end
        
        bbx_end=j+1;
%         for back_i = 1:1
%             while(1)
%                 ver = sort(randi(res(1),1,2));
%                 hor = sort(randi(res(2),1,2));
%                 bg_pos = [hor(1) ver(1) hor(2)-hor(1) ver(2)-ver(1)];
%                 bg_area = bg_pos(3)*bg_pos(4);
%                 if bg_pos(3) < 40 || bg_pos(4) < 40
%                     continue
%                 end
%                 
%                 flag = 0;
%                 for bbox_i = 1:size(data_label(i).bboxes_tracked,1)
%                     inter_area = rectint(bg_pos, data_label(i).bboxes_tracked(bbox_i,:));
%                     if inter_area >= 0.2*data_label(i).bboxes_tracked(bbox_i,3)*0.2*data_label(i).bboxes_tracked(bbox_i,4)
%                         flag = 1;
%                     end
%                 end
%                 if flag == 0
%                     break
%                 end
%             end
%             bg_crop = img(bg_pos(2):bg_pos(2)+bg_pos(4),bg_pos(1):bg_pos(1)+bg_pos(3),:);
%             data_label(i).bboxes_tracked = [data_label(i).bboxes_tracked; bg_pos(2) bg_pos(1) bg_pos(4) bg_pos(3) 0];
% %             imshow(img);
% %             rectangle('Position',bg_pos,'EdgeColor','r');
%             bg_crop = imresize(bg_crop,[256 256]);
%             num=num+1;
% %             img_count = img_count + 1;
%             imwrite(bg_crop,sprintf([ patches_path 'patches/%d.jpg'], num));
%             background_name = sprintf('%d.jpg', num);
%             label = 0;
% %             end_t(1,img_count) = end_t(1,img_count)  + 1;
% %             train_file{1,img_count}{1,end_t(1,img_count) } =  sprintf('%s %d\n',background_name,label);
% %             cropped_file{num} = sprintf('%s %d\n',imname,label);
%             cropped_file{num}.string = sprintf('%s %d\n',background_name,label); 
% %             cropped_file{num}.framenum = frameN   um; 
%             cropped_file{num}.actionLabel = label;
%             a=1;
%         end
%         a=1;
    end
    video_patches_indice{k}.index = num;
end
end

numt = 0;
numv = 0;
if write_train
    file = fopen([patches_path 'train_0831.txt'],'w');
    for v = video_train
            for i = video_patches_start{v}.index:video_patches_indice{v}.index
                fprintf(file,'%s',cropped_file{i}.string);
                numt = numt+1;
            end
    end
    fclose(file);
    numt
end
if write_val
    file = fopen([patches_path 'val_0831.txt'],'w');
    %mean_img = '1.jpg 0\n';
    %fprintf(file,'%s',mean_img);
    for v = video_val
            for i = video_patches_start{v}.index:video_patches_indice{v}.index
                fprintf(file,'%s',cropped_file{i}.string);
                numv = numv+1;
            end
    end
    fclose(file);
    numv
end
if write_test
    file = fopen([patches_path 'test.txt'],'w');
    for v = video_test
        try
            for i = video_patches_start{v}.index + 1:video_patches_indice{v}.index
                fprintf(file,'%s',cropped_file{i}.string);
            end
        catch
            continue;
        end
    end
    fclose(file);
end

% for i=1:size(train_file,2)
%     file = fopen([patches_path 'train.txt'],'w');
%     for j = 1:size(train_file{i},2)
%         fprintf(file,'%s',train_file{i}{j});
%     end
% end
% 
% source_prefix = '/media/storage/mzhai/final_with_track/';
% img_prefix = '/media/storage/chenleic/data/Nursing/';
% 
% % crop image patches
% num = 0;
% video_train = [1:20];
% video_val = [21:43];
% video_test = [50,52,53,54,55,57,58,59,60,61,62,63,64,65,71,72,73,74,75,76];
% MAXIMUM = 9;  % Set maximum patch numbers per frame
% val_file = cell(1,MAXIMUM);
% end_v = zeros(1,MAXIMUM);
% img_count = 0;
% for k=video_val
%     %load(['/media/storage/UCLA_courtyard_Dataset/annotation/annotation10_',num2str(k),'.mat']);
%     try
%         load([source_prefix 'annot_video' num2str(k) '_track.mat']);
%     catch
%         continue;
%     end
%     for i = 1:5:length(data_label)
%         img_count = 0;
%         imname = data_label(i).detect.imgname;
%         ed = length(imname);
%         st = ed - 15 + 1;
%         imname = [img_prefix 'Frames/' imname(1,st:ed)];
%         img = imread(imname);
%         img_w = size(img,2);
%         img_h = size(img,1);
%         res = [size(img,1) size(img,2)];
%         for j=1:size(data_label(i).bboxes_tracked,1)
%             img_count = img_count + 1;
%             if img_count > MAXIMUM-2
%                 break;
%             end
%             bbox = data_label(i).bboxes_tracked(j,1:4);
%             patch = img(max(1,bbox(1,2)): min(img_h, bbox(1,2)+bbox(1,4)), max(1,bbox(1,1)): min(img_w,bbox(1,1)+bbox(1,3)),:);
%             patch = imresize(patch,[256 256]);
%             num = num+1;
%             imname = sprintf('%d.jpg', num);
%             label = data_label(i).action(j);
%             imwrite(patch,sprintf('/media/storage/mzhai/nursinghome_temp_round_1/patches/%d.jpg', num));
%             end_v(1,img_count)  = end_v(1,img_count)  + 1;
%             val_file{1,img_count}{1,end_v(1,img_count) } =  sprintf('%s %d\n',imname,label);
%             a=1;
%         end
%         
%         bbx_end=j+1;
%         for back_i = 1:1
%             while(1)
%                 ver = sort(randi(res(1),1,2));
%                 hor = sort(randi(res(2),1,2));
%                 bg_pos = [hor(1) ver(1) hor(2)-hor(1) ver(2)-ver(1)];
%                 bg_area = bg_pos(3)*bg_pos(4);
%                 if bg_pos(3) < 40 || bg_pos(4) < 40
%                     continue
%                 end
%                 
%                 flag = 0;
%                 for bbox_i = 1:size(data_label(i).bboxes_tracked,1)
%                     inter_area = rectint(bg_pos, data_label(i).bboxes_tracked(bbox_i,:));
%                     if inter_area >= 0.2*data_label(i).bboxes_tracked(bbox_i,3)*0.2*data_label(i).bboxes_tracked(bbox_i,4)
%                         flag = 1;
%                     end
%                 end
%                 if flag == 0
%                     break
%                 end
%             end
%             bg_crop = img(bg_pos(2):bg_pos(2)+bg_pos(4),bg_pos(1):bg_pos(1)+bg_pos(3),:);
%             data_label(i).bboxes_tracked = [data_label(i).bboxes_tracked; bg_pos(2) bg_pos(1) bg_pos(4) bg_pos(3) 0];
% %             imshow(img);
% %             rectangle('Position',bg_pos,'EdgeColor','r');
%             bg_crop = imresize(bg_crop,[256 256]);
%             num=num+1;
%             img_count = img_count + 1;
%             imwrite(bg_crop,sprintf('/media/storage/mzhai/nursinghome_temp_round_1/patches/%d.jpg', num));
%             background_name = sprintf('%d.jpg', num);
%             label = 0;
%             end_v(1,img_count) = end_v(1,img_count)  + 1;
%             val_file{1,img_count}{1,end_v(1,img_count) } =  sprintf('%s %d\n',background_name,label);
%             a=1;
%         end
%         if img_count <= MAXIMUM
%             while img_count < MAXIMUM
%                 patchname = 'imagenet_mean.jpeg';
%                 label = 0;
%                 img_count = img_count + 1;
%                 end_v(1,img_count)  = end_v(1,img_count)  + 1;
%                 val_file{1,img_count}{1,end_v(1,img_count)} = sprintf('%s %d\n',patchname,label);
%             end
%         else
%             display('Exceed Maximum Persons per Frame!');
%             return;
%         end
%         a=1;
%     end
% end
% 
% for i=1:1:MAXIMUM
%     file = fopen(['/media/storage/mzhai/nursinghome_temp_round_1/TXT_PER_FRAME/val_',num2str(i),'.txt'],'w');
%     for j = 1:size(val_file{i},2)
%         fprintf(file,'%s',val_file{i}{j});
%     end
% end
