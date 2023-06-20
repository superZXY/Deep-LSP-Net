clc;clear;
close all;
DATASET_NAME = 'NUST-SIRST';
%% MUAA MUDT 1K
% strDir = ['J:\paper\trans\trans6\dataset\', DATASET_NAME, '\images\'];
% mat_file = ['J:\paper\trans\trans6\医学\SLR-Net-master\IPI-for-small-target-detection-master\mat\', DATASET_NAME, '\'];
% labelDir = ['J:\paper\trans\trans6\dataset\', DATASET_NAME, '\masks\'];
% database = build_database(['J:\paper\trans\trans6\dataset\', DATASET_NAME, '\images'],'.png');

%% NUST
strDir = ['J:\paper\trans\trans6\dataset\', DATASET_NAME, '\MDvsFA_cGAN-master\data\test_org\'];
mat_file = ['J:\paper\trans\trans6\医学\SLR-Net-master\IPI-for-small-target-detection-master\mat\', DATASET_NAME, '\'];
labelDir = ['J:\paper\trans\trans6\dataset\', DATASET_NAME, '\MDvsFA_cGAN-master\data\test_gt\'];
database = build_database(['J:\paper\trans\trans6\dataset\', DATASET_NAME, '\MDvsFA_cGAN-master\data\test_org\'],'.png');

files = database.cname;
opt.dw = 50;
opt.dh = 50;
opt.x_step = 10;
opt.y_step = 10;
% figure('units','normalized','outerposition',[0 0 1 1]);
for i=1:length(files)
% for i=1:10
    fprintf('%d/%d: %s\n', length(files), i, files{i});
    name = files{i};
    label_name = [name(1:end-4) name(end-3:end)];
    I = imread([strDir files{i}]);
    I = I(:,:,1);
    I = imresize(I,[300,300]);
    imwrite(I,['train/image_complex/' 'image_' num2str(i) '.tif']);
    I_label = imread([labelDir label_name]);I_label = I_label(:,:,1);
%     I = imresize(I,[300,300]);
    I_label = imresize(I_label,[300,300]);
    imwrite(I_label,['train/label_complex/' 'label_' num2str(i) '.tif']);
%     figure,imshow(I_label,[])
    [m n] = size(I);
    dw = opt.dw;
    dh = opt.dh;
    x_step = opt.x_step;
    y_step = opt.y_step;
    I1 = I ; % imfilter(I, fspecial('gaussian', 5));
    data = uint8([]);
    label = uint8([]);
    for ii = 1:y_step:m-dh+1
        for jj = 1:x_step:n-dw+1
            temp = I1(ii:ii+dh-1, jj:jj+dw-1);
            data = [data, temp(:)];
            temp2 = I_label(ii:ii+dh-1, jj:jj+dw-1);
            label = [label, temp2(:)];          
        end
    end
    data = data';
    label = label';


%     mat_name = [name(1:end-4) '.mat'];
    mat_name = ['Train_' num2str(i) '.mat'];

    save([mat_file,mat_name],'data','label')
    save_name{i} = name(1:end-4);
end
