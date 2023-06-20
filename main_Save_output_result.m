clc;clear;close all;
% load('filename_simple.mat')
% files_name = files;
database = build_database('J:\paper\trans\trans6\医学\SLR-Net-master3\results\stable\NUDT_PART','.mat');
% database = build_database('G:\paper\trans6\医学\SLR-Net-master\results\stable\2022-03-05T10-49-47ZXY_NET_SIRST12_lr_0.001SLRNET16_lr_0.001','.mat');
files = database.cname;
% load('name.mat')
y_step = 10;
x_step = 10;
dh = 50;
dw = 50;
m = 300;
n = 300;
% C = zeros(m, n);

i_file2 = 1;
for i_file=1:length(files)
    index = 0;
    T = load(['J:\paper\trans\trans6\医学\SLR-Net-master3\results\stable\NUDT_PART\',files{i_file}]);
    fields_T = cell2mat(fieldnames(T));
    T2 = eval(['T.' fields_T]);
    A1 = T2';

    C = zeros(m, n);
    temp = zeros(dh, dw);
    A_hat = zeros(300,300);
    AA = zeros(300, 300, 100);
%     figure,imshow(A1',[])
    for i = 1:y_step:m-dh+1
        for j = 1:x_step:n-dw+1
            index = 1+index;
            temp(:) = A1(:, index);
            C(i:i+dh-1, j:j+dw-1) = C(i:i+dh-1, j:j+dw-1)+1;
            for ii = i:i+dh-1
                for jj = j:j+dw-1
                    AA(ii,jj, C(ii,jj)) = temp(ii-i+1, jj-j+1);
                end
            end
        end
    end
%     figure,imshow(AA(:,:,10),[])
    %     C(find(C==0)) = 1000;
    for i=1:m
        for j=1:n
            if C(i,j) > 0
                A_hat(i,j) = median(AA(i,j,1:C(i,j)));
            end
        end
    end
    file_name = files{i_file};
    img_result = uint8(guiyihua(A_hat)*255);
    aa = file_name(1:end-4);
    

%     num2 = str2num(aa(3:end));
%     file_name2 = files_name{i_file2};
%     i_file2 = i_file2+1;
%     if i_file2 > length(files)/2
%         i_file2 = 1;
%     end    
    

%     img_result(img_result==255)=0;
%     figure,imshow(A_hat,[])
    imwrite(img_result, ['NUDT_PART/', aa, '.png'])
    fprintf('%d/%d: %s\n', length(files), i_file, files{i_file});
    clear T A1 temp C AA A_hat
end

