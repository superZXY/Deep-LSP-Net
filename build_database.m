function database = build_database(rt_data_dir,suffix)
% This function is to build a database for the image sets 
% Input:  rt_data_dir -- direction of image sets
%         suffix      -- image format like 'jpg'
% Output: database    -- database that contains all the information of
%                        images
 
% Written by Wei Q
% July. 16, 2013
 
fprintf('dir the database...');
subfolders = dir(rt_data_dir);   
 
database = [];
 
database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;
 
for ii = 3:length(subfolders)
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..')
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        frames = dir(fullfile(rt_data_dir, subname, suffix));
        c_num = length(frames);
                    
        database.imnum = database.imnum + c_num;
        database.label = [database.label; ones(c_num, 1)*database.nclass];
        
        for jj = 1:c_num,
            c_path = fullfile(rt_data_dir, subname, frames(jj).name);
            database.path = [database.path, c_path];
        end;    
    end;
end;
disp('done!');
