%%
%{
==============================================================================
Decoded Strain Image Extraction and Standardization Pipeline
==============================================================================

Purpose:
    Extracts decoded strain identity images generated from multiplexed FISH
    barcode decoding and converts them into standardized spatial layers for
    downstream multi-FOV stitching.

Core Method:
    1. Automatically detects decoded Position folders.
    2. Converts strain-specific decoded images into channel-based filenames.
    3. Preserves unknown populations and visualization outputs.
    4. Generates standardized stacked_Pos indexed files compatible with the
       stitching pipeline.

Robustness & Memory Design:
    1. File-level processing:Images are copied individually without loading 
       complete datasets.
    2. Regex-based classification:Automatically identifies decoded strain 
       channels and special outputs.
==============================================================================
%}


clear; clc;


% ================== 1. Directory Configuration ==================

input_folder = uigetdir('', 'Select decoded image folder');

if input_folder == 0
    error('No input folder selected.');
end

input_folder = regexprep(input_folder,'[\\/]+$','');
[parent_dir, folder_name, folder_ext] = fileparts(input_folder);
folder_name = [folder_name folder_ext];

output_folder = fullfile(parent_dir,[folder_name '_stacked']);

if ~exist(output_folder,'dir')
    mkdir(output_folder);
end

fprintf('Input folder: %s\n',input_folder);
fprintf('Output folder: %s\n',output_folder);

% ================== 2. Position Folder Detection ==================

pos_dirs = dir(fullfile(input_folder,'Pos*'));
pos_dirs = pos_dirs([pos_dirs.isdir]);

if isempty(pos_dirs)
    error('No Position folders detected.');
end


fprintf('Detected %d positions.\n',length(pos_dirs));


% ================== 3. Decoded Image Standardization ==================

parfor p = 1:length(pos_dirs)
    pos_name = pos_dirs(p).name;
    pos_token = regexp(pos_name,'Pos(\d+)','tokens','once');

    if isempty(pos_token)
        warning('Unable to identify position: %s',pos_name);
        continue;
    end

    pos_number = pos_token{1};
    tif_files = dir(fullfile(input_folder,pos_name,'*.tif'));
    filenames = {tif_files.name};
    fprintf('\nProcessing %s (%d files)\n',pos_name,length(filenames));

    for i = 1:length(filenames)
        old_name = filenames{i};
        new_name = "";
        % Convert decoded strain images into channel format
        strain_token = regexp(old_name,'StrainImage(\d+)_Strain\d+\.tif','tokens','once');
        
        if ~isempty(strain_token)
            channel_id = str2double(strain_token{1}) - 1;
            new_name = sprintf('stacked_Pos%s_channel_%d.tif',pos_number,channel_id);
        elseif contains(old_name,'Unknown')
            new_name = sprintf('stacked_Pos%s_Unknown.tif',pos_number);
        elseif strcmp(old_name,'StrainImageAll.tif')
            new_name = sprintf('stacked_Pos%s_StrainImageAll.tif',pos_number);
        elseif strcmp(old_name,'ColorStrainImageAll.tif')
            new_name = sprintf('stacked_Pos%s_ColorStrainImageAll.tif',pos_number);
        end


        if strlength(new_name) > 0
            copyfile(fullfile(input_folder,pos_name,old_name),...
                fullfile(output_folder,new_name));
            fprintf('Exported: %s\n',new_name);
        else
            fprintf('Skipped: %s\n',old_name);
        end
    end
end

fprintf('Decoded image extraction completed successfully.\n');
fprintf('Output saved at:\n%s\n',output_folder);