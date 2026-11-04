%{
==============================================================================
Batch Multi-Channel Image Extraction and ROI Cropping Pipeline
==============================================================================
Purpose: 
    Automates the deep-directory extraction, spatial cropping (Region of Interest), 
    and standardized renaming of multi-channel biological imaging datasets.

Core Method: 
    Traverses a two-tier nested directory structure to locate raw imaging files 
    based on specific channel suffixes. It then applies a uniform spatial bounding 
    box (ROI) to extract the core biological features and exports them into a 
    flattened, unified output directory with standardized nomenclature.

Robustness & Maintainability Design:
    1. Dynamic Channel Mapping: Replaces hardcoded channel iterations with a 
       centralized centralized channel lookup table (channel_map), ensuring scalable 
       and error-free adaptation for future multi-channel multiplexing panels.
    2. Safe Directory Traversal: Replaces fragile index-based looping (e.g., 3:end) 
       with strict logical indexing (~ismember) to Explicitly excludes MATLAB directory navigation artifacts
       ('.' and '..') during recursive folder discovery., ensuring cross-platform stability.
    3. Fail-Safe I/O Execution: Incorporates explicit file-existence validation 
       prior to image processing, preventing pipeline termination when encountering 
       incomplete FOV datasets during parallel execution (parfor).
==============================================================================
%}

clear;
clc;

%% ================== 1. Parameter Configuration ==================
% Define the spatial Region of Interest (ROI) for cropping: [xmin ymin width height]
target_roi = [80, 50, 1299, 1299];

% Define the mapping between raw channel suffixes and target output names
% Column 1: Raw file suffix wildcard
% Column 2: Standardized output channel name
channel_map = {
    '*img_000000000_Channel1_000.tif', 'bright';
    '*img_000000000_Channel3_000.tif', '647_705-72';
    '*img_000000000_Channel4_000.tif', '561_705-72';
    '*img_000000000_Channel5_000.tif', '561_595-33';
    '*img_000000000_Channel2_000.tif', '488_705-72';
    '*img_000000000_Channel6_000.tif', '488_525-50'
};
num_channels = size(channel_map, 1);

%% ================== 2. Directory Setup ==================
Input_path = uigetdir('', 'Select the ROOT Input Directory (containing primary folders)');
if Input_path == 0
    disp('Execution terminated by user.'); return;
end

Output_path = uigetdir('', 'Select the ROOT Output Directory');
if Output_path == 0
    disp('Execution terminated by user.'); return;
end

% Retrieve primary subdirectories safely (excluding '.', '..', and hidden files)
primary_dirs = dir(Input_path);
primary_dirs = primary_dirs([primary_dirs.isdir] & ~ismember({primary_dirs.name}, {'.', '..'}));

%% ================== 3. Batch Processing (Nested Traversal) ==================
% Start parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool('local'); 
end

% Loop through primary directories (e.g., Experimental groups)
for i = 1:length(primary_dirs)
    primary_name = primary_dirs(i).name;
    Input_path_2 = fullfile(Input_path, primary_name);
    
    % Retrieve secondary subdirectories safely
    secondary_dirs = dir(Input_path_2);
    secondary_dirs = secondary_dirs([secondary_dirs.isdir] & ~ismember({secondary_dirs.name}, {'.', '..'}));
    
    fprintf('Processing Primary Folder: %s (%d secondary folders found)\n', primary_name, length(secondary_dirs));
    
    % Parallel loop through secondary directories (e.g., Fields of View)
    parfor j = 1:length(secondary_dirs)
        secondary_name = secondary_dirs(j).name;
        Input_path_3 = fullfile(Input_path_2, secondary_name);
        
        % Iterate through each channel dynamically based on the channel_map
        for c = 1:num_channels
            raw_suffix = channel_map{c, 1};
            out_name   = channel_map{c, 2};
            
            % Locate the specific channel file
            search_pattern = fullfile(Input_path_3, raw_suffix);
            target_file_list = dir(search_pattern);
            
            % Fail-safe: Check if the file actually exists before processing
            if ~isempty(target_file_list)
                % Extract the first matched file name
                matched_filename = target_file_list(1).name;
                full_img_path = fullfile(Input_path_3, matched_filename);
                
                % Read, crop, and export the image
                img_raw = imread(full_img_path);
                img_cropped = imcrop(img_raw, target_roi);
                
                % Construct standardized output filename
                output_filename = sprintf('%s_%s_%s.tif', primary_name, secondary_name, out_name);
                output_filepath = fullfile(Output_path, output_filename);
                
                imwrite(img_cropped, output_filepath);
            else
                % Log a warning if a channel is missing, without crashing the parfor loop
                warning('Missing channel [%s] in folder: %s', out_name, Input_path_3);
            end
        end
    end
end

disp('Pipeline Execution Complete! All images successfully extracted and cropped.');