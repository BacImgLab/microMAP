%{
==============================================================================
Multi-Round Multiplexed Image Background Subtraction Pipeline
==============================================================================
Purpose: 
    Automated and quantitative background subtraction for multi-round biological 
    multiplexed imaging (e.g., stripping off auto-fluorescence and non-specific 
    binding noise).

Core Method: 
    Extracts a reference background matrix from a designated terminal imaging 
    round (e.g., Round 11 or 12) and linearly subtracts it from all preceding 
    rounds across respective channels.

Key Engineering Tricks:
    1. Signal Preservation:Background-corrected images are exported as uint16 TIFF files,
       with negative corrected intensities constrained to zero.
       underflow/overflow artifacts and preserving true signal integrity.
    2. Hardware-Aware Parallelism: Employs isolated local memory allocation 
       within a `parfor` loop, preventing thread collisions and enabling 
       high-throughput, memory-safe batch processing of massive FOV datasets.
==============================================================================
%}

clear;
clc;

%% ================== 1. Parameter Configuration ==================
num_pos = 1:572;            % Array of Field of View (FOV) indices
num_round = 1:11;           % Array of total experimental imaging rounds
bkg_round_idx = 11;         % Index of the round utilized as the baseline background
num_channel = 1:6;          % Channels 1-5: Fluorescence; Channel 6: Brightfield

file_prefix = 'registered_SPF_colon_';
file_suffix = '.tif';
channel_name = {'647_705-72', '561_705-72', '561_595-33', '488_705-72', '488_525-50', 'bright'};

% --- Background Subtraction Ratio Parameters ---
% Multiplier applied per specific fluorescent channel
bkgChannelRatio = [1, 1, 1, 1, 1]; 
% Multiplier applied per temporal round (length must equal the total number of rounds)
bkgRoundRatio = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]; 

%% ================== 2. Directory Preparation ==================
% Prompt user to select the source directory containing registered images
input_folder = uigetdir('', 'Select the input directory containing registered TIFFs');
if input_folder == 0
    disp('User canceled directory selection. Execution terminated.');
    return;
end

% Construct output directory dynamically based on input path
clean_input_folder = regexprep(input_folder, '[\\/]+$', '');
[parent_dir, folder_name, folder_ext] = fileparts(clean_input_folder); 

output_folder_name = strcat([folder_name, folder_ext], '_SUB_Background'); 
output_folder = fullfile(parent_dir, output_folder_name);

if ~exist(output_folder, 'dir')
     mkdir(output_folder); 
else
     disp(['Output directory already exists: ', output_folder]);
end

% 2.1 Save Background Correction Parameters 
parameter_file = fullfile(output_folder,'Background_parameters.csv');

fid = fopen(parameter_file,'w');

fprintf(fid,'Parameter,Value\n');
fprintf(fid,'Background_round,%d\n', bkg_round_idx);

fprintf(fid,'\nChannel,Ratio\n');

for i = 1:length(bkgChannelRatio)
    fprintf(fid,'Channel_%d,%.4f\n', i, bkgChannelRatio(i));
end

fprintf(fid,'\nRound,Ratio\n');

for i = 1:length(bkgRoundRatio)
    fprintf(fid,'Round_%d,%.4f\n', i, bkgRoundRatio(i));
end

fclose(fid);
%% ================== 3. Image Dimension Extraction ==================
% Retrieve matrix dimensions (Height x Width) using Round 1, FOV 1, Brightfield image
ref_imgname = strcat(file_prefix, num2str(num_round(1)-1), '_Pos', num2str(num_pos(1)), '_', channel_name{6}, file_suffix);
ref_img = imread(fullfile(input_folder, ref_imgname));
[height, width] = size(ref_img);

%% ================== 4. Parallel Processing ==================
if isempty(gcp('nocreate'))
    parpool('local'); 
end

parfor pos_idx = 1:length(num_pos)
    pos = num_pos(pos_idx);
    
    % Pre-allocate local memory for each parallel worker to avoid thread collision
    local_Pos_imgs = cell(1, length(num_channel));
    local_img_names = cell(1, length(num_channel));
    
    for i = 1:length(num_channel)
        % Initialize 3D matrices (Height x Width x Rounds) in uint16 precision
        local_Pos_imgs{i} = zeros(height, width, length(num_round), 'uint16');
        local_img_names{i} = cell(1, length(num_round));
    end
    
    % Load all rounds and channels for the current FOV
    for ch = 1:length(num_channel)
        for round_idx = 1:length(num_round)
            round_num = num_round(round_idx);
            imgname = strcat(file_prefix, num2str(round_num-1), '_Pos', num2str(pos), '_', channel_name{ch}, file_suffix);
            
            img_path = fullfile(input_folder, imgname);
            if exist(img_path, 'file')
                local_Pos_imgs{ch}(:,:,round_idx) = imread(img_path);
            else
                warning('File not found: %s', img_path);
            end
            
            local_img_names{ch}{round_idx} = imgname;
        end
    end

    % Execute batch background subtraction via local function
    processed_cells = BatchBkgSubtract(bkgChannelRatio, bkgRoundRatio, bkg_round_idx, local_Pos_imgs{6}, ...
                             local_Pos_imgs{1}, local_Pos_imgs{2}, local_Pos_imgs{3}, local_Pos_imgs{4}, local_Pos_imgs{5});

    % Export processed images (Exclude brightfield channel (index 6) from rewriting)
    for ch = 1:5 
        for round_idx = 1:length(num_round)
            output_filename = fullfile(output_folder, local_img_names{ch}{round_idx});
            imwrite(processed_cells{ch}(:,:,round_idx), output_filename);
        end
    end
    
    % Log progress status
    fprintf('Successfully processed FOV %d / %d\n', pos_idx, length(num_pos));
end

disp('Pipeline Execution Complete! Background subtraction finished for all FOVs.');

%% ================== 5. Local Functions ==================
function cell_out = BatchBkgSubtract(bkgChannelRatio, bkgRoundRatio, bkg_idx, PhaseImage, varargin)
    % Extracts background matrices and subtracts them from all target rounds
    
    ImageOutNum = size(varargin, 2); 
    cell_out = cell(1, ImageOutNum + 1);

    % Load dynamic channel arguments into a structured cell array
    for i = 1:ImageOutNum
        cell_out{i} = varargin{i};
    end

    % --- Background Matrix Extraction ---
    % Note: 647_705 channel baseline is strictly set to 100 based on empirical parameters.
    % To use image-based background, replace 100 with cell_out{1}(:, :, bkg_idx).
    ch647_705bkg = 100; 
    ch561_705bkg = cell_out{2}(:, :, bkg_idx);
    ch561_595bkg = cell_out{3}(:, :, bkg_idx);
    ch488_705bkg = cell_out{4}(:, :, bkg_idx);
    ch488_525bkg = cell_out{5}(:, :, bkg_idx);
    
    bkgall = {ch647_705bkg, ch561_705bkg, ch561_595bkg, ch488_705bkg, ch488_525bkg};

    % --- Execution of Matrix Subtraction ---
    % Iterates through all rounds except the designated background round itself
    for k = 1:size(PhaseImage, 3) - 1        
        if ImageOutNum > 0          
            for outnum = 1:ImageOutNum
                % Matrix subtraction utilizing uint16 properties (auto-clamps < 0 to 0)
                cell_out{outnum}(:, :, k) = cell_out{outnum}(:, :, k) - ...
                    bkgRoundRatio(k) * bkgChannelRatio(outnum) * bkgall{outnum};
            end
        end
    end
    
    % Append the unaltered Brightfield image to the final cell output
    cell_out{6} = PhaseImage;
end