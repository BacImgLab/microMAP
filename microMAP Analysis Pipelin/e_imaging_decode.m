%% 
%{
==============================================================================
microMAP Multiplexed Signal Decoding and Quantification Pipeline
==============================================================================
Purpose: 
    Automates the quantitative identification, spatial mapping, and decoding of 
    fluorescent signals from multi-round FISH (Fluorescence In Situ Hybridization) 
    images. 

Core Method:
    1. Extracts morphological masks (Ground Truth) and crops to defined ROIs.
    2. Employs statistical hypothesis testing (Welch's t-test) on single-cell 
       pixel intensities vs. local backgrounds to rigorously validate true signals.
    3. Decodes multiplexed combinatorial barcodes against a predefined Codebook 
       incorporating Hamming-distance error correction.
==============================================================================
%}

clear
clc
tic;

% ================== 1. Parameter Configuration ==================
% Define ROI coordinates (obtained via ImageJ bounding box).
x = 0;
y = 0;
w = 1260; % ROI Width (X-axis)
h = 1260; % ROI Height (Y-axis)
% Decouple explicit height and width metrics
M = h; % Image height
N = w; % Image width
row_start = y + 1;
col_start = x + 1;
h_patch_size = h;
w_patch_size = w;

% Define the biological target scope
ActualStrainNum = 11; % Number of specific bacterial strains targeted (Max <= 96)

% Signal-to-Noise Ratio (SNR) Thresholds & Statistical Parameters
BkgNoise = 20 ; % Baseline ambient optical noise 
global_bkg_ratio = 2.8; % Global Threshold: True signal must exceed x the global background mode.
local_bkg_ratio = 2.0; % Local Threshold: True signal must exceed y the immediate surrounding local background.
CorrBit = 4;  % Error correction capacity (Hamming distance threshold)

% Specific correction multiplier for Fluorophore 2 (ATTO490) based on empirical imaging efficiency
if 1.3 > global_bkg_ratio
    Fluo2_correct_parameters = global_bkg_ratio;
else
    Fluo2_correct_parameters = 1.3;
end

bkg_box = 50; % Local background bounding box expansion (in pixels) for single-cell sampling
num_channel = 5; % Total fluorescence channels utilized

% --- Dynamic Round Configuration ---
Index = 1:11; % Iteration indices corresponding to actual rounds processed
numImages = 11; % Total hybridization rounds
% -----------------------------------

% ================== 2. Codebook Initialization ==================
% Load the experimental combinatorial barcode codebook
% Load the experimental combinatorial barcode codebook
[cb_file, cb_path] = uigetfile({'*.xlsx;*.xls;*.csv', 'Codebook Files (*.xlsx, *.xls, *.csv)'}, 'Select the Barcode Codebook File');
if isequal(cb_file, 0); error('Codebook file not selected.'); end
dataTable = readtable(fullfile(cb_path, cb_file), 'ReadVariableNames', false);
StrainCode = table2array(dataTable(:, 2:(numImages + 1)))'; % Extract numeric barcode array and transpose
StrainCode = StrainCode(:, 1:ActualStrainNum);
disp(['Codebook dimension successfully loaded: ', num2str(size(StrainCode))]);
    
% Morphological filtering parameters
need_erode = false; % Toggle binary erosion for ground-truth masks
sePattern = strel('disk', 1); % Structuring element for erosion

% ================== 3. Directory Configuration ==================
input_folder = uigetdir('', 'Select the input directory containing imaging TIFFs');
if input_folder == 0; error('Input directory not selected.'); end

gt_folder = uigetdir('', 'Select the directory containing Ground Truth (GT) masks');
if gt_folder == 0; error('GT mask directory not selected.'); end

% Construct unique output directory parameterized by thresholds
[parent_dir, folder_name, ~] = fileparts(input_folder); % [Optimization] Avoid redundant strsplit
dateTimeStr2 = datestr(now, 'yyyymmdd'); 
output_folder_name = strcat(folder_name, '_results_', dateTimeStr2, '_', num2str(BkgNoise), 'bac_box_', num2str(bkg_box), '-G_', num2str(global_bkg_ratio), '-L_', num2str(local_bkg_ratio));
output_folder = fullfile(parent_dir, output_folder_name);

if ~exist(output_folder, 'dir')
     mkdir(output_folder); 
else
     disp('Output folder already exists.');
end
output_folderMain = output_folder;

tgtfilename = {'647_705-72', '561_705-72', '561_595-33', '488_705-72', '488_525-50', 'bright'};

% ================== 4. FOV (Field of View) Indexing & I/O Optimization ==================
% Parse all TIFF files to extract unique Pos (FOV) indices
files = dir(fullfile(input_folder, '*.tif'));
pos_list = [];

for i = 1:length(files)
    filename = files(i).name;
    match = regexp(filename, '_Pos(\d+)_', 'tokens');
    if ~isempty(match)
        pos = str2double(match{1}{1});
        pos_list(end+1) = pos;  
    end
end
 
% Deduplicate and sort FOV indices
pos_list = unique(pos_list);
pos_list = sort(pos_list);

% Read mask filenames ONCE before the parallel loop to prevent I/O bottleneck
png_files = dir(fullfile(gt_folder, '*.png'));
png_names = {png_files.name};

% Initialize statistical accumulation matrices
pos_num = length(pos_list);
bacteriaCountsall1 = zeros(1, ActualStrainNum+1);
bacteriaCountsall2 = zeros(pos_num, ActualStrainNum+1);
levelall1 = zeros(1, 3);
levelall2 = zeros(pos_num, 3);

% Configure Multi-core DataQueue for Clean Progress Bar
if isempty(gcp('nocreate'))
    parpool('local');
end

clear updateProgressBar; 
D = parallel.pool.DataQueue;
afterEach(D, @(~) updateProgressBar(pos_num)); 
disp('Starting parallel FOV processing...');

% ================== 5. Core Parallel Processing Loop ==================
parfor pos = 1:pos_num
    % Construct specific sub-directory for the current FOV
    output_folder = fullfile(output_folderMain, ['Pos', num2str(pos_list(pos))]);
    if ~exist(output_folder, 'dir')
         mkdir(output_folder); 
    end

    % Pre-allocate stack memory for all channels
    Stacks = cell(1, 5); 
    for ch = 1:5  % Process fluorescence channels 1 to 5
        currentStack = zeros(h_patch_size, w_patch_size, numImages, 'uint16');
        for round = 1:numImages  
            file_pattern = ['*_', num2str(round-1), '_Pos', num2str(pos_list(pos)), '_', tgtfilename{ch}, '.tif'];
            candidate_files = dir(fullfile(input_folder, file_pattern));
            if isempty(candidate_files)
                error('File not found: %s', file_pattern);
            else
                file_name = candidate_files(1).name;
                file_path = fullfile(input_folder, file_name);
            end

            % Read specific ROI to optimize RAM usage
            img = imread(file_path, 'PixelRegion', { ...
                [row_start, row_start + h_patch_size - 1], ...
                [col_start, col_start + w_patch_size - 1]});
            
            % Subtract baseline ambient noise
            currentStack(:, :, round) = img - BkgNoise;
        end
        Stacks{ch} = currentStack;
    end
    
    % --- Accurate Ground Truth Mask Matching ---
    current_pos_number = pos_list(pos); 
    mask_found = false;
    masks = []; % Predefine to avoid parfor transparency warnings
    
    for i = 1:length(png_names)
        % Rigorously extract Pos number to prevent regex mismatch
        match = regexp(png_names{i}, 'Pos(\d+)_', 'tokens');
        if ~isempty(match) && str2double(match{1}{1}) == current_pos_number
            mask_path = fullfile(gt_folder, png_names{i});
            masks = imread(mask_path);
            mask_found = true;
            break;
        end
    end
    
    if ~mask_found
        error('CRITICAL ERROR: Corresponding GT Mask for Pos%d not found!', current_pos_number);
    end
    
    % Crop mask to ROI and apply morphological erosion if required
    BW_Image_masks = masks(row_start:row_start+h_patch_size-1, col_start:col_start+w_patch_size-1);
    if need_erode == true
        BW_Image_Filtered = imerode(BW_Image_masks, sePattern);
    else
        BW_Image_Filtered = BW_Image_masks;
    end

    % Master Connected Components evaluation (Avoids recalculations)
    CC = bwconncomp(BW_Image_Filtered);

    % --- Global Background Estimation (Mode Pooling) ---
    ModePool = zeros(5, numImages);
    for ch = 1:5
        for r = 1:numImages
            [counts, grayLevels] = imhist(Stacks{ch}(:, :, r), 65536);
            [~, idx] = max(counts); 
            modeGrayLevel = grayLevels(idx);
            ModePool(ch, r) = modeGrayLevel;
        end
    end
    
    % Convert microscope acquisition order:
    % [Cy5,AF594,Cy3,ATTO490LS,AF488]
    % into barcode decoding order:
    % [AF488,ATTO490,Cy3,AF594,CY5]
    ModePool_M = zeros(5, numImages);
    ModePool_M(1, :) = ModePool(5, :); 
    ModePool_M(2, :) = ModePool(4, :); 
    ModePool_M(3, :) = ModePool(3, :); 
    ModePool_M(4, :) = ModePool(2, :); 
    ModePool_M(5, :) = ModePool(1, :); 
    ModePool1 = ModePool_M;
    ModePool_M(2, :) = uint8(ModePool_M(2, :)/Fluo2_correct_parameters);  
    ModePool2 = ModePool_M;
    
    fluo_name = ["F1-AF488_488 525", "F2-ATTO490LS_488 705", "F3-Cy3_561 595", "F4-AF594_561 705", "F5-Cy5_647 705"];
    
    % Export Background Mode metrics dynamically based on numImages
    output_file_path = fullfile(output_folder, 'Mode.csv'); 
    fid = fopen(output_file_path, 'w'); 
    if fid == -1; error('File Lock Error: Close Excel or other programs using %s', output_file_path); end
    
    fprintf(fid, 'Mode');
    for r = 1:numImages; fprintf(fid, ',Round%d', r); end
    fprintf(fid, '\n');
    
    for c = 1:5
        fprintf(fid, fluo_name{c});
        for i = 1:numImages
            fprintf(fid, ', %.3f', ModePool1(c, i));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    % Generate connected component labels from cached CC
    BW_labels = labelmatrix(CC);
    filename = 'BW_labels.tif'; 
    fullpath = fullfile(output_folder, filename); 
    imwrite(BW_labels, fullpath); 
    
    % ================== 6. Single-Cell Spatial & Statistical Analysis ==================
    CellRegion = regionprops(CC, 'Area'); % [Optimization] Fast regionprops using cached CC
    HypAllCell = cell(numImages, 1);
    ProbAllCell = cell(numImages, 1);
    IntenAllCell = cell(numImages, 1);
    FluorescentHyp = zeros(num_channel, size(CellRegion, 1));
    FluorescentProb = zeros(num_channel, size(CellRegion, 1));
    CodexRes = zeros(size(CellRegion, 1), numImages);
    IntensityColorAll = zeros(num_channel, size(CellRegion, 1));
    
    Fluorophore = cell(1, 5);
    for Round = 1:numImages

        % Channel realignment for spatial mapping
        Fluorophore{1} = Stacks{5}(:, :, Index(Round)); % Code 1: AF488
        Fluorophore{2} = Stacks{4}(:, :, Index(Round)); % Code 2: ATTO490LS
        Fluorophore{3} = Stacks{3}(:, :, Index(Round)); % Code 3: Cy3
        Fluorophore{4} = Stacks{2}(:, :, Index(Round)); % Code 4: AF594
        Fluorophore{5} = Stacks{1}(:, :, Index(Round)); % Code 5: Cy5

        for C = 1:5
            CellRegion = regionprops(CC, Fluorophore{C}, 'Area', 'BoundingBox', 'PixelValues', 'MeanIntensity');
            CellRegionCell = struct2cell(CellRegion);
            CellRegionBoundingBox = CellRegionCell(2, :);  
            
            for k = 1:size(CellRegion, 1)
                % Define local background bounded box limits safely
                CellRegionBoxX = max(floor(CellRegionBoundingBox{k}(1)) - bkg_box, 1):min(floor(CellRegionBoundingBox{k}(1)) + CellRegionBoundingBox{k}(3) + bkg_box, size(Fluorophore{C}, 2));
                CellRegionBoxY = max(floor(CellRegionBoundingBox{k}(2)) - bkg_box, 1):min(floor(CellRegionBoundingBox{k}(2)) + CellRegionBoundingBox{k}(4) + bkg_box, size(Fluorophore{C}, 1));
                
                % Extract signal pixels and local background pixels dynamically
                ChannelImageCut = Fluorophore{C}(CellRegionBoxY, CellRegionBoxX);
                BW_ImageCut = BW_Image_Filtered(CellRegionBoxY, CellRegionBoxX);
                BackGroundCut = ChannelImageCut(~BW_ImageCut);
                
                % Compute background-subtracted absolute mean intensity
                IntensityColorAll(C, k) = cell2mat(CellRegionCell(4, k)) - ModePool1(C, Round); 
                
                dataCell = double(CellRegionCell{3, k}); 
                dataBackground = double(BackGroundCut);
                
                % --- Rigorous Statistical Validation (Welch's t-test) ---
                if numel(dataCell) >= 2 && numel(dataBackground) >= 2 && var(dataCell) ~= 0 && var(dataBackground) ~= 0  && mean(dataCell) > global_bkg_ratio*ModePool2(C, Round) && mean(dataCell)>local_bkg_ratio*mean(dataBackground)
                    [h, p] = ttest2(dataCell, dataBackground, 'Tail', 'right', 'Alpha', 0.001, 'Vartype', 'unequal');
                else
                    h = 0; % Null hypothesis retained
                    p = 1;
                end
                
                FluorescentHyp(C, k) = h;
                FluorescentProb(C, k) = p; 
            end
        end
        
        % Collate statistics for output
        HypAllCell{Round} = FluorescentHyp';
        ProbAllCell{Round} = FluorescentProb';
        IntenAllCell{Round} = IntensityColorAll';

        % --- Standard Round Signal Logic (Applied Uniformly to All Rounds) ---
        NonSignalIndex = min(FluorescentProb(:, :), [], 1) > 0.001;
        ProbIndex1 = sum(FluorescentHyp(:, :), 1) == 1;
        ProbIndex2 = sum(FluorescentHyp(:, :) , 1) == 2;
        ProbIndex2_magenta = false(1, size(FluorescentHyp(:, :), 2));
        ProbIndex2_yellow = false(1, size(FluorescentHyp(:, :), 2));

        ProbIndex = ~NonSignalIndex & ~ProbIndex1 & ~ProbIndex2;
            for colIndex = 1:size(FluorescentHyp(:, :), 2)
                if ProbIndex2(colIndex)
                    if FluorescentHyp(3, colIndex) == 1 && FluorescentHyp(4, colIndex) == 1 
                        if (IntensityColorAll(4, colIndex)/IntensityColorAll(3, colIndex)) >= 0.3  && (IntensityColorAll(4, colIndex)/IntensityColorAll(3, colIndex)) <= 1.2 
                            ProbIndex2_magenta(colIndex) = true;
                        else
                            ProbIndex2_yellow(colIndex) = true;
                        end
                    end
                end   
            end
        
        [ProbColor1, ~] = find(FluorescentHyp(:, ProbIndex1) == 1);
        [ProbColor2_magenta,~] = find(FluorescentHyp(:, ProbIndex2_magenta) == 1);
        [ProbColor2_yellow,~] = find(FluorescentHyp(:, ProbIndex2_yellow) == 1);
        ProbIndex2_other = ProbIndex2 & ~ProbIndex2_magenta & ~ProbIndex2_yellow;
        [~, ProbColor2_other] = max(IntensityColorAll(:, ProbIndex2_other), [], 1);

        [high_intensity,~] = find(IntensityColorAll(:,:) == max(IntensityColorAll(:,:), [], 1));
        
        CodexResRound = zeros(size(CellRegion, 1), 1);
        CodexResRound(NonSignalIndex) = 0;
        CodexResRound(ProbIndex1) = ProbColor1;
        CodexResRound(ProbIndex2_magenta) = ProbColor2_magenta(2:2:end);
        CodexResRound(ProbIndex2_yellow) = ProbColor2_yellow(1:2:end);
        CodexResRound(ProbIndex2_other) = ProbColor2_other;
        CodexResRound(ProbIndex) = high_intensity(ProbIndex);
        CodexResRound(NonSignalIndex) = 0;
        CodexRes(:, Round) = CodexResRound;
    end
    
    % ================== 7. Single-Cell Data Export (Dynamic Headers & Safeguarded I/O) ==================
    % Export intensity metrics
    output_file_path = fullfile(output_folder, 'IntensityAllCell.csv'); 
    fid = fopen(output_file_path, 'w'); 
    if fid == -1; error('File Lock Error: Close Excel or other programs using %s', output_file_path); end
    for k = 1:size(CodexRes, 1)
        fprintf(fid, 'Cell%d', k);
        for r = 1:numImages; fprintf(fid, ',Round%d', r); end
        fprintf(fid, '\n'); 
        for c = 1:5
            fprintf(fid, '%s', fluo_name(c));
            for i = 1:numImages
                fprintf(fid, ', %.3f', IntenAllCell{i}(k, c));
            end
            fprintf(fid, '\n');
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    % Export hypothesis matrix
    output_file_path = fullfile(output_folder, 'HypAllCell.csv'); 
    fid = fopen(output_file_path, 'w'); 
    if fid == -1; error('File Lock Error: %s', output_file_path); end
    for k = 1:size(CodexRes, 1)
        fprintf(fid, 'Cell%d', k);
        for r = 1:numImages; fprintf(fid, ',Round%d', r); end
        fprintf(fid, '\n');
        for c = 1:5
            fprintf(fid, '%s', fluo_name(c));
            for i = 1:numImages
                fprintf(fid, ', %d', HypAllCell{i}(k, c));
            end
            fprintf(fid, '\n');
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    % Export p-value probabilities
    output_file_path = fullfile(output_folder, 'ProbAllCell.csv'); 
    fid = fopen(output_file_path, 'w'); 
    if fid == -1; error('File Lock Error: %s', output_file_path); end
    for k = 1:size(CodexRes, 1)
        fprintf(fid, 'Cell%d', k);
        for r = 1:numImages; fprintf(fid, ',Round%d', r); end
        fprintf(fid, '\n');
        for c = 1:5
            fprintf(fid, '%s', fluo_name(c));
            for i = 1:numImages
                fprintf(fid, ', %.3f', ProbAllCell{i}(k, c));
            end
            fprintf(fid, '\n');
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    % Export final resolved barcode sequences
    output_file_path = fullfile(output_folder, 'Acquired_code.txt');
    fid = fopen(output_file_path, 'w'); 
    if fid == -1; error('File Lock Error: %s', output_file_path); end
    fprintf(fid,'Rows: Individual Cells (Bacteria) | Columns: Resolved Barcode Sequence Array\n');
    for i = 1:size(CodexRes, 1)
        fprintf(fid, '%d ', CodexRes(i, :));
        fprintf(fid, '\n');
    end
    fclose(fid);

    % ================== 8. Codebook Matching & Decoding ==================
    [StrainLikehood, Decode, level] = StrainIdentify(StrainCode, CodexRes, CorrBit);
    
    % Export Error-Correction Metrics
    output_file_path1 = fullfile(output_folder, 'StrainLikehood.txt');
    fid = fopen(output_file_path1, 'w');
    if fid == -1; error('File Lock Error: %s', output_file_path1); end
    fprintf(fid,'Codebook Matching Likelihood Score (Max: %d)\n', numImages);
    for i = 1:size(StrainLikehood, 1)
        fprintf(fid, '%d ', StrainLikehood(i, :));
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    output_file_path2 = fullfile(output_folder, 'Decode.txt');
    fid = fopen(output_file_path2, 'w'); 
    if fid == -1; error('File Lock Error: %s', output_file_path2); end
    fprintf(fid,'Column 1: Identified Strain ID | Column 2: Max Likelihood Score\n');
    for i = 1:size(Decode, 1)
        fprintf(fid, '%d  ', Decode(i, :));
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    output_file_path2 = fullfile(output_folder, 'Decode.csv');
    fid = fopen(output_file_path2, 'w'); 
    if fid == -1; error('File Lock Error: %s', output_file_path2); end
    for i = 1:size(Decode, 1)
        fprintf(fid, '%d  ', Decode(i, 1));
        fprintf(fid, ',%d  ', Decode(i, 2));
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    % Comprehensive Integration Export
    output_file_path3 = fullfile(output_folder, 'All_AcqCode&Decode.txt');
    fid = fopen(output_file_path3, 'w'); 
    if fid == -1; error('File Lock Error: %s', output_file_path3); end
    fprintf(fid,'     Acquired Barcode       Identified Strain / Max Score          Codebook Likelihood Scores\n');
    for i = 1:size(Decode, 1)
        fprintf(fid, 'Cell%d: ', i);
        fprintf(fid, '%d', CodexRes(i, :));
        fprintf(fid, '                     ');
        fprintf(fid, '%d   ', Decode(i, :));
        fprintf(fid, '                     ');
        fprintf(fid, '%d ', StrainLikehood(i, :));
        fprintf(fid, '\n');
    end
    fclose(fid);
    
    % ================== 9. Visualization & Segmentation Labeling ==================
    h_steps = 16;   
    s_steps = 3;    
    v_steps = 2;    
    
    h_val = linspace(0, 1, h_steps + 1)'; h_val = h_val(1:end-1);  
    s_val = linspace(0.4, 1, s_steps)';                    
    v_val = linspace(0.5, 1, v_steps)';                    
    [H, S, V] = meshgrid(h_val, s_val, v_val);
    
    HSV = [H(:), S(:), V(:)];
    RGB = hsv2rgb(HSV);
    
    RGB = RGB(1:ActualStrainNum,:);
    colors = [RGB; 0.8, 0.8, 0.8];
    
    bacteriaName = arrayfun(@(x) sprintf('Strain%d', x), 1:96, 'UniformOutput', false)';
    bacteriaName = [bacteriaName(1:ActualStrainNum); {'Unknown'}];
    
    [StrainImageAll, bacteriaCounts] = LabelImage1(CC, Decode, bacteriaName, output_folder, size(BW_Image_Filtered));
    
    output_file_path = fullfile(output_folder, 'bacteriaCounts.csv'); 
    fid = fopen(output_file_path, 'w'); 
    if fid == -1; error('File Lock Error: %s', output_file_path); end
    fprintf(fid, 'Strain, counts\n'); 
    for k = 1:size(bacteriaCounts, 2)
        fprintf(fid, '%s, %d\n', bacteriaName{k}, bacteriaCounts(k));
    end
    fprintf(fid, '\n');
    fprintf(fid, 'PerfectMatch,One2ThreeMisMatch,Four2FiveMisMatch\n'); 
    fprintf(fid, '%d, %d, %d\n', level(1,1), level(1,2), level(1,3));
    fclose(fid);
    
    % Update local arrays
    bacteriaCountsall1 = bacteriaCountsall1 + bacteriaCounts;
    bacteriaCountsall2(pos,:) = bacteriaCounts;
    levelall1 = levelall1 + level;
    levelall2(pos,:) = level;
    
    background_color = 'k'; 
    ColorStrainImageAll = label2rgb(StrainImageAll, colors, background_color);
    
    fontsize = floor(N*0.01);
    filename = 'ColorStrainImageAll.tif';
    fullpath = fullfile(output_folder, filename);
    imwrite(ColorStrainImageAll, fullpath);
    
    right_width1 = floor(3*N / 8);
    canvas = zeros(M, N+right_width1, 3);
    canvas(:, N+1:end, :) = 1; 
    text_len = floor(0.3*right_width1);
    
    color_block_height = floor(M / numel(bacteriaName));
    border_thickness = 6; 
    for i = 1:numel(bacteriaName)
        start_row = (i-1)*color_block_height+1;
        end_row = i*color_block_height;
        canvas(start_row:end_row, N+1:N+text_len, :) = repmat(reshape(colors(i, :), 1, 1, 3), color_block_height, text_len);
    
        if start_row > border_thickness
            canvas(start_row-border_thickness:start_row-1, N+1:N+text_len, :) = 0;
        end
        if end_row + border_thickness <= size(canvas, 1)
            canvas(end_row+1:end_row+border_thickness, N+1:N+text_len, :) = 0;
        end
        canvas(start_row:end_row, N+1:N+1+border_thickness-1, :) = 0;
        canvas(start_row:end_row, N+text_len-border_thickness+1:N+text_len, :) = 0;
        
        position = [(N + text_len + 5), ((i-1)*color_block_height)];
        canvas = insertText(canvas, position, bacteriaName{i}, 'FontSize', fontsize, 'BoxOpacity', 0, 'TextColor', colors(i, :), 'Font', 'Arial Italic');
    end
    
    if ~isa(canvas, 'uint8')
        canvas = uint8(canvas * 255);
    end
    canvas(:, 1:N, :) = ColorStrainImageAll; 
    
    final_filename = 'FinalColorStrainImageAll.tif';
    final_fullpath = fullfile(output_folder, final_filename);
    imwrite(canvas, final_fullpath);
    
    % --- Notify DataQueue that one FOV is completed ---
    send(D, pos);
end

%% ================== 10. Global Post-Processing & Accumulation ==================
disp('Aggregating global statistics across all FOVs...');
bacteriaName = arrayfun(@(x) sprintf('Strain%d', x), 1:96, 'UniformOutput', false)';
bacteriaName = [bacteriaName(1:ActualStrainNum); {'Unknown'}];

subfolders = dir(fullfile(output_folderMain, 'Pos*'));
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'}));
subfolderNames = {subfolders.name};
numbers = cellfun(@(x) sscanf(x, 'Pos%d'), subfolderNames);
[~, sortedIdx] = sort(numbers); 
sortedSubfolderNames = subfolderNames(sortedIdx);
sortedNumbers = numbers(sortedIdx);  
n = length(sortedSubfolderNames);

% Re-initialize master variables for post-processing safety
bacteriaCountsall1 = zeros(1, ActualStrainNum+1);
bacteriaCountsall2 = zeros(n, ActualStrainNum+1);
levelall1 = zeros(1, 3);
levelall2 = zeros(n, 3);

for i = 1:n
    posFolder = sortedSubfolderNames{i};
    csvFile = fullfile(output_folderMain, posFolder, 'bacteriaCounts.csv');
    
    data = readtable(csvFile, 'ReadVariableNames', false, 'HeaderLines', 1, 'NumHeaderLines', 1);
    
    strainCounts = data{1:ActualStrainNum+1, 2};
    strainCounts(isnan(strainCounts)) = 0; 
    bacteriaCountsall2(i, :) = strainCounts';
    bacteriaCountsall1 = bacteriaCountsall1 + strainCounts';
    
    fileID = fopen(csvFile, 'r');
    if fileID == -1; error('File Lock Error: %s', csvFile); end
    
    foundLevels = false;
    while ~feof(fileID)
        lineText = fgetl(fileID);
        if contains(lineText, 'PerfectMatch,One2ThreeMisMatch,Four2FiveMisMatch')
            foundLevels = true;
            break;
        end
    end
    
    if foundLevels
        lineText = fgetl(fileID);
        levelData = sscanf(lineText, '%f,%f,%f')';
        levelData(isnan(levelData)) = 0; 
    else
        levelData = [0, 0, 0];
    end
    
    fclose(fileID);
    
    levelall2(i, :) = levelData;
    levelall1 = levelall1 + levelData;
end

outputFile = fullfile(output_folderMain, 'allBacteriaCounts.csv');
fid = fopen(outputFile, 'w');
if fid == -1; error('File Lock Error: Close Excel file %s', outputFile); end

fprintf(fid, 'Strain,CountsAll');
for i = 1:n
    fprintf(fid, ',Pos%d', sortedNumbers(i));  
end
fprintf(fid, '\n');

for j = 1:ActualStrainNum+1
    fprintf(fid, '%s,%d', bacteriaName{j}, bacteriaCountsall1(j));
    for i = 1:n
        fprintf(fid, ',%d', bacteriaCountsall2(i, j));
    end
    fprintf(fid, '\n');
end

fprintf(fid, ',');
for i = 1:n
    fprintf(fid, ',');
end
fprintf(fid, '\n');

levels = {'PerfectMatch'; 'One2ThreeMisMatch'; 'Four2FiveMisMatch'};
for j = 1:3
    fprintf(fid, '%s,%d', levels{j}, levelall1(j));
    for i = 1:n
        fprintf(fid, ',%d', levelall2(i, j));
    end
    fprintf(fid, '\n');
end

fclose(fid);
elapsedTimeTotal = toc;
fprintf('Full Batch Execution Completed in %.2f seconds.\nUnified logs exported to: %s\n', elapsedTimeTotal, outputFile);

%% ================== 11. Helper Functions ==================
function updateProgressBar(total_tasks)
    persistent completed_tasks;
    if isempty(completed_tasks)
        completed_tasks = 1;
    else
        completed_tasks = completed_tasks + 1;
    end
    fprintf('Processed FOV: %d / %d (%.1f%%)\n', completed_tasks, total_tasks, (completed_tasks/total_tasks)*100);
end

%% ==============================================================================
%                          LOCAL FUNCTIONS (Subroutines)
% ==============================================================================

function [StrainLikehood, Decode, level] = StrainIdentify(StrainCode, CodexRes, CorrBit)
    % --------------------------------------------------------------------------
    % Resolves acquired cellular barcodes against the predefined Codebook.
    % Incorporates Hamming-distance error correction (up to 5 bits).
    % --------------------------------------------------------------------------
    
    % Initialize likelihood matrix (Cells x Target Strains)
    StrainLikehood = zeros(size(CodexRes, 1), size(StrainCode, 2));

    % Compute initial matching likelihoods (exact position matches)
    for StrainIndex = 1:size(StrainCode, 2)
        StrainLikehood(:, StrainIndex) = sum(CodexRes(:, :) == StrainCode(:, StrainIndex)', 2);
    end
    
    % Dynamically determine total hybridization rounds
    TotalRounds = size(CodexRes, 2); 

    % Identify absolute noise/artifacts (zero signal across ALL rounds)
    RowZero = find(sum(CodexRes(:, :) == 0, 2) == TotalRounds);
    
    % Initialize output decoding matrix [Identified Strain ID, Match Score]
    Decode = zeros(size(CodexRes, 1), 2);
    
    % --- Perfect Matches ---
    [RowExact, ColExact] = find(StrainLikehood(:, :) == size(StrainCode, 1));
    Decode(RowExact, 1) = ColExact;
    Decode(RowExact, 2) = size(StrainCode, 1);
    
    % Level tracker: [Perfect, 1-3 Mismatches, 4-5 Mismatches]
    level = zeros(1,3);
    level(1,1) = numel(RowExact);
    
    % --- Dynamic Error Correction (1 to 5-bit Mismatches) ---
    % Implements a vectorized Hamming-distance tolerance threshold.
    max_corr = min(CorrBit, 5); % Cap maximum allowable error correction at 5 bits
    
    for n = 1:max_corr
        % Define the target matching score (Perfect code length minus 'n' mismatches)
        target_score = size(StrainCode, 1) - n;
        
        % Identify valid cells strictly meeting two rigorous criteria:
        % 1. The maximum likelihood strictly equals the target score.
        % 2. The match is unique (sum == 1) to prevent ambiguous combinatorial assignment.
        valid_rows = find(max(StrainLikehood, [], 2) == target_score & sum(StrainLikehood == target_score, 2) == 1);
        
        if ~isempty(valid_rows)
            % Extract the specific strain (column) index for these valid candidate cells
            [row_idx, col_idx] = find(StrainLikehood(valid_rows, :) == target_score);
            
            % Assign resolved strain IDs and their corresponding likelihood scores
            Decode(valid_rows(row_idx), 1) = col_idx;
            Decode(valid_rows(row_idx), 2) = target_score;
            
            % Hierarchical statistical tracking:
            % Maps 1-3 bit mismatches to level(1,2) and 4-5 bit mismatches to level(1,3).
            lvl_idx = 2 + (n >= 4); 
            level(1, lvl_idx) = level(1, lvl_idx) + numel(valid_rows);
        end
    end
    
    % Re-assign absolute noise/artifacts (zero signal across all rounds) to 'Unknown' state
    % Prevents artifactual misclassification and data inflation.
    Decode(RowZero, 1) = 0;
    Decode(RowZero, 2) = 4;
end

% ------------------------------------------------------------------------------

function [StrainImageAll, bacteriaCounts] = LabelImage1(CC, Decode, bacteriaName, output_folder, img_size)
    % --------------------------------------------------------------------------
    % Maps decoded physical states back to spatial coordinates, generating 
    % quantitative counts and rendering multiplexed pseudo-colored spatial maps.
    % --------------------------------------------------------------------------
    
    Labels = labelmatrix(CC);
    [StrainIndex, ~, UniqueIndex] = unique(Labels);
    
    StrainIndex = double(StrainIndex); 
    
    StrainImageAll = zeros(img_size);
    bacteriaCounts = zeros(1, size(bacteriaName, 1));
    
    % Iterate through targeted biological strains
    for i = 1:size(bacteriaName, 1)-1 
        StrainIndex(2:end) = double(Decode(:, 1) == i);
        StrainImage = reshape(StrainIndex(UniqueIndex), img_size);

        % [Optimization] Massive Speedup: Direct count from matrix, bypassing regionprops
        bacteriaCounts(i) = sum(Decode(:, 1) == i);
        
        % Export isolated strain image
        filename1 = sprintf('StrainImage%d_', i);
        filename = strcat(filename1, bacteriaName{i}, '.tif');
        imwrite(uint8(StrainImage .* i), fullfile(output_folder, filename));

        % Accumulate into master spatial map (i serves as grayscale weight)
        StrainImageAll = StrainImageAll + StrainImage .* i;

        % Export accumulated map
        filename = strcat('StrainImageAll', '.tif');
        imwrite(uint8(StrainImageAll), fullfile(output_folder, filename));
    end

    % --- Process Unassigned/Unknown Cells ---
    StrainIndex(2:end) = double(Decode(:, 1) == 0);
    StrainImage = reshape(StrainIndex(UniqueIndex), img_size);
    
    % [Optimization] Fast counting for Unknowns
    bacteriaCounts(end) = sum(Decode(:, 1) == 0); 
    
    % Export isolated unknown image
    filename1 = sprintf('StrainImage%d_', i + 1);
    filename = strcat(filename1, bacteriaName{i + 1}, '.tif');
    imwrite(uint8(StrainImage .* (i + 1)), fullfile(output_folder, filename));

    % Incorporate unknowns into master spatial map
    StrainImageAll = StrainImageAll + StrainImage .* (i + 1);
end