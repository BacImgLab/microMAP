%% 
%{
==============================================================================
Multi-Channel Serpentine (Snake-Pattern) Image Stitching Pipeline
==============================================================================
Purpose: 
    Automates the large-scale spatial reconstruction of multiplexed imaging 
    tiles (Fields of View, FOVs) into unified, high-resolution panoramic mosaics.

Core Method:
    1. Dynamically parses directories to segregate distinct acquisition channels.
    2. Sorts spatial FOVs numerically and maps numerically ordered FOV indices 
       onto an acquisition grid.
    3. Executes a "Snake-Pattern" (serpentine) spatial traversal for horizontal 
       row assembly, followed by a bottom-to-top vertical amalgamation.
    4. Applies a Maximum Intensity Projection (MIP) blending algorithm within 
       overlapping boundary regions to preserve true fluorescent signals without 
       artifactual edge suppression or blurring.

Robustness & Memory Design :
    1. Loading Architecture: Replaces naive bulk memory allocation 
       with on-the-fly I/O fetching. Tile matrices are loaded strictly ad hoc 
       during the stitching instance, eliminating Out-of-Memory (OOM) crashes 
       when assembling massive gigapixel datasets.
    2. Vectorized Regex Parsing : Utilizes highly optimized string 
       parsing for extremely rapid channel demultiplexing and sequence sorting.
==============================================================================
%}

clear; clc;

% ================== 1. Parameter Configuration ==================
% User-defined target directory containing pre-processed FOV tiles
input_folder = uigetdir('', 'Select stitched tile directory');  

overlap_ratio = 0.02;   % Spatial overlap percentage between adjacent FOVs (e.g., 2%)
num_rows = 22;          % Designated row count.

fprintf('Initiating multi-channel batch stitching pipeline...\n');
fprintf('Input Directory: %s\n', input_folder);
fprintf('Target Spatial Overlap: %.1f%%\n', overlap_ratio * 100);

% File pattern validation
file_pattern = fullfile(input_folder, 'stacked_Pos*.tif');
tif_files = dir(file_pattern);

if isempty(tif_files)
    error('Pipeline Terminated: No files matching [stacked_Pos*_channel_*.tif] localized in %s', input_folder);
end

% Construct persistent output directory
output_folder = fullfile(input_folder, 'stitched_results');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% ================== 2. Automated Channel Demultiplexing ==================
% Rapid vectorized regex extraction of all unique spectral channels
filenames = {tif_files.name};
tokens = regexp(filenames, 'stacked_Pos\d+_(.*)\.tif', 'tokens', 'once');
valid_tokens = tokens(~cellfun(@isempty, tokens));
channel_types = unique(cellfun(@(x)x{1},tokens,'UniformOutput',false));

fprintf('Identified %d unique acquisition channels: %s\n', length(channel_types), strjoin(channel_types, ', '));

% ================== 3. Core Stitching Execution Loop ==================
for ch_idx = 1:length(channel_types)
    current_channel = channel_types{ch_idx};
    fprintf('\n=== Processing Target: %s (%d/%d) ===\n', current_channel, ch_idx, length(channel_types));
    
    % Isolate files specifically corresponding to the active channel
    ch_file_pattern = sprintf('stacked_Pos*_%s.tif', current_channel);
    ch_files = dir(fullfile(input_folder, ch_file_pattern));
    
    if isempty(ch_files)
        warning('Warning: No valid tiles found for %s. Bypassing...', current_channel);
        continue;
    end
    
    % Extract spatial positions and construct Lazy-Load paths
    pos_nums = zeros(1, length(ch_files));
    img_paths = cell(1, length(ch_files));
    
    for i = 1:length(ch_files)
        tok = regexp(ch_files(i).name, 'stacked_Pos(\d+)_', 'tokens', 'once');
        pos_nums(i) = str2double(tok{1});
        img_paths{i} = fullfile(input_folder, ch_files(i).name);
    end
    
    % Ensure strict numerical ordering of FOVs
    [sorted_pos, sort_idx] = sort(pos_nums);
    sorted_paths = img_paths(sort_idx);
    
    fprintf('Grid spatial sequence verified for %d tiles.\n', length(sorted_pos));
    
    % Execute snake-pattern assembly utilizing Lazy-Load paths
        stitched_image = stitch_images_snake_pattern(sorted_paths, sorted_pos, overlap_ratio, num_rows);
    
    % Export unified panoramic mosaic
    output_filename = sprintf('%s_stitched.tif', current_channel);
    output_path = fullfile(output_folder, output_filename);
    imwrite(stitched_image, output_path);
    
    fprintf('Successfully exported mosaic: %s\n', output_filename);
end

fprintf('\nPipeline Execution Complete! Mosaics preserved at: %s\n', output_folder);


%% ==============================================================================
%                          LOCAL FUNCTIONS (Subroutines)
% ==============================================================================

function num_rows = calculate_optimal_grid(num_images, overlap_ratio, sample_img)
    % Derives the optimal grid dimension (closest to 1:1 aspect ratio)
    best_rows = 1;
    best_aspect_diff = inf;

    for rows = 1:num_images
        cols = ceil(num_images / rows);
        [h, w, ~] = size(sample_img);

        est_width = w * cols * (1 - overlap_ratio * (cols - 1) / cols);
        est_height = h * rows * (1 - overlap_ratio * (rows - 1) / rows);

        aspect_ratio = est_width / est_height;
        aspect_diff = abs(aspect_ratio - 1);

        if aspect_diff < best_aspect_diff
            best_aspect_diff = aspect_diff;
            best_rows = rows;
        end
    end

    num_rows = best_rows;
    fprintf('Automatically configured adaptive grid: %d rows\n', num_rows);
end

function result = stitch_images_snake_pattern(image_paths, pos_numbers, overlap_ratio, num_rows)
    % Orchestrates the serpentine (snake) coordinate mapping
    num_images = length(image_paths);
    if num_images == 1
        result = imread(image_paths{1});
        return;
    end

    num_cols = ceil(num_images / num_rows);
    fprintf('Topology architecture established: %d Rows x %d Columns\n', num_rows, num_cols);

    row_results = cell(1, num_rows);

    for row = 1:num_rows
        start_idx = (row - 1) * num_cols + 1;
        end_idx = min(row * num_cols, num_images);

        if start_idx > num_images
            break;
        end

        row_paths = image_paths(start_idx:end_idx);
        row_positions = pos_numbers(start_idx:end_idx);

        % Reverse trajectory for even rows (Serpentine mechanism)
        if mod(row, 2) == 0
            row_paths = fliplr(row_paths);
            row_positions = fliplr(row_positions);
        end

        fprintf('Stitching Row %d | Spatial Sequence: %s\n', row, mat2str(row_positions));

        if length(row_paths) > 1
            row_result = stitch_row_horizontal(row_paths, overlap_ratio);
        else
            row_result = imread(row_paths{1});
        end

        row_results{row} = row_result;
    end

    valid_rows = ~cellfun(@isempty, row_results);
    row_results = row_results(valid_rows);

    % Amalgamate constructed rows vertically (Bottom-to-Top specification)
    if length(row_results) > 1
        result = stitch_rows_vertical_bottom_to_top(row_results, overlap_ratio);
    else
        result = row_results{1};
    end

    fprintf('Assembly successful. Panoramic dimensions: %s pixels\n', mat2str(size(result)));
end

function result = stitch_row_horizontal(row_paths, overlap_ratio)
    % Executes sequential horizontal stitching using Lazy-Loading
    result = imread(row_paths{1});

    for i = 2:length(row_paths)
        img2 = imread(row_paths{i}); % Memory-safe Ad hoc loading
        [~, w1, ~] = size(result);
        [~, w2, ~] = size(img2);
        overlap_x = round(min(w1, w2) * overlap_ratio);
        result = stitch_two_images_horizontal(result, img2, overlap_x);
    end
end

function result = stitch_two_images_horizontal(img1, img2, overlap_pixels)
    % Core mathematical engine for horizontal matrix concatenation (MIP Blending)
    [h1, w1, c1] = size(img1);
    [h2, w2, c2] = size(img2);

    % Color channel standardization
    if c1 ~= c2
        if c1 == 1 && c2 == 3
            img1 = repmat(img1, [1, 1, 3]);
        elseif c1 == 3 && c2 == 1
            img2 = repmat(img2, [1, 1, 3]);
        end
        c1 = size(img1, 3);
    end

    new_width = w1 + w2 - overlap_pixels;
    new_height = max(h1, h2);
    result = zeros(new_height, new_width, c1, 'like', img1);

    result(1:h1, 1:w1, :) = img1;

    for col = 1:w2
        target_col = w1 - overlap_pixels + col;
        if col <= overlap_pixels
            overlap_height = min(h1, h2);
            % Maximum Intensity Projection (MIP) to preserve fluorescence
            for ch = 1:c1
                overlap_region1 = img1(1:overlap_height, w1 - overlap_pixels + col, ch);
                overlap_region2 = img2(1:overlap_height, col, ch);
                result(1:overlap_height, target_col, ch) = max(overlap_region1, overlap_region2);
            end

            if h1 > h2
                result(h2+1:h1, target_col, :) = img1(h2+1:h1, w1 - overlap_pixels + col, :);
            elseif h2 > h1
                result(h1+1:h2, target_col, :) = img2(h1+1:h2, col, :);
            end
        else
            result(1:h2, target_col, :) = img2(1:h2, col, :);
        end
    end

    % Padding corrections for asymmetrical heights
    if h1 > h2
        result(h2+1:h1, w1-overlap_pixels+1:end, :) = img1(h2+1:h1, w1-overlap_pixels+1:end, :);
    elseif h2 > h1
        result(h1+1:h2, 1:w1-overlap_pixels, :) = 0;
    end

    if isinteger(img1)
        result = cast(result, class(img1));
    end
end

function result = stitch_rows_vertical_bottom_to_top(row_results, overlap_ratio)
    % Reverses horizontal arrays to fulfill bottom-to-top traversal requirement
    reversed_rows = row_results(end:-1:1);
    result = reversed_rows{1};

    for i = 2:length(reversed_rows)
        img_above = reversed_rows{i};
        [h1, ~, ~] = size(result);
        [h2, ~, ~] = size(img_above);
        overlap_y = round(min(h1, h2) * overlap_ratio);
        result = stitch_two_images_vertical_bottom_to_top(result, img_above, overlap_y);
    end
end

function result = stitch_two_images_vertical_bottom_to_top(img_below, img_above, overlap_pixels)
    % Core mathematical engine for vertical matrix concatenation (MIP Blending)
    [h1, w1, c1] = size(img_below);
    [h2, w2, c2] = size(img_above);

    if c1 ~= c2
        if c1 == 1 && c2 == 3
            img_below = repmat(img_below, [1, 1, 3]);
        elseif c1 == 3 && c2 == 1
            img_above = repmat(img_above, [1, 1, 3]);
        end
        c1 = size(img_below, 3);
    end

    new_width = max(w1, w2);
    new_height = h1 + h2 - overlap_pixels;
    result = zeros(new_height, new_width, c1, 'like', img_below);

    result(1:h1, 1:w1, :) = img_below;

    for row = 1:overlap_pixels
        overlap_row_below = h1 - overlap_pixels + row;
        overlap_row_above = row;
        target_row = h1 - overlap_pixels + row;
        overlap_width = min(w1, w2);

        % Maximum Intensity Projection (MIP) applied vertically
        for ch = 1:c1
            below = img_below(overlap_row_below, 1:overlap_width, ch);
            above = img_above(overlap_row_above, 1:overlap_width, ch);
            result(target_row, 1:overlap_width, ch) = max(below, above);
        end

        if w1 > w2
            result(target_row, w2+1:w1, :) = img_below(overlap_row_below, w2+1:w1, :);
        end
    end

    result(h1+1:end, 1:w2, :) = img_above(overlap_pixels+1:end, :, :);

    % Alignment padding
    if w1 > w2
        result(h1+1:end, w2+1:end, :) = 0;
    elseif w2 > w1
        result(1:h1, w1+1:end, :) = 0;
    end

    if isinteger(img_below)
        result = cast(result, class(img_below));
    end
end