"""
==============================================================================
Multi-Round Multi-Channel Image Registration Pipeline
==============================================================================
Purpose: 
    Automated sub-pixel displacement estimation and image registration for 
    multi-round FISH imaging.

Core Method: 
    Utilizes the `StackReg` algorithm to compute translational shifts strictly 
    from the Brightfield channel. These physical shifts are then rigidly 
    applied to align all associated sparse fluorescent channels.

Key Engineering Tricks:
    1. Chained Reference: Aligns Round N to Round N-1 (instead of Round 1) 
       to gracefully handle gradual sample morphological changes over time.
    2. Global Rescue (Two-Pass): Computes a slide-wide average displacement 
       to artificially "rescue" specific FOVs that fail algorithmic tracking 
       due to local artifacts (e.g., bubbles or debris).
    3. Signal Preservation: Performs spatial warping using `float32` for 
       interpolation precision, then strictly clips and reverts to `uint16` 
       to prevent data overflow and preserve raw fluorescence intensities.
==============================================================================
"""
from pystackreg import StackReg
from skimage import exposure, filters
import numpy as np
import os
import re
from glob import glob
from tqdm import tqdm
import warnings
import tifffile
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import csv
from skimage.transform import SimilarityTransform, warp

# Ignore low contrast image warnings to clean up console output
warnings.filterwarnings("ignore", message=".*is a low contrast image.*")    

# ==================== Directory & File Parsing ====================

def select_directory():
    """Select directory using GUI"""
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select Directory Containing TIFF Files")
    root.destroy()
    return directory

def extract_file_components(filename):
    """
    Extract round, position, and channel information from the filename.
    """
    # Use raw string for regex
    pattern = r".*_(\d+)_Pos(\d+)_(.+)\.tif"  
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    
    return None, None, None

def find_all_files(directory):
    """Find all TIFF files within the specified directory"""
    pattern = os.path.join(directory, "*.tif")
    return glob(pattern)

def organize_files_by_position(files):
    """Organize files systematically based on their Field of View (FOV) position"""
    positions = {}
    for file in files:
        round_idx, pos_idx, channel = extract_file_components(os.path.basename(file))
        if pos_idx is None:
            print(f"Warning: Could not parse filename: {os.path.basename(file)}")
            continue
            
        if pos_idx not in positions:
            positions[pos_idx] = {}
            
        if round_idx not in positions[pos_idx]:
            positions[pos_idx][round_idx] = {}
            
        positions[pos_idx][round_idx][channel] = file
        
    return positions

# ==================== Core Registration Math & Preprocessing ====================

def preprocess_image_16bit(img):
    """16-bit image preprocessing, preserving 16-bit precision depth"""
    # Ensure the image array is 16-bit
    if img.dtype != np.uint16:
        img = img.astype(np.uint16)
    
    # Contrast stretching (2nd - 98th percentile)
    p2, p98 = np.percentile(img, (2, 98))
    
    # Apply contrast stretching limits
    img_rescale = np.clip((img - p2) * (65535 / (p98 - p2 + 1e-10)), 0, 65535).astype(np.uint16)
    
    return img_rescale

def get_translation_from_matrix(matrix):
    """Extract translation displacements (dx, dy) from the transformation matrix"""
    # Validate matrix existence and ensure it conforms to standard transformation dimensions
    if matrix is not None and matrix.shape in [(3, 3), (2, 3)]:
        dx = matrix[0, 2]
        dy = matrix[1, 2]
        return dx, dy
    else:
        return 0, 0

def find_bright_channel(round_files):
    """
    Locate the brightfield (reference) channel.
    Raises a ValueError if the brightfield channel is strictly absent.
    """
    for channel_name, channel_path in round_files.items():
        if 'bright' in channel_name.lower():
            return channel_name, channel_path
            
    raise ValueError("Critical Error: Brightfield reference channel not found in the current round.")

def apply_translation(image, dx, dy):
    """Apply rigid translation transformation to the image"""

    frame = image.astype(np.float32)
    # Generate translation transformation
    tform = SimilarityTransform(translation=(dx, dy))
    # Execute warp transformation
    translated_frame = warp(frame, tform, preserve_range=True)
    translated = np.clip(translated_frame, 0, 65535).astype(np.uint16)
    
    return translated

# ==================== Parallel Execution Logic ====================

def process_single_position(args):
    """Function to process a single position, intended for parallel multiprocessing"""
    pos_idx, position_files, output_dir, recorrection_data = args
    
    # Retrieve all rounds and sort sequentially
    rounds = sorted(position_files.keys())
    if not rounds:
        return f"Position {pos_idx} skipped (no rounds)", []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store displacement metadata (only tracking brightfield channel displacements)
    displacement_data = []
    
    # Check if this position requires secondary recorrection
    need_recorrection = recorrection_data and pos_idx in recorrection_data
    
    # Initialize reference image (brightfield channel of the first round)
    ref_round = rounds[0]
    ref_bright_channel, ref_bright_path = find_bright_channel(position_files[ref_round])
    
    # Preprocess reference image, maintaining 16-bit depth
    ref_bright = tifffile.imread(ref_bright_path) 
    ref_bright_preprocessed = preprocess_image_16bit(ref_bright)
    
    # Record initial round metadata (displacement is 0)
    displacement_data.append({
        'position': pos_idx,
        'round': ref_round,
        'channel': ref_bright_channel,
        'filename': os.path.basename(ref_bright_path),
        'dx': 0.0,
        'dy': 0.0,
        'reference_round': ref_round,
        'recorrected': need_recorrection
    })
    
    # Process all channels of the first round (No registration required)
    for channel_name, channel_path in position_files[ref_round].items():
        channel_img = tifffile.imread(channel_path)
        
        # Apply average displacement if recorrection is triggered
        if need_recorrection and ref_round in recorrection_data[pos_idx]:
            avg_dx, avg_dy = recorrection_data[pos_idx][ref_round]
            channel_img = apply_translation(channel_img, avg_dx, avg_dy)
        
        # Save image utilizing the original nomenclature structure
        original_filename = os.path.basename(channel_path)
        output_filename = f"registered_{original_filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Export image
        tifffile.imwrite(output_path, channel_img, photometric='minisblack')
    
    # Process subsequent rounds (Registered against the brightfield image of the previous round)
    prev_bright = ref_bright_preprocessed.copy()
    
    # Process each round sequentially
    for round_idx in rounds[1:]:
        round_files = position_files[round_idx]
        
        # Locate the brightfield channel for the current round
        current_bright_channel, current_bright_path = find_bright_channel(round_files)
            
        current_bright = tifffile.imread(current_bright_path)
        
        # Preprocess current brightfield image, maintaining 16-bit depth
        current_bright_preprocessed = preprocess_image_16bit(current_bright)
        
        # Register using translation transformation (Assuming solely translational drift)
        try:
            # Apply average displacement if recorrection is triggered
            if need_recorrection and round_idx in recorrection_data[pos_idx]:
                avg_dx, avg_dy = recorrection_data[pos_idx][round_idx]
                
                # Record displacement metadata utilizing the average
                displacement_data.append({
                    'position': pos_idx,
                    'round': round_idx,
                    'channel': current_bright_channel,
                    'filename': os.path.basename(current_bright_path),
                    'dx': avg_dx,
                    'dy': avg_dy,
                    'reference_round': rounds[rounds.index(round_idx) - 1],
                    'recorrected': True
                })
                
                # Apply average displacement to all channels in the current round
                for channel_name, channel_path in round_files.items():
                    channel_img = tifffile.imread(channel_path)
                    
                    # Apply transformation to the channel
                    registered_channel = apply_translation(channel_img, avg_dx, avg_dy)
                    
                    # Save the registered image
                    original_filename = os.path.basename(channel_path)
                    output_filename = f"registered_{original_filename}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Export image
                    tifffile.imwrite(output_path, registered_channel, photometric='minisblack')
                
                # Update reference image to the corrected brightfield of the current round
                registered_bright = apply_translation(current_bright_preprocessed, avg_dx, avg_dy)
                prev_bright = registered_bright
                
            else:
                # Instantiate StackReg object with default parameters
                sr = StackReg(StackReg.TRANSLATION)
                sr.register(prev_bright, current_bright_preprocessed)
                
                # Retrieve transformation matrix and extract displacement values
                transformation_matrix = sr.get_matrix()
                dx, dy = get_translation_from_matrix(transformation_matrix)
                
                # Record displacement metadata
                displacement_data.append({
                    'position': pos_idx,
                    'round': round_idx,
                    'channel': current_bright_channel,
                    'filename': os.path.basename(current_bright_path),
                    'dx': dx,
                    'dy': dy,
                    'reference_round': rounds[rounds.index(round_idx) - 1],  # Refers to the previous round
                    'recorrected': False
                })
                
                # Apply computed transformation to all channels in the current round
                for channel_name, channel_path in round_files.items():
                    channel_img = tifffile.imread(channel_path)
                    
                    # Apply transformation to the channel, preserving 16-bit precision
                    transformed = sr.transform(channel_img.astype(np.float32))
                    registered_channel = np.clip(transformed, 0, 65535).astype(np.uint16)
                    
                    # Save the registered image
                    original_filename = os.path.basename(channel_path)
                    output_filename = f"registered_{original_filename}"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Export image
                    tifffile.imwrite(output_path, registered_channel, photometric='minisblack')
                
                # Update reference image to the corrected brightfield of the current round
                registered_bright = sr.transform(current_bright_preprocessed.astype(np.float32))
                prev_bright = np.clip(registered_bright, 0, 65535).astype(np.uint16)
            
        except Exception as e:
            print(f"Registration failed for round {round_idx}: {e}")
            
            # Record failed registration metadata (displacement is 0)
            displacement_data.append({
                'position': pos_idx,
                'round': round_idx,
                'channel': current_bright_channel,
                'filename': os.path.basename(current_bright_path),
                'dx': 0.0,
                'dy': 0.0,
                'reference_round': rounds[rounds.index(round_idx) - 1],
                'note': 'registration_failed',
                'recorrected': need_recorrection
            })
            
            # Bypass registration and save raw images directly upon failure
            for channel_name, channel_path in round_files.items():
                channel_img = tifffile.imread(channel_path)
                
                original_filename = os.path.basename(channel_path)
                output_filename = f"registered_{original_filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                tifffile.imwrite(output_path, channel_img, photometric='minisblack')
    
    return f"Position {pos_idx} processing completed", displacement_data

# ==================== Outlier Filtering & Dual-Pass Recorrection ====================

def filter_outliers(data, low_percentile=20, high_percentile=80):
    """Filter outliers, retaining only data within the specified percentile threshold"""
    if not data:
        return []
    
    # Compute percentiles for raw data (retaining algebraic signs)
    low_threshold = np.percentile(data, low_percentile)
    high_threshold = np.percentile(data, high_percentile)
    
    # Filter data: Retain original data bounded by the percentile limits
    filtered_data = [x for x in data if low_threshold <= x <= high_threshold]
    
    return filtered_data

def calculate_round_averages(displacement_data):
    """Compute the average displacement for each round, excluding outliers"""
    round_data = {}
    
    for item in displacement_data:
        round_idx = item['round']
        if round_idx not in round_data:
            round_data[round_idx] = {'dx_values': [], 'dy_values': []}
        
        round_data[round_idx]['dx_values'].append(item['dx'])
        round_data[round_idx]['dy_values'].append(item['dy'])
    
    # Compute the average displacement for each round, excluding outliers
    round_averages = {}
    for round_idx, data in round_data.items():
        # Filter outliers
        filtered_dx = filter_outliers(data['dx_values'])
        filtered_dy = filter_outliers(data['dy_values'])
        
        # Calculate averages
        if filtered_dx:
            avg_dx = sum(filtered_dx) / len(filtered_dx)
        else:
            avg_dx = 0.0
            
        if filtered_dy:
            avg_dy = sum(filtered_dy) / len(filtered_dy)
        else:
            avg_dy = 0.0
        
        # Record the count of raw data points vs filtered data points
        original_count = len(data['dx_values'])
        filtered_dx_count = len(filtered_dx)
        filtered_dy_count = len(filtered_dy)
        
        round_averages[round_idx] = {
            'avg_dx': avg_dx,
            'avg_dy': avg_dy,
            'original_positions': original_count,
            'filtered_dx_positions': filtered_dx_count,
            'filtered_dy_positions': filtered_dy_count
        }
    
    return round_averages

def identify_recorrection_positions(displacement_data, round_averages, threshold=10):
    """Identify positions that need secondary recorrection"""
    recorrection_data = {}
    
    # Organize displacement data hierarchically by position and round
    position_round_data = {}
    for item in displacement_data:
        pos_idx = item['position']
        round_idx = item['round']
        
        if pos_idx not in position_round_data:
            position_round_data[pos_idx] = {}
        
        position_round_data[pos_idx][round_idx] = {
            'dx': item['dx'],
            'dy': item['dy']
        }
    
    # Inspect each position across all rounds
    for pos_idx, round_data in position_round_data.items():
        for round_idx, displacement in round_data.items():
            if round_idx in round_averages:
                avg_dx = round_averages[round_idx]['avg_dx']
                avg_dy = round_averages[round_idx]['avg_dy']
                
                dx_diff = abs(displacement['dx'] - avg_dx)
                dy_diff = abs(displacement['dy'] - avg_dy)
                
                # Flag the position for recorrection if the dx or dy difference exceeds the threshold
                if dx_diff > threshold or dy_diff > threshold:
                    if pos_idx not in recorrection_data:
                        recorrection_data[pos_idx] = {}
                    
                    # Record the average displacement to be utilized for all rounds of this position
                    for r_idx in round_data.keys():
                        if r_idx in round_averages:
                            recorrection_data[pos_idx][r_idx] = (
                                round_averages[r_idx]['avg_dx'],
                                round_averages[r_idx]['avg_dy']
                            )

                    # Flag the entire position for recorrection upon a single threshold breach
                    print(f"Position {pos_idx} flagged for recorrection: Round {round_idx} displacement discrepancy (dx: {dx_diff:.4f}, dy: {dy_diff:.4f}) exceeded threshold {threshold}")
                    break  
    
    return recorrection_data

# ==================== Data Export & I/O ====================

def save_displacement_data(displacement_data, round_averages, recorrection_data, output_dir):
    """Export displacement metadata to CSV files"""
    if displacement_data:
        # Export comprehensive displacement metadata
        detailed_csv_path = os.path.join(output_dir, "registration_displacements_detailed.csv")
        
        # Utilize csv module to write detailed data
        with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Consolidate all possible fieldnames
            fieldnames = set()
            for item in displacement_data:
                fieldnames.update(item.keys())
            fieldnames = sorted(fieldnames)  # Sort alphabetically for consistency
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in displacement_data:
                writer.writerow(item)
        
        print(f"Detailed displacement data successfully exported to: {detailed_csv_path}")
        
        # Export average displacement across rounds
        averages_csv_path = os.path.join(output_dir, "registration_round_averages.csv")
        with open(averages_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['round', 'avg_dx', 'avg_dy', 'original_positions', 'filtered_dx_positions', 'filtered_dy_positions', 'recorrected_positions']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Tally the number of recorrected positions per round
            recorrected_counts = {}
            for pos_idx in recorrection_data:
                for round_idx in recorrection_data[pos_idx]:
                    if round_idx not in recorrected_counts:
                        recorrected_counts[round_idx] = 0
                    recorrected_counts[round_idx] += 1
            
            for round_idx, data in round_averages.items():
                row_data = {
                    'round': round_idx,
                    'avg_dx': data['avg_dx'],
                    'avg_dy': data['avg_dy'],
                    'original_positions': data['original_positions'],
                    'filtered_dx_positions': data['filtered_dx_positions'],
                    'filtered_dy_positions': data['filtered_dy_positions'],
                    'recorrected_positions': recorrected_counts.get(round_idx, 0)
                }
                writer.writerow(row_data)
        
        print(f"Round average displacement data successfully exported to: {averages_csv_path}")
        
        # Export positional recorrection documentation
        if recorrection_data:
            recorrection_csv_path = os.path.join(output_dir, "registration_recorrection_info.csv")
            with open(recorrection_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['position', 'rounds_recorrected']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for pos_idx, round_data in recorrection_data.items():
                    rounds_str = ', '.join([str(r) for r in round_data.keys()])
                    writer.writerow({
                        'position': pos_idx,
                        'rounds_recorrected': rounds_str
                    })
            
            print(f"Recorrection documentation successfully exported to: {recorrection_csv_path}")
        
        # Output round average displacements to console
        print("\nAverage Displacements per Round (Outliers Excluded):")
        print("Round\tAvg X Shift\tAvg Y Shift\tRaw Pos Count\tFiltered X Count\tFiltered Y Count\tRecorrected Pos Count")
        for round_idx, data in sorted(round_averages.items()):
            recorrected_count = recorrected_counts.get(round_idx, 0)
            print(f"{round_idx}\t{data['avg_dx']:.4f}\t\t{data['avg_dy']:.4f}\t\t{data['original_positions']}\t\t{data['filtered_dx_positions']}\t\t{data['filtered_dy_positions']}\t\t{recorrected_count}")
    else:
        print("No displacement data available for export.")

# ==================== Main Orchestrator ====================

def main(input_dir):
    """Main function: Orchestrates the image registration pipeline"""
    # Retrieve current date for timestamping
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Construct output directory path (peer to input directory, appended with _reg and date)
    input_dir_name = os.path.basename(input_dir)
    parent_dir = os.path.dirname(input_dir)
    output_dir = os.path.join(parent_dir, f"{input_dir_name}_reg_{current_date}")
    
    # Locate all compatible files
    all_files = find_all_files(input_dir)
    
    if not all_files:
        print("No TIFF files detected. Please verify the input directory.")
        return
    
    # Organize files systematically by FOV position
    positions = organize_files_by_position(all_files)
    
    if not positions:
        print("No valid files detected. Please verify the filename nomenclature formatting.")
        print("Expected regular expression pattern: .*_(\\d+)_Pos(\\d+)_(.+)\\.tif")
        print("Example: bacteria_mutiround_0_0_Pos1_647_705-72.tif")
        return
    
    # Formulate argument parameter list
    args_list = [(pos_idx, position_files, output_dir, {}) for pos_idx, position_files in positions.items()]
    
    # Initialize parallel processing across available CPU cores
    num_processes = min(max(1, cpu_count() - 4), len(args_list))
    
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Detected {len(positions)} positions for processing.")
    print(f"Deploying {num_processes} parallel processes.")
    print("Note: The current PyStackReg framework does not support explicit parameterization for pyramid levels and iteration counts.")
    print("Displacement calculations will strictly exclude outliers outside the 20%-80% percentile bounds.")
    print("Executing a dual-pass registration: Primary pass via StackReg, secondary pass utilizing averaged displacements for outlier positions.")
    
    # Pass 1: Primary Registration
    print("\n=== Pass 1: Primary Registration ===")
    all_displacement_data = []
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_position, args_list),
            total=len(args_list),
            desc="Pass 1 Progress"
        ))
    
    # Consolidate displacement metadata and execution logs
    result_messages = []
    for result in results:
        if isinstance(result, tuple) and len(result) == 2:
            message, displacement_data = result
            result_messages.append(message)
            all_displacement_data.extend(displacement_data)
        else:
            result_messages.append(result)
    
    # Output Pass 1 execution logs
    for message in result_messages:
        print(message)
    
    # Compute average displacement per round
    print("\n=== Computing Round-Wise Average Displacements ===")
    round_averages = calculate_round_averages(all_displacement_data)
    
    # Identify positions need secondary recorrection
    print("\n=== Identifying Positions for Recorrection ===")
    recorrection_data = identify_recorrection_positions(all_displacement_data, round_averages, threshold=10)
    
    # Execute Pass 2 (Recorrection) if outlier positions are detected
    if recorrection_data:
        print(f"\n=== Pass 2: Secondary Recorrection ({len(recorrection_data)} positions flagged) ===")
        
        # Formulate argument parameter list for Pass 2
        recorrection_args_list = [
            (pos_idx, position_files, output_dir, recorrection_data) 
            for pos_idx, position_files in positions.items() 
            if pos_idx in recorrection_data
        ]
        
        # Initialize parallel processing for recorrection
        with Pool(processes=min(max(1, cpu_count() - 4), len(recorrection_args_list))) as pool:
            recorrection_results = list(tqdm(
                pool.imap(process_single_position, recorrection_args_list),
                total=len(recorrection_args_list),
                desc="Pass 2 Progress"
            ))
        
        # Update and reconcile displacement metadata
        for i, (pos_idx, _, _, _) in enumerate(recorrection_args_list):
            if i < len(recorrection_results) and isinstance(recorrection_results[i], tuple):
                _, new_displacement_data = recorrection_results[i]
                
                # Purge deprecated displacement metadata
                all_displacement_data = [item for item in all_displacement_data if item['position'] != pos_idx]
                
                # Append updated displacement metadata
                all_displacement_data.extend(new_displacement_data)
        
        print("Pass 2 Recorrection Successfully Completed.")
    else:
        print("\nNo positions flagged for recorrection.")
    
    # Export definitive displacement metadata
    print("\n=== Exporting Final Results ===")
    save_displacement_data(all_displacement_data, round_averages, recorrection_data, output_dir)
    
    print(f"Pipeline Execution Complete! Output successfully exported to: {output_dir}")

if __name__ == "__main__":
    # Configure multiprocessing initialization methodology
    mp.set_start_method('spawn', force=True)
    
    # Prompt user for input directory via GUI
    print("Awaiting selection of the source TIFF directory...")
    input_directory = select_directory()
    
    if not input_directory:
        print("No directory selected. Execution terminated.")
        exit()
    
    # Initiate image processing pipeline
    main(input_directory)