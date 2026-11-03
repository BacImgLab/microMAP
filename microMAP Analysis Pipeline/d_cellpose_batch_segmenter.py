"""
==============================================================================
High-Throughput Cellpose Instance Segmentation Pipeline
==============================================================================
Purpose: 
    Executes deep-learning-based instance segmentation on multi-FOV bacterial 
    imaging datasets to generate highly accurate single-cell boundaries.

Core Method: 
    Uses a fine-tuned Cellpose model with customized inference parameters 
    optimized for the target imaging morphology.

Key Engineering Tricks:
    1. 1-Pixel Boundary Constraint: Employs a custom morphological split function 
       to strictly enforce a 1-pixel artificial boundary between touching 
       cells, preventing contiguous instance merging.
==============================================================================
"""

import os
import re
import datetime
import numpy as np
import warnings
from cellpose import models, io as cp_io
from skimage import io, measure
from tqdm import tqdm
from tkinter import Tk
from tkinter.filedialog import askdirectory

# ==================== Pipeline Configuration ====================
total_channels = 1   # Number of dimensions for multi-channel input
model_type = None
# Custom fine-tuned Cellpose model for specific bacterial morphology
pretrained_model = 'cpsam_20250730_095145'  
channels = [0, 0]    # Grayscale input configuration for Cellpose

# Normalization parameter intervals
interval = [[0.1, 100], [0.1, 99], [1, 99]] 

# ==================== Segmentation Parameters ====================
processing_params = {
    # 'diameter': None,        # Bypass auto-estimation; explicitly define if needed
    'flow_threshold': 0.6,     # Stringency for spatial flow dynamics
    'cellprob_threshold': 0,   # Threshold for cell probability maps
    'normalize': {             # Custom spatial normalization metrics
        'lowhigh': None,
        'percentile': interval[2], # Apply robust 1-99th percentile intensity normalization
        'normalize': True,
        'norm3D': False,
        'sharpen_radius': 0,       # Disable computational sharpening (preserve raw edges)
        'smooth_radius': 0,
        'tile_norm_blocksize': 0,  # Disable adaptive tile normalization
        'tile_norm_smooth3D': 0,
        'invert': False
    },
    'niter': 0,                # Disable iterative optimization for speed
    'resample': True,          # Preserve native optical resolution
    'augment': False,          # Disable test-time augmentation (TTA)
    'batch_size': 8            # VRAM-dependent batch processing size
}

# Suppress non-critical user warnings to maintain a clean console output
warnings.simplefilter("ignore", UserWarning)

# ==================== Directory Initialization & GUI Binding ====================
root = Tk() 
# Trigger native OS directory selection dialog
input_folder0 = askdirectory(title='Select Target Input Folder for Segmentation')

# Force hardware acceleration (GPU) for high-throughput processing
gpu_choice=input(
"Use GPU? y/n:"
)
GPU = gpu_choice in {'Y', 'y'}  
print(f"Hardware Acceleration (GPU) Status: {GPU}")
root.destroy() # Release GUI resources to prevent memory leaks

# Construct timestamped output directory
input_root = os.path.dirname(input_folder0)
ymd = datetime.datetime.now().strftime('%Y-%m-%d')
output_root = os.path.join(input_root, f'cellpose_masks_{ymd}')
os.makedirs(output_root, exist_ok=True)

# ==================== Optimized I/O Filtering (Regex Optimization) ====================
# Extract raw contents from the designated directory
all_files = sorted([f for f in os.listdir(input_folder0) if f.endswith('.tif')])

# Strictly isolate target fluorophore channel (e.g., 647_705-72)
pattern = re.compile(r"([a-zA-Z0-9_ \-\.]+)_647_705-72\.tif$")
files_tif = []

for fname in all_files:
    if pattern.match(fname):
        # Only append the filename (not absolute path) to ensure robust downstream I/O
        files_tif.append(fname)

print(f"Successfully identified {len(files_tif)} target TIFF files for batch processing.")

# ==================== Core Functions ====================

def get_dtype_max_value(image_array):
    """
    Dynamically retrieves the theoretical maximum value of the matrix data type. 
    """
    dtype = image_array.dtype
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).max
    else:
        raise ValueError(f"Unsupported internal datatype structure: {dtype}")

def init_model(use_gpu):
    """
    Initializes the specified custom fine-tuned Cellpose model
    """
    return models.CellposeModel(
        gpu=use_gpu,              
        pretrained_model=pretrained_model
    )

def refine_instance_boundaries(img):
    """
    [Morphological Post-processing]
    Enforces a strict 1-pixel artificial boundary constraint between adjacent or 
    touching cell instances to prevent morphological merging during downstream analysis.
    """
    rows, cols = img.shape
    labeledImage, numLabels = measure.label(img > 0, connectivity=2, return_num=True)
    outputImage = img.copy()
    
    for label in range(1, numLabels + 1):
        positions = np.where(labeledImage == label)
        rowIdx, colIdx = positions
        for idx in range(len(rowIdx)):
            i = rowIdx[idx]
            j = colIdx[idx]
            
            # Scan the 8-connected neighborhood for conflicting instance labels
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if labeledImage[ni, nj] == label and img[ni, nj] != img[i, j]:
                            # Suppress pixels with lower confidence/intensity at boundaries
                            if img[i, j] < img[ni, nj]:
                                outputImage[i, j] = 0
                                break  
                else:
                    continue
                break  
    return outputImage

def process_position(model, img, output_root, filename):
    """
    Executes the neural network inference and topological refinement per FOV.
    """
    try:
        # Secure string manipulation for cross-platform compatibility
        eachImg_path = os.path.join(output_root, filename.replace('.tif', '.png'))
        
        # Execute customized instance segmentation
        masks, _, _ = model.eval(
            img,
            channels=channels,
            **processing_params
        )
        
        # Enforce physical boundaries
        masks = refine_instance_boundaries(masks)
        
        # Export refined instance mask to persistent storage
        io.imsave(eachImg_path, masks, check_contrast=False)       
        return True
        
    except Exception as e:
        print(f"Pipeline Interrupted at {eachImg_path}: {str(e)}")
        return False
    
# ==================== Main Execution Block ====================
if __name__ == "__main__":
    # Instantiate Neural Network
    model = init_model(GPU)
    
    # Initialize Multi-core Progress Dashboard
    total_tasks = len(files_tif) * total_channels
    with tqdm(total=total_tasks, desc="Evaluating Spatial ROIs") as pbar: 
        
        for pos in range(len(files_tif)):
            current_filename = files_tif[pos]
            pos_dir = os.path.join(input_folder0, current_filename)
            
            # Load raw matrix into memory
            img = io.imread(pos_dir)
            
            # Optional Background Subtraction (Currently disabled)
            # img[img < 1.5 * (np.median(img))] = 0
            
            # Trigger execution pipeline
            if process_position(model, img, output_root, current_filename):
                pbar.update(1)