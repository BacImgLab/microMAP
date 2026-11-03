"""
==============================================================================
Multi-Channel Spatial Centroid & Morphological Mapping Pipeline
==============================================================================
Purpose: 
    Automates the high-throughput extraction of spatial centroids and 
    morphological masks from multiplexed, stitched TIFF images.

Core Method: 
    Utilizes connected-component labeling and morphological region property 
    computations (via scikit-image) to isolate individual bacterial cells.
    Synthesizes universal coordinate datasets and multi-channel spatial overlays 
    for downstream single-cell spatial profiling.

Key Engineering Tricks:
    1. Maximum Intensity Projection (MIP): Seamlessly integrates multi-channel 
       masks into a master visual canvas.
    2. Zero-Loss Data Extraction: Independently extracts numerical centroid 
       coordinates per channel before rendering, entirely bypassing spatial 
       overwrite artifacts in high-density regions.
==============================================================================
"""

import numpy as np
import tifffile
from skimage import measure
import pandas as pd
import os
import re

def analyze_folder_centroids(input_folder, output_dir=None, pixel_size_um=0.1):
    """
    Automates the high-throughput extraction of spatial centroids and morphological masks 
    from multiplexed, stitched TIFF images. Synthesizes universal coordinate datasets 
    and multi-channel spatial overlays for downstream single-cell spatial analysis.
    """
    print(f"Initiating batch centroid extraction pipeline...")
    print(f"Target Directory: {input_folder}")
    
    # Establish persistent output directory for unified analysis results
    if output_dir is None:
        output_dir = os.path.join(input_folder, "combined_centroid_analysis")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Metadata dictionary for comprehensive report generation
    params = {
        'pixel_size_um': pixel_size_um,
        'input_folder': input_folder,
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 1. Regex-based dynamic file localization
    file_pattern = re.compile(r'channel_(\d+)_stitched\.tif$')
    tif_files = []
    for f in os.listdir(input_folder):
        match = file_pattern.search(f)
        if match:
            tif_files.append((f, int(match.group(1))))
            
    if not tif_files:
        print("Error: No valid images matching 'channel_X_stitched.tif' were found.")
        return None, None, None, output_dir
        
    # Sort files strictly by channel ID to ensure reproducible analytical sequence
    tif_files.sort(key=lambda x: x[1])
    print(f"Successfully identified {len(tif_files)} channel files. Channel IDs: {[f[1] for f in tif_files]}")
    
    all_results = []
    combined_centroid_mask = None
    combined_label_mask = None  
    
    # 2. Sequential morphological iteration across all identified channels
    for file_name, channel_id in tif_files:
        file_path = os.path.join(input_folder, file_name)
        print(f"\nProcessing Channel {channel_id} (File: {file_name})...")
        
        try:
            img = tifffile.imread(file_path)
        except Exception as e:
            print(f"  I/O Failure: {e}. Bypassing current file.")
            continue
            
        # Enforce 2D array structure; collapse multi-dimensional stacks (e.g., RGB) if necessary
        if img.ndim > 2:
            img = img[0] if img.shape[0] <= 3 else img[:,:,0]
            
        # Dynamically initialize master canvases based on the spatial dimensions of the first valid tile
        if combined_centroid_mask is None:
            combined_centroid_mask = np.zeros_like(img, dtype=np.uint16)
        if combined_label_mask is None:
            combined_label_mask = np.zeros_like(img, dtype=np.uint16)
            
        # Binarize current channel and assign a unique biological identifier (Bacteria ID)
        binary_mask = (img > 0).astype(np.uint8)
        bacteria_id = channel_id + 1
        
        # Construct the universal morphological area mask for the current strain
        current_area_mask = binary_mask.astype(np.uint16) * bacteria_id
        
        # Integrate current strain into the master canvas via Maximum Intensity Projection (MIP)
        combined_label_mask = np.maximum(combined_label_mask, current_area_mask)
        
        # Apply 8-connectivity morphological labeling to isolate discrete cellular regions
        labeled_mask = measure.label(binary_mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        print(f"  Channel {channel_id} (Bacteria ID {bacteria_id}) yielded {len(regions)} discrete cellular domains.")
        
        # Extract topological properties for each identified cell
        for i, region in enumerate(regions):
            centroid = region.centroid
            row, col = int(centroid[0]), int(centroid[1])
            
            # Aggregate spatial and morphological metrics into the global dictionary
            all_results.append({
                'bacteria_id': bacteria_id,
                'cell_id': i + 1,
                'centroid_row': row,
                'centroid_col': col,
                'area_pixels': region.area,
                'area_um2': region.area * (pixel_size_um ** 2),
                'equivalent_diameter': region.equivalent_diameter,
                'equivalent_diameter_um': region.equivalent_diameter * pixel_size_um
            })
            
            # Anchor the spatial centroid onto the master centroid canvas
            if 0 <= row < combined_centroid_mask.shape[0] and 0 <= col < combined_centroid_mask.shape[1]:
                combined_centroid_mask[row, col] = bacteria_id

    # 3. Construct and index the unified Pandas DataFrame for downstream statistical evaluation
    df = pd.DataFrame(all_results)
    if len(df) > 0:
        df['global_cell_id'] = range(1, len(df) + 1)
        
    # 4. Data Export / Disk I/O Operations
    centroid_path = os.path.join(output_dir, 'labels_centroids.tif')
    tifffile.imwrite(centroid_path, combined_centroid_mask)
    
    overlay_path = os.path.join(output_dir, 'labels_area_overlay.tif')
    tifffile.imwrite(overlay_path, combined_label_mask)
    
    csv_path = os.path.join(output_dir, 'labels_centroids.csv')
    df.to_csv(csv_path, index=False, float_format='%.2f')
    
    report_path = os.path.join(output_dir, 'combined_report.txt')
    generate_report(df, report_path, params)
    
    print(f"\nPipeline Execution Complete! Total cellular centroids acquired: {len(df)}.")
    print(f"Export directory: {output_dir}")
    print(f"- Spatial Centroid Map: {os.path.basename(centroid_path)}")
    print(f"- Global Area Overlay: {os.path.basename(overlay_path)}")
    print(f"- Unified Metrics CSV: {os.path.basename(csv_path)}")
    
    return df, combined_centroid_mask, combined_label_mask, output_dir


def generate_report(df, report_path, params):
    """
    Generates a structured, human-readable statistical summary report of the spatial analysis.
    """
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Bacterial Spatial Centroid & Morphological Analysis Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Source Directory: {params['input_folder']}\n")
        f.write(f"Timestamp: {params['analysis_date']}\n")
        f.write(f"Optical Pixel Size: {params['pixel_size_um']} μm/pixel\n\n")
        
        f.write("Global Statistical Summary:\n")
        f.write(f"- Total Detected Cells (All Strains): {len(df)}\n")
        
        if len(df) > 0:
            bac_ids = sorted(df['bacteria_id'].unique())
            f.write(f"- Distinct Biological Targets (Bacteria IDs): {len(bac_ids)}\n")
            f.write(f"- Target Strain Array: {bac_ids}\n\n")
            
            f.write("Strain-Specific Morphological Metrics:\n")
            counts = df['bacteria_id'].value_counts().sort_index()
            for bac_id, count in counts.items():
                type_df = df[df['bacteria_id'] == bac_id]
                avg_area = type_df['area_um2'].mean()
                f.write(f"- Strain [ID: {bac_id}]: {count} cells detected | Mean Spatial Area: {avg_area:.2f} μm²\n")
            
            f.write("\nDetailed Morphological Profiling (Top 10 Acquisitions):\n")
            f.write(df.head(10).to_string(index=False))
        else:
            f.write("Evaluation Terminated: No distinct cellular signals detected across all channels.\n")

# ==================== Main Execution Block ====================
if __name__ == "__main__":
    # Define the absolute directory paths generated by the preceding MATLAB stitching pipeline
    input_directory = r"D:\data an\SPF\20250925\BacUnet\stitched_results"  
    output_directory = r"D:\data an\SPF\20250925\BacUnet"
    pixel_size = 0.065  # Physical dimension of a single pixel (in micrometers)
    
    # Execute the centroid evaluation and spatial mapping pipeline
    df, centroid_mask, area_mask, output_dir = analyze_folder_centroids(
        input_directory, 
        output_directory, 
        pixel_size
    )