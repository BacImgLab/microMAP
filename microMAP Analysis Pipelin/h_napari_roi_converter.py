"""
==============================================================================
Napari ROI Annotation Conversion Pipeline
==============================================================================
Purpose: 
    Converts hand-drawn vector coordinates (polygons/lines) exported from Napari 
    into absolute binary morphological masks (filled areas and outlines). 
    These masks serve as manual annotation masks for downstream spatial distance 
    analysis.

Core Method: 
    1. Lazy-loads macroscopic TIFF dimensions to construct a blank canvas.
    2. Iteratively stitches fragmented vector segments based on proximity heuristics.
    3. Transposes Napari's native [Y, X] coordinate system to OpenCV's [X, Y] format.
    4. Renders absolute 8-bit masks (filled and contour) mapped to the original WSI.

Key Engineering Tricks:
    1. Fragment Auto-Stitching: Automatically detects and closes disconnected 
       annotation segments caused by manual pen-lifts during drawing.
==============================================================================
"""

import os
import json
import cv2
import numpy as np
from tifffile import imwrite, TiffFile

def convert_napari_roi_to_masks(img_path, coords_json, out_dir=None, connection_tolerance=3.0):
    """
    Parses Napari JSON vectors and generates boolean/uint8 masks for spatial analysis.
    
    Parameters:
        img_path (str): Path to the reference image (used strictly for dimension extraction).
        coords_json (str): Path to the Napari-exported JSON file containing polygons.
        out_dir (str): Destination directory for the generated masks.
        connection_tolerance (float): Pixel threshold to forcefully connect disjoint segments.
    """
    
    print(f"Initiating Napari ROI Conversion...")
    print(f"Reference Image: {os.path.basename(img_path)}")
    print(f"Vector JSON: {os.path.basename(coords_json)}")

    # ==================== Metadata-Only Dimension Extraction ====================
    # [Optimization] Extract macro-array dimensions exclusively via TIFF metadata headers,
    # circumventing whole-slide memory loading to categorically prevent OOM failures.
    try:
        with TiffFile(img_path) as tif:
            c_shape = tif.series[0].shape
            
        # Parse multi-dimensional structures (e.g., C, H, W vs H, W)
        if len(c_shape) > 2:
            H, W = c_shape[-2], c_shape[-1]
        else:
            H, W = c_shape[:2]
        print(f"Target Canvas Dimensions (H, W): ({H}, {W})")
    except Exception as e:
        raise RuntimeError(f"Failed to extract header metadata from {img_path}. Error: {e}")

    # ==================== JSON Vector Parsing & Topology Stitchin ====================
    with open(coords_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Cast extracted coordinates to high-precision NumPy float arrays
    segments = [np.asarray(p, dtype=np.float32) for p in data["polygons"]]
    print(f"Identified Annotation Segments: {len(segments)}")
    
    if len(segments) == 0:
        raise RuntimeError("Critical Error: JSON geometry array is completely empty.")

    # Initialize the master geometric outline with the first segment
    combined = segments[0].copy()

    # Iteratively stitch subsequent disjoint segments caused by manual annotation pen-lifts
    for seg in segments[1:]:
        seg = np.asarray(seg, dtype=np.float32)

        end_prev = combined[-1] # Terminal point of the active concatenated line
        start = seg[0]          # Origin point of the incoming segment
        end = seg[-1]           # Terminal point of the incoming segment

        # Calculate Euclidean distances to determine optimal connection trajectory
        d_start = np.linalg.norm(start - end_prev)
        d_end = np.linalg.norm(end - end_prev)

        if d_start <= d_end:
            # Incoming start point is proximate; retain vector direction
            seg_use = seg
            connect_point = start
        else:
            # Incoming end point is proximate; invert vector direction for continuous loop
            seg_use = seg[::-1]
            connect_point = seg_use[0]

        # Splice vectors depending on proximity threshold
        if np.linalg.norm(connect_point - end_prev) < connection_tolerance:
            # Suppress redundant overlapping node
            combined = np.vstack([combined, seg_use[1:]])
        else:
            print("  ⚠️ Warning: Significant spatial gap detected between segments. Executing forced linear interpolation.")
            combined = np.vstack([combined, seg_use])

    print(f"Master Polygon Synthesized. Total Nodes: {len(combined)}")

    # Evaluate loop closure integrity
    d_close = np.linalg.norm(combined[-1] - combined[0])
    print(f"Terminal-to-Origin spatial deviation: {d_close:.2f} pixels")

    # ==================== Coordinate System Transposition & Rendering ====================
    poly = combined.copy()
    
    # Strictly bound coordinates to canvas dimensions to prevent segmentation faults
    poly[:, 0] = np.clip(poly[:, 0], 0, H - 1)  # Bound Y
    poly[:, 1] = np.clip(poly[:, 1], 0, W - 1)  # Bound X

    # Crucial Conversion: Transpose Napari [Y, X] array to OpenCV [X, Y] array
    poly_xy = np.stack(
        [np.round(poly[:, 1]), np.round(poly[:, 0])],
        axis=1
    ).astype(np.int32)

    polys_cv = [poly_xy]

    # Initialize blank 8-bit canvases
    filled_mask = np.zeros((H, W), dtype=np.uint8)
    contour_mask = np.zeros((H, W), dtype=np.uint8)

    # Render macroscopic Ground Truth geometries
    cv2.fillPoly(filled_mask, polys_cv, 255)
    cv2.polylines(contour_mask, polys_cv, isClosed=True, color=255, thickness=1)

    # ==================== Data Export ====================
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    if out_dir is None:
        out_dir = os.path.dirname(img_path)
    
    os.makedirs(out_dir, exist_ok=True)

    filled_tif = os.path.join(out_dir, f"{base_name}_roi_filled_mask_cv2_merged.tif")
    filled_npy = os.path.join(out_dir, f"{base_name}_roi_filled_mask_cv2_merged.npy")

    contour_tif = os.path.join(out_dir, f"{base_name}_roi_contour_mask_cv2_merged.tif")
    contour_npy = os.path.join(out_dir, f"{base_name}_roi_contour_mask_cv2_merged.npy")

    # Persist spatial masks utilizing efficient binary protocols
    imwrite(filled_tif, filled_mask)
    imwrite(contour_tif, contour_mask)

    np.save(filled_npy, filled_mask.astype(bool))
    np.save(contour_npy, contour_mask.astype(bool))

    print(f"\nPipeline Execution Complete! Masks preserved at: {out_dir}")
    print(f"  Filled ROI Masks:")
    print(f"    - {os.path.basename(filled_tif)}")
    print(f"    - {os.path.basename(filled_npy)}")
    print(f"  Contour ROI Masks:")
    print(f"    - {os.path.basename(contour_tif)}")
    print(f"    - {os.path.basename(contour_npy)}")


if __name__ == "__main__":
    # Configure input directories
    data_dir = r"D:\data an\SPF\20260308\outline" 
    
    # Target files
    img_filename = "bright_total_8.tif"
    json_filename = "intestine_roi_coords.json"
    
    # Synthesize absolute paths
    img_path = os.path.join(data_dir, img_filename)
    coords_json = os.path.join(data_dir, json_filename)
    
    # Execute conversion pipeline
    convert_napari_roi_to_masks(
        img_path=img_path, 
        coords_json=coords_json, 
        connection_tolerance=3.0
    )