"""
==============================================================================
Spatial Microbiome Distance Mapping & Double-ROI Registration Pipeline
==============================================================================
Purpose: 
    Calculates the absolute physical distance (in micrometers) from individual 
    bacterial single-cell centroids to macroscopic tissue boundaries.

Core Method: 
    1. Extracts feature points to compute Homography matrices, translation offset 
       estimation based on feature registration.
    2. Constructs a cKDTree (K-Dimensional Tree) from the tissue contour mask, 
       executing ultra-fast O(N log N) nearest-neighbor spatial queries.
    3. Excludes false-positive extracellular artifacts via boolean tissue masking.

Key Engineering Tricks:
    1. Memory-Mapped KD-Tree Initialization: Sequentially loads boundary matrices 
       in chunks to circumvent Out-of-Memory (OOM) crashes on gigapixel arrays.
==============================================================================
"""   

import tifffile as tiff
import cv2
import numpy as np
from skimage import measure
import pandas as pd
import matplotlib.pyplot as plt 
import os
from scipy.spatial import cKDTree

class BacteriaBoundaryDistanceAnalyzer:
    """
    Bacterial-to-Tissue Boundary Spatial Distance Analyzer
    """

    def __init__(self, pixel_size_um=0.1):
        self.pixel_size_um = pixel_size_um
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.boundary_tree = None
        self.boundary_shape = None

    # ==================== Image I/O & Memory Management ====================
    def load_roi_u8_gray(self, image_path, roi=None):
        """
        Safely loads explicit ROIs via memory-mapping and normalizes to 8-bit grayscale.
        """
        if roi:
            x0, y0, x1, y1 = map(int, roi)
            print(f"  Reading ROI: x=[{x0}:{x1}], y=[{y0}:{y1}] from {os.path.basename(image_path)}")
        else:
            print(f"  Reading Full FOV: {os.path.basename(image_path)}")

        img_2d = None
        try:
            mm = tiff.memmap(image_path)
            if mm.ndim == 2:
                img_2d = mm
            elif mm.ndim == 3:
                if mm.shape[0] < mm.shape[2]: img_2d = mm[0]
                else: img_2d = mm[..., 0]
        except Exception:
            try:
                full_img = tiff.imread(image_path)
                if full_img.ndim == 3:
                    img_2d = full_img[0] if full_img.shape[0] < full_img.shape[2] else full_img[..., 0]
                else:
                    img_2d = full_img
            except Exception as e:
                print(f"  I/O Failure: {e}")
                return None

        if roi is not None:
            h, w = img_2d.shape
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(h, y1), min(w, x1)
            crop = img_2d[y0:y1, x0:x1]
        else:
            crop = img_2d

        crop = np.array(crop, copy=True)
        # Normalize to 8-bit (0-255) to satisfy OpenCV descriptor requirements
        if crop.dtype != np.uint8:
            crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return crop

    # ==================== Registration Diagnostics ====================
    def visualize_side_by_side_matches(self, img1, kp1, img2, kp2, good_matches, output_dir, base_name):
        """
        Generates diagnostic plots visualizing local spatial correspondence.
        """
        print(f"  Generating matching diagnostic plot (Total matches: {len(good_matches)})...")
        
        import random
        vis_matches = good_matches
        if len(good_matches) > 100:
            vis_matches = random.sample(good_matches, 100)
            print(f"  [Notice] Visualizing a random subset of 100 matches to prevent plot saturation.")
        
        match_img = cv2.drawMatches(
            img1, kp1, 
            img2, kp2, 
            vis_matches, 
            None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0)
        )
        
        # Demarcation line separating the two FOVs
        boundary_x = img1.shape[1]
        h_total = match_img.shape[0]
        cv2.line(match_img, (boundary_x, 0), (boundary_x, h_total), (255, 0, 0), 3)

        save_path = os.path.join(output_dir, f"{base_name}_match_verification_lines.jpg")
        cv2.imwrite(save_path, match_img)
        print(f"  [Verification] Diagnostic plot saved: {os.path.basename(save_path)}")

    # ==================== Geometric Registration ====================
    def detect_and_match_features(self, img1, img2, feature_detector='ORB', min_matches=10, 
                                  visualize=False, output_dir=None, base_name=""):
        """
        Computes the Homography matrix correlating the decoded sub-ROI to the macroscopic slide.
        """
        if img1 is None or img2 is None: return None
        if img1.size == 0 or img2.size == 0: return None

        if feature_detector == 'ORB': detector = cv2.ORB_create(nfeatures=20000)
        elif feature_detector == 'SIFT': detector = cv2.SIFT_create()
        else: detector = cv2.ORB_create(nfeatures=20000)

        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            print("  Warning: Insufficient localized keypoints detected.")
            return None

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) if feature_detector=='SIFT' else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try: matches = matcher.knnMatch(des1, des2, k=2)
        except: return None

        # Lowe's ratio test to filter false-positive ambiguous matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance: good_matches.append(m)

        print(f"  - Robust matches validated: {len(good_matches)}")
        if len(good_matches) < min_matches: return None

        if visualize and output_dir:
            self.visualize_side_by_side_matches(img1, kp1, img2, kp2, good_matches, output_dir, base_name)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Robust Homography estimation utilizing RANSAC paradigm
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return homography

    # ==================== Global Transformation ====================
    def calculate_global_offset(self, homography, roi_decoded, roi_full):
        """
        Translates localized rigid transformation matrix into global coordinate offsets.
        """
        if homography is None: return 0, 0
        
        # Extract direct translational components from the 3x3 Homography matrix
        dx_local = float(homography[0, 2])
        dy_local = float(homography[1, 2])
        
        dec_x0 = roi_decoded[0] if roi_decoded else 0
        dec_y0 = roi_decoded[1] if roi_decoded else 0
        full_x0 = roi_full[0] if roi_full else 0
        full_y0 = roi_full[1] if roi_full else 0
        
        # Absolute shift = Macro coordinate shift + Local feature shift
        self.offset_x = (full_x0 - dec_x0) + dx_local
        self.offset_y = (full_y0 - dec_y0) + dy_local
        
        print("-" * 40)
        print(f"Registration Metrics:")
        print(f"  1. Decoded ROI Origin: ({dec_x0}, {dec_y0})")
        print(f"  2. Full ROI Origin:    ({full_x0}, {full_y0})")
        print(f"  3. Local Feature Shift: dx={dx_local:.2f}, dy={dy_local:.2f}")
        print(f"  => Master Global Offset: x={self.offset_x:.2f}, y={self.offset_y:.2f}")
        print("-" * 40)
        
        return self.offset_x, self.offset_y

    # ==================== Spatial KD-Tree Initialization ====================
    def load_boundary_mask_and_build_tree(self, boundary_mask_path, chunk_rows=4096, boundary_sample_step=1):
        """
        Populates a cKDTree using chunked iteration to prevent RAM exhaustion.
        """
        print(f"Constructing Spatial KD-Tree: {os.path.basename(boundary_mask_path)}")
        try: arr = tiff.memmap(boundary_mask_path)
        except: arr = tiff.imread(boundary_mask_path)
        
        if arr.ndim == 3: arr = arr[..., 0]
        self.boundary_shape = tuple(arr.shape)
        H, W = self.boundary_shape
        
        ys, xs = [], []
        # Chunked iteration to bypass memory constraints on massive arrays
        for y0 in range(0, H, chunk_rows):
            y1 = min(H, y0 + chunk_rows)
            block = arr[y0:y1, :]
            yy, xx = np.nonzero(block)
            if yy.size:
                yy, xx = yy + y0, xx
                if boundary_sample_step > 1:
                    keep = np.arange(0, yy.size, boundary_sample_step)
                    yy, xx = yy[keep], xx[keep]
                ys.append(yy); xs.append(xx)
            if (y0 // chunk_rows) % 50 == 0: print(f"  Scanning chunk: {y1}/{H}", end='\r')
        print("")
        if not ys: raise RuntimeError("Critical Error: Boundary Mask is completely empty.")
        
        pts = np.column_stack([np.concatenate(ys), np.concatenate(xs)]).astype(np.float32)
        self.boundary_tree = cKDTree(pts)
        print("KD-Tree successfully initialized.")
        return self.boundary_tree

    # ==================== Single-Cell Extraction & K-Nearest Neighbors ====================
    def analyze_centroid_map(self, centroid_map_path):
        """
        Extracts single-cell morphological metrics and local centroids from instance masks.
        """
        print(f"Parsing Instance Centroids: {os.path.basename(centroid_map_path)}")
        try: img = tiff.imread(centroid_map_path)
        except: return []
        
        if img.ndim > 2: img = img[0] if img.shape[0]<=3 else img[:,:,0]
        img_int = img.astype(np.int64, copy=False)
        values = np.unique(img_int)
        values = values[values != 0]
        
        all_centroids = []
        for v in values:
            strain = int(v)
            mask = (img_int == v).astype(np.uint8)
            # Evaluate morphological region properties
            regions = measure.regionprops(measure.label(mask, connectivity=2))
            for i, r in enumerate(regions, 1):
                cy, cx = r.centroid
                all_centroids.append({
                    'bacteria_id': strain, 'cell_id_in_file': i,
                    'local_centroid_row': float(cy), 'local_centroid_col': float(cx),
                    'area_pixels': float(r.area), 'area_um2': float(r.area) * (self.pixel_size_um**2),
                    'mask_file': os.path.basename(centroid_map_path)
                })
        return all_centroids

    def calculate_global_centroids(self, local_centroids):
        """Applies spatial offsets to unify coordinates into the macro reference frame."""
        res = []
        for c in local_centroids:
            d = c.copy()
            d['global_centroid_row'] = c['local_centroid_row'] + self.offset_y
            d['global_centroid_col'] = c['local_centroid_col'] + self.offset_x
            res.append(d)
        return res

    def calculate_boundary_distances_wrapper(self, global_centroids):
        """
        Deploys multi-threaded KD-Tree queries to locate the absolute nearest contour point.
        """
        H, W = self.boundary_shape
        pts = np.array([[c['global_centroid_row'], c['global_centroid_col']] for c in global_centroids], dtype=np.float32)
        in_bounds = (pts[:,0]>=0) & (pts[:,0]<H) & (pts[:,1]>=0) & (pts[:,1]<W)
        dist = np.full(len(global_centroids), -1.0, dtype=np.float32)
        
        if np.any(in_bounds):
            try: d, _ = self.boundary_tree.query(pts[in_bounds], k=1, workers=-1)
            except: d, _ = self.boundary_tree.query(pts[in_bounds], k=1)
            dist[in_bounds] = d.astype(np.float32)
        
        results = []
        for i, c in enumerate(global_centroids):
            c['distance_to_boundary_pixels'] = float(dist[i])
            results.append(c)
        return results

    def correct_distances_by_tissue_mask(self, results, filled_mask_path):
        """
        Applies a boolean tissue mask to exclude artifacts located outside the specimen bounds.
        """
        print(f"Applying Boolean Tissue Mask Filter: {os.path.basename(filled_mask_path)}")
        try:
            mask = tiff.memmap(filled_mask_path)
            if mask.ndim == 3: mask = mask[..., 0]
            H, W = mask.shape
            cnt_bg = 0
            
            for c in results:
                r, col = int(round(c['global_centroid_row'])), int(round(c['global_centroid_col']))
                if 0 <= r < H and 0 <= col < W:
                    # If coordinate lands in ambient background
                    if mask[r, col] == 0: 
                        c['distance_to_boundary_pixels'] = 0.0
                        c['distance_to_boundary_um'] = 0.0
                        c['is_on_tissue'] = False
                        cnt_bg += 1
                        continue
                
                # Convert raw pixels to physical micrometers (um)
                orig = c['distance_to_boundary_pixels']
                c['distance_to_boundary_um'] = orig * self.pixel_size_um if orig >= 0 else 0.0
                c['is_on_tissue'] = True
            print(f"  {cnt_bg} centroids localized in background (Distance forced to 0).")
        except Exception as e:
            print(f"Mask filtering interrupted: {e}")
        return results

    # ==================== Data Export & Statistical Visualization ====================
    def generate_full_reports(self, df, output_dir, base_name):
        print("Exporting analytical datasets...")
        
        csv_path = os.path.join(output_dir, f"{base_name}_all_bacteria.csv")
        cols = ['global_cell_id', 'bacteria_id', 'cell_id_in_file', 'mask_file',
                'global_centroid_row', 'global_centroid_col',
                'local_centroid_row', 'local_centroid_col',
                'distance_to_boundary_pixels', 'distance_to_boundary_um',
                'is_on_tissue', 'area_pixels', 'area_um2']
        df_out = df[[c for c in cols if c in df.columns]]
        df_out.to_csv(csv_path, index=False, float_format='%.2f')
        print(f"  - CSV Export: {os.path.basename(csv_path)}")
        
        for s in sorted(df['bacteria_id'].unique()):
            sub = df[df['bacteria_id'] == s]
            txt_path = os.path.join(output_dir, f"{base_name}_strain_{s}_report.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Strain {s} Report\nOffset: x={self.offset_x:.2f}, y={self.offset_y:.2f}\n")
                f.write("-" * 50 + "\n")
                valid = sub[sub['distance_to_boundary_pixels'] > 0]['distance_to_boundary_pixels']
                if len(valid) > 0:
                    f.write(f"Mean: {valid.mean():.2f} px\nMax: {valid.max():.2f} px\nStd: {valid.std():.2f} px\n\n")
                f.write(f"{'ID':<6} {'Row':<10} {'Col':<10} {'Dist(px)':<10} {'OnTissue':<8}\n")
                for _, r in sub.iterrows():
                    ont = "Y" if r.get('is_on_tissue', True) else "N"
                    f.write(f"{int(r['cell_id_in_file']):<6} {r['global_centroid_row']:<10.2f} {r['global_centroid_col']:<10.2f} {r['distance_to_boundary_pixels']:<10.2f} {ont:<8}\n")
        
        self.generate_summary_csv(df, output_dir, base_name)
        self.create_simple_visualization(df, output_dir, base_name)

    def generate_summary_csv(self, df, output_dir, base_name):
        rows = []
        df_valid = df[df['distance_to_boundary_pixels'] >= 0]
        
        dp = df_valid['distance_to_boundary_pixels']
        du = df_valid['distance_to_boundary_um']
        rows.append({
            "bacteria_id": "ALL", "count": len(df_valid),
            "mean_pixels": dp.mean(), "mean_um": du.mean(),
            "median_pixels": dp.median(), "median_um": du.median(),
            "min_pixels": dp.min(), "min_um": du.min(),
            "max_pixels": dp.max(), "max_um": du.max(),
            "std_pixels": dp.std(), "std_um": du.std()
        })
        
        for s in sorted(df_valid['bacteria_id'].unique()):
            sub = df_valid[df_valid['bacteria_id'] == s]
            dp = sub['distance_to_boundary_pixels']
            du = sub['distance_to_boundary_um']
            rows.append({
                "bacteria_id": int(s), "count": len(sub),
                "mean_pixels": dp.mean(), "mean_um": du.mean(),
                "median_pixels": dp.median(), "median_um": du.median(),
                "min_pixels": dp.min(), "min_um": du.min(),
                "max_pixels": dp.max(), "max_um": du.max(),
                "std_pixels": dp.std(), "std_um": du.std()
            })
            
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, f"{base_name}_all_strains_summary.csv"), index=False)
        
        meta = [
            {"key": "pixel_size_um", "value": self.pixel_size_um},
            {"key": "registration_method", "value": "ORB_Homography"},
            {"key": "offset_x_pixels", "value": self.offset_x},
            {"key": "offset_y_pixels", "value": self.offset_y},
            {"key": "roi_decoded", "value": str(self.roi_decoded)},
            {"key": "roi_full", "value": str(self.roi_full)},
            {"key": "total_centroids_all", "value": len(df)}
        ]
        pd.DataFrame(meta).to_csv(os.path.join(output_dir, f"{base_name}_all_strains_summary_meta.csv"), index=False)
        print(f"  - Summary & Meta CSV: Successfully Compiled")

    def create_simple_visualization(self, df, output_dir, base_name):
        valid_df = df[(df['distance_to_boundary_um'] > 0) & (df['distance_to_boundary_um'] < 999999)]
        if len(valid_df) == 0: return

        fig, ax = plt.subplots(1, 2, figsize=(14, 6)) 

        # 1. Histogram (Density Distribution)
        ax[0].hist(valid_df['distance_to_boundary_um'], bins=30, alpha=0.7, edgecolor='k')
        ax[0].set_title('Spatial Distance Distribution')
        ax[0].set_xlabel('Distance (μm)')
        ax[0].set_ylabel('Cell Count')
        mean_val = valid_df['distance_to_boundary_um'].mean()
        median_val = valid_df['distance_to_boundary_um'].median()
        ax[0].axvline(mean_val, color='b', linestyle='--', label=f'Mean: {mean_val:.1f}')
        ax[0].axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.1f}')
        ax[0].legend()
        
        # 2. Boxplot (Stratified by Strain)
        data = []
        labels = []
        for s in sorted(valid_df['bacteria_id'].unique()):
            sub = valid_df[valid_df['bacteria_id']==s]['distance_to_boundary_um']
            if len(sub) > 0:
                data.append(sub)
                labels.append(f"Strain {s}")
        if data:
            ax[1].boxplot(data, labels=labels)
            ax[1].set_title('Spatial Distance Stratified by Strain')
            ax[1].set_xlabel('Biological Strain ID')
            ax[1].set_ylabel('Absolute Distance (μm)')
            # Enforce 45-degree rotation for legibility
            plt.setp(ax[1].get_xticklabels(), rotation=45, ha='right')
            
        plt.suptitle(f'Spatial Microbiome Metrics - Evaluated {len(valid_df)} Distinct Centroids')
        plt.tight_layout()
        
        # Bbox_inches ensures axis labels are not clipped during export
        plt.savefig(os.path.join(output_dir, f"{base_name}_distance_statistics.png"), bbox_inches='tight')
        plt.close()
        print(f"  - Statistical Plots: Successfully Rendered")

    # ==================== Main Orchestrator ====================
    def analyze_bacteria_boundary_distances_from_centroid_map(
        self, 
        centroid_map_path, 
        boundary_mask_path, 
        filled_mask_path, 
        decoded_image_path, 
        full_image_path, 
        roi_decoded, 
        roi_full, 
        output_dir=None
    ):
        self.roi_decoded = roi_decoded
        self.roi_full = roi_full
        print(f"Initiating Spatial Distance Pipeline [Dual-ROI Registration Mode]...")
        base_name = os.path.splitext(os.path.basename(centroid_map_path))[0]
        
        if not output_dir: 
            output_dir = os.path.join(os.path.dirname(centroid_map_path), f"{base_name}_strain_reports")
        os.makedirs(output_dir, exist_ok=True)
        try: 
            with tiff.TiffFile(centroid_map_path) as tif:
                c_shape = tif.series[0].shape
            if len(c_shape) > 2: 
                c_shape = c_shape[:2]
        except Exception as e:
            print(f"Shape metadata extraction failed: {e}")
            c_shape = (0,0)
        
        self.load_boundary_mask_and_build_tree(boundary_mask_path)
        
        is_global = (c_shape == self.boundary_shape)

        if is_global:
            print("Coordinate frameworks strictly aligned. Bypassing dual-ROI registration.")
            self.offset_x, self.offset_y = 0.0, 0.0
        else:
            print("Executing macroscopic Homography registration...")
            img_dec_roi = self.load_roi_u8_gray(decoded_image_path, roi_decoded)
            img_full_roi = self.load_roi_u8_gray(full_image_path, roi_full)
            
            homography = None
            for det in ['ORB', 'SIFT', 'AKAZE']:
                print(f"Attempting keypoint detection via {det} protocol...")
                homography = self.detect_and_match_features(
                    img_dec_roi, img_full_roi, det, 
                    visualize=True, output_dir=output_dir, base_name=base_name
                )
                if homography is not None: break
            
            if homography is None:
                print("Critical Pipeline Failure: Unable to establish spatial homography.")
                return None
            
            self.calculate_global_offset(homography, roi_decoded, roi_full)

        local_centroids = self.analyze_centroid_map(centroid_map_path)
        global_centroids = self.calculate_global_centroids(local_centroids)
        
        results = self.calculate_boundary_distances_wrapper(global_centroids)
        results = self.correct_distances_by_tissue_mask(results, filled_mask_path)
        
        df = pd.DataFrame(results)
        df['global_cell_id'] = range(1, len(df) + 1)
        self.generate_full_reports(df, output_dir, base_name)
        
        print(f"Pipeline Execution Complete! Please review outputs at: {output_dir}")
        return df

if __name__ == "__main__":
    # Define absolute physical pixel scale (um/pixel)
    analyzer = BacteriaBoundaryDistanceAnalyzer(pixel_size_um=0.065)

    # File routing configurations
    centroid_map_path = r"D:\data an\SPF\20250925\BacUnet\labels_centroids.tif"
    boundary_mask_path = r"D:\data an\SPF\20250925\outline\bright_total_8_roi_contour_mask_cv2_merged.tif"
    filled_mask_path = r"D:\data an\SPF\20250925\outline\bright_total_8_roi_filled_mask_cv2_merged.tif"
    decoded_image_path = r"D:\data an\SPF\20250925\outline\bright_8.tif"
    full_image_path = r"D:\data an\SPF\20250925\outline\bright_total_8.tif"

    # Precise Field of View spatial bounds
    roi_decoded = (1184, 4172, 8032,12080) 
    roi_full = (8317, 7868, 15317, 15868)   
    
    # Trigger spatial analysis orchestrator
    analyzer.analyze_bacteria_boundary_distances_from_centroid_map(
        centroid_map_path, 
        boundary_mask_path, 
        filled_mask_path, 
        decoded_image_path, 
        full_image_path, 
        roi_decoded, 
        roi_full
    )