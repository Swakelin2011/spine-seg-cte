"""
COMPLETE BONE ANALYSIS PIPELINE V10 - ROBUST INTEGRATION
=========================================================
Interactive command-line version with MR-assisted labeling
"""

import os
import sys
import subprocess
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform, axcodes2ornt
from tqdm import tqdm
import json
from datetime import datetime
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_closing, binary_erosion, binary_dilation, sobel
from scipy.ndimage import label as scipy_label, generate_binary_structure
from sklearn.decomposition import PCA
from collections import deque
from functools import partial

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'skip_totalsegmentator': False,
    'use_gpu': True,
    'force_gpu': False,
    'totalseg_workers': 4,
    'analysis_workers': 16,
    'num_layers': 100,
    'min_vertebra_volume': 1000,
    'save_segmentations': True,
    'save_individual_vertebrae': False,
    'save_combined_patient': True,
    'verbose_workers': False,
    'use_plane_pruning': True,
    'min_overlap_fraction': 0.20,
    'min_gap': 0.05,
}

os.environ['TOTALSEG_LICENSE'] = 'aca_P9TM0Y0SECR9WA'

# ============================================================
# VERTEBRAE LABEL MAPPING
# ============================================================

VERTEBRAE_LABELS = {
    26: 'sacrum',
    27: 'L5', 28: 'L4', 29: 'L3', 30: 'L2', 31: 'L1',
    32: 'T12', 33: 'T11', 34: 'T10', 35: 'T9', 36: 'T8', 37: 'T7',
    38: 'T6', 39: 'T5', 40: 'T4', 41: 'T3', 42: 'T2', 43: 'T1',
    44: 'C7', 45: 'C6', 46: 'C5', 47: 'C4', 48: 'C3', 49: 'C2', 50: 'C1'
}

def split_vertebrae_body_by_total_labels(vertebrae_body_path, total_path):
    """
    Split the vertebrae_body segmentation into individual vertebrae using total task labels.
    """
    from nibabel.processing import resample_from_to
    
    # Load both segmentations
    vb_img = nib.load(vertebrae_body_path)
    vb_data = np.asanyarray(vb_img.dataobj).astype(np.int16)
    
    total_img = nib.load(total_path)
    total_data = np.asanyarray(total_img.dataobj).astype(np.int16)
    
    # Ensure same grid (resample if needed)
    if vb_data.shape != total_data.shape or not np.allclose(vb_img.affine, total_img.affine, atol=1e-4):
        total_img = resample_from_to(total_img, vb_img, order=0)
        total_data = np.asanyarray(total_img.dataobj).astype(np.int16)
    
    # Get the vertebrae_body mask (everything labeled as 1)
    vb_mask = (vb_data == 1)
    
    # Find which total labels (26-50) are present in the vertebrae_body region
    total_labels_in_vb = total_data[vb_mask]
    vertebrae_labels_found = [int(l) for l in np.unique(total_labels_in_vb) if 26 <= l <= 50]
    
    # Create individual vertebra masks
    individual_vertebrae = {}
    
    for total_label in vertebrae_labels_found:
        # Get all voxels that are BOTH vertebrae_body AND this specific total label
        individual_mask = vb_mask & (total_data == total_label)
        
        # Check if we have enough voxels
        voxel_count = np.sum(individual_mask)
        if voxel_count < 1000:  # Skip if too small
            continue
        
        # Calculate confidence
        total_vertebra_size = np.sum(total_data == total_label)
        confidence = voxel_count / total_vertebra_size if total_vertebra_size > 0 else 0
        
        anatomical_label = VERTEBRAE_LABELS.get(total_label, f'V{total_label}')
        
        individual_vertebrae[total_label] = {
            'anatomical_label': anatomical_label,
            'mask': individual_mask,
            'confidence': float(confidence),
            'voxel_count': int(voxel_count)
        }
    
    return individual_vertebrae, vb_img

def vprint(*args, **kwargs):
    """Verbose print"""
    if CONFIG.get('verbose_workers', False):
        print(*args, **kwargs)

# ============================================================
# CANONICAL ORIENTATION
# ============================================================

def reorient_to_ras(img):
    """Reorient image to RAS+ canonical space"""
    return nib.as_closest_canonical(img)

def reorient_back_to_original(data_ras, ras_img, orig_img):
    """Reorient processed data from RAS+ back to original orientation"""
    temp_ras = nib.Nifti1Image(data_ras, ras_img.affine, ras_img.header)
    ras_ornt = io_orientation(ras_img.affine)
    orig_ornt = io_orientation(orig_img.affine)
    transform = ornt_transform(ras_ornt, orig_ornt)
    reoriented = temp_ras.as_reoriented(transform)
    return reoriented.get_fdata(), orig_img.affine

# ============================================================
# SIGNED DISTANCE FIELD NORMALS
# ============================================================

def compute_sdf_normals(mask, voxel_spacing):
    """Compute surface normals from signed distance field"""
    edt_outside = distance_transform_edt(~mask, sampling=voxel_spacing)
    edt_inside = distance_transform_edt(mask, sampling=voxel_spacing)
    sdf = edt_outside - edt_inside
    
    grad_x = sobel(sdf, axis=0) / voxel_spacing[0]
    grad_y = sobel(sdf, axis=1) / voxel_spacing[1]
    grad_z = sobel(sdf, axis=2) / voxel_spacing[2]
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2) + 1e-8
    
    normal_x = grad_x / grad_mag
    normal_y = grad_y / grad_mag
    normal_z = grad_z / grad_mag
    
    return normal_x, normal_y, normal_z, grad_mag

def orient_normals_outward(normal_x, normal_y, normal_z, vertebra_mask):
    """Ensure normals point outward from vertebra"""
    surface = vertebra_mask & ~binary_erosion(vertebra_mask)
    surface_coords = np.where(surface)
    
    if len(surface_coords[0]) == 0:
        return normal_x, normal_y, normal_z
    
    sample_size = min(100, len(surface_coords[0]))
    indices = np.random.choice(len(surface_coords[0]), sample_size, replace=False)
    
    outward_count = 0
    for idx in indices:
        x, y, z = surface_coords[0][idx], surface_coords[1][idx], surface_coords[2][idx]
        nx, ny, nz = normal_x[x, y, z], normal_y[x, y, z], normal_z[x, y, z]
        step_size = 2
        
        new_x = int(np.clip(x + nx * step_size, 0, vertebra_mask.shape[0] - 1))
        new_y = int(np.clip(y + ny * step_size, 0, vertebra_mask.shape[1] - 1))
        new_z = int(np.clip(z + nz * step_size, 0, vertebra_mask.shape[2] - 1))
        
        if not vertebra_mask[new_x, new_y, new_z]:
            outward_count += 1
    
    if outward_count < sample_size * 0.5:
        return -normal_x, -normal_y, -normal_z
    
    return normal_x, normal_y, normal_z

# ============================================================
# CONSTRAINED REGION GROWING
# ============================================================

def region_grow_constrained(seeds, cortical_mask, normal_x, normal_y, normal_z, 
                           target_normal, normal_threshold, max_thickness_mm, 
                           voxel_spacing, vertebra_mask):
    """Region grow with normal alignment and thickness constraints"""
    result = np.zeros_like(seeds, dtype=bool)
    result[seeds] = True
    
    seed_distance = distance_transform_edt(~seeds, sampling=voxel_spacing)
    
    queue = deque()
    seed_coords = np.where(seeds)
    for i in range(len(seed_coords[0])):
        queue.append((seed_coords[0][i], seed_coords[1][i], seed_coords[2][i]))
    
    visited = np.zeros_like(seeds, dtype=bool)
    visited[seeds] = True
    
    neighbors = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    
    while queue:
        x, y, z = queue.popleft()
        
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            if not (0 <= nx < result.shape[0] and 
                   0 <= ny < result.shape[1] and 
                   0 <= nz < result.shape[2]):
                continue
            
            if visited[nx, ny, nz]:
                continue
            
            visited[nx, ny, nz] = True
            
            if not (cortical_mask[nx, ny, nz] and vertebra_mask[nx, ny, nz]):
                continue
            
            if seed_distance[nx, ny, nz] > max_thickness_mm:
                continue
            
            normal = np.array([normal_x[nx, ny, nz], 
                              normal_y[nx, ny, nz], 
                              normal_z[nx, ny, nz]])
            
            dot_product = np.dot(normal, target_normal)
            
            if dot_product > normal_threshold:
                result[nx, ny, nz] = True
                queue.append((nx, ny, nz))
    
    return result

# ============================================================
# PLANARITY & RANSAC
# ============================================================

def prune_mask_to_best_fit_plane(mask, voxel_spacing, max_plane_dist_mm=3.5, min_size=80, 
                                 use_ransac=True, ransac_threshold_mm=2.0):
    """RANSAC plane fitting to remove non-planar structures"""
    coords = np.array(np.where(mask)).T
    if coords.shape[0] < 20:
        return mask
    
    coords_mm = coords * np.array(voxel_spacing)[None, :]
    
    if use_ransac:
        best_inliers = []
        best_plane_normal = None
        best_plane_point = None
        max_inliers = 0
        
        for _ in range(100):
            if len(coords_mm) < 3:
                break
            sample_indices = np.random.choice(len(coords_mm), 3, replace=False)
            sample_points = coords_mm[sample_indices]
            
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
            
            normal = normal / np.linalg.norm(normal)
            plane_point = sample_points[0]
            
            distances = np.abs((coords_mm - plane_point) @ normal)
            inliers = distances < ransac_threshold_mm
            n_inliers = np.sum(inliers)
            
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                best_inliers = inliers
                best_plane_normal = normal
                best_plane_point = plane_point
        
        if best_plane_normal is None:
            use_ransac = False
        else:
            inlier_points = coords_mm[best_inliers]
            centroid = inlier_points.mean(axis=0)
            centered = inlier_points - centroid
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            plane_normal = eigenvectors[:, 0]
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    if not use_ransac:
        centroid = coords_mm.mean(axis=0)
        pca = PCA(n_components=3)
        pca.fit(coords_mm)
        plane_normal = pca.components_[2]
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    
    point_to_plane_dist = np.abs((coords_mm - centroid) @ plane_normal)
    keep_indices = point_to_plane_dist <= max_plane_dist_mm
    
    pruned = np.zeros_like(mask, dtype=bool)
    pruned[coords[keep_indices, 0], coords[keep_indices, 1], coords[keep_indices, 2]] = True
    
    pruned = keep_large_components(pruned, min_size=min_size)
    pruned = keep_largest_component(pruned)
    
    return pruned

def keep_large_components(mask, min_size=50):
    """Keep all components above minimum size"""
    if np.sum(mask) == 0:
        return mask
    
    labeled, num_components = scipy_label(mask)
    if num_components == 0:
        return mask
    
    filtered = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_components + 1):
        component = (labeled == i)
        if np.sum(component) >= min_size:
            filtered |= component
    
    return filtered

def keep_largest_component(mask):
    """Keep only the largest connected component"""
    if np.sum(mask) == 0:
        return mask
    
    labeled, num_components = scipy_label(mask)
    if num_components <= 1:
        return mask
    
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = np.argmax(sizes)
    return (labeled == largest_label)

# ============================================================
# CORE SEGMENTATION
# ============================================================

def compute_distance_from_surface(mask, voxel_spacing=None):
    """Compute distance from outer surface"""
    if voxel_spacing is not None:
        distance = distance_transform_edt(mask, sampling=voxel_spacing)
    else:
        distance = distance_transform_edt(mask)
    return distance

def get_radial_hu_profile(ct_data, mask, voxel_spacing, num_layers=50):
    """Create radial profiles from outside to inside"""
    distance_map = compute_distance_from_surface(mask, voxel_spacing)
    max_dist = distance_map.max()
    layer_thickness = max_dist / num_layers
    
    profiles = []
    distances = []
    
    for i in range(num_layers):
        inner_dist = i * layer_thickness
        outer_dist = (i + 1) * layer_thickness
        layer_mask = (distance_map >= inner_dist) & (distance_map < outer_dist) & (mask > 0)
        
        if np.sum(layer_mask) > 0:
            hu_values = ct_data[layer_mask]
            profiles.append({
                'distance': (inner_dist + outer_dist) / 2,
                'mean_hu': np.mean(hu_values),
                'std_hu': np.std(hu_values),
                'voxel_count': np.sum(layer_mask)
            })
            distances.append((inner_dist + outer_dist) / 2)
    
    return profiles, distances

def compute_hu_gradient(profiles):
    """Compute gradient of HU values"""
    distances = [p['distance'] for p in profiles]
    hu_values = [p['mean_hu'] for p in profiles]
    gradient = np.gradient(hu_values, distances)
    return gradient

def adaptive_segment_using_gradient(ct_data, mask, voxel_spacing, num_layers=100):
    """Use gradient analysis to find cortical-trabecular boundary"""
    profiles, distances = get_radial_hu_profile(ct_data, mask, voxel_spacing, num_layers)
    gradient = compute_hu_gradient(profiles)
    gradient_smooth = gaussian_filter(gradient, sigma=2)
    
    second_derivative = np.gradient(gradient_smooth)
    transition_idx = np.argmax(np.abs(second_derivative[:len(second_derivative)//2]))
    transition_distance = distances[transition_idx]
    
    distance_map = compute_distance_from_surface(mask, voxel_spacing)
    cortical_mask = (distance_map > 0) & (distance_map <= transition_distance) & (mask > 0)
    trabecular_mask = (distance_map > transition_distance) & (mask > 0)
    
    return {
        'cortical': cortical_mask,
        'trabecular': trabecular_mask,
        'transition_distance': transition_distance,
        'max_distance': distance_map.max(),
        'profiles': profiles,
        'gradient': gradient_smooth,
        'distances': distances
    }

# ============================================================
# ENDPLATE DETECTION
# ============================================================

def identify_endplate_surface_points(vertebra_mask, cortical_mask, ct_data, voxel_spacing, 
                                    normal_x, normal_y, normal_z, is_sacrum_flag=False):
    """Robust endplate surface point identification"""
    
    outer_surface = vertebra_mask & ~binary_erosion(vertebra_mask)
    outer_cortical_surface = outer_surface & cortical_mask
    
    coords = np.array(np.where(vertebra_mask)).T
    z_coords = np.where(vertebra_mask)[2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_range = z_max - z_min
    
    if is_sacrum_flag:
        superior_z_start = z_max - 0.35 * z_range
    else:
        superior_z_start = z_max - 0.20 * z_range
        inferior_z_end = z_min + 0.20 * z_range
    
    z_grid = np.arange(vertebra_mask.shape[2])
    superior_region = (z_grid >= superior_z_start) & (z_grid <= z_max)
    superior_cortical = outer_cortical_surface & superior_region[None, None, :]
    
    if not is_sacrum_flag:
        inferior_region = (z_grid >= z_min) & (z_grid <= inferior_z_end)
        inferior_cortical = outer_cortical_surface & inferior_region[None, None, :]
    else:
        inferior_cortical = np.zeros_like(superior_cortical, dtype=bool)
    
    grad_x_ct = sobel(ct_data, axis=0) / voxel_spacing[0]
    grad_y_ct = sobel(ct_data, axis=1) / voxel_spacing[1]
    grad_z_ct = sobel(ct_data, axis=2) / voxel_spacing[2]
    grad_mag = np.sqrt(grad_x_ct**2 + grad_y_ct**2 + grad_z_ct**2)
    
    vertebra_gradients = grad_mag[vertebra_mask]
    gradient_threshold = np.percentile(vertebra_gradients, 75)
    
    superior_candidates = (
        superior_cortical &
        (normal_z > 0.5) &
        (grad_mag > gradient_threshold * 0.7)
    )
    
    if not is_sacrum_flag:
        inferior_candidates = (
            inferior_cortical &
            (normal_z < -0.4) &
            (grad_mag > gradient_threshold * 0.7)
        )
    else:
        inferior_candidates = np.zeros_like(superior_candidates, dtype=bool)
    
    if np.sum(superior_candidates) < 50:
        z_extreme_superior = z_max
        extreme_superior = (
            superior_cortical &
            (z_grid[None, None, :] >= z_extreme_superior - 3) &
            (grad_mag > gradient_threshold * 0.5)
        )
        superior_candidates = superior_candidates | extreme_superior
    
    if not is_sacrum_flag and np.sum(inferior_candidates) < 50:
        z_extreme_inferior = z_min
        extreme_inferior = (
            inferior_cortical &
            (z_grid[None, None, :] <= z_extreme_inferior + 3) &
            (grad_mag > gradient_threshold * 0.5)
        )
        inferior_candidates = inferior_candidates | extreme_inferior
    
    superior_candidates = keep_large_components(superior_candidates, min_size=5)
    if not is_sacrum_flag:
        inferior_candidates = keep_large_components(inferior_candidates, min_size=5)
    
    return superior_candidates, inferior_candidates

def grow_endplate_constrained(seeds, cortical_mask, normal_x, normal_y, normal_z,
                              voxel_spacing, vertebra_mask, direction='superior'):
    """Grow endplate using constrained region growing"""
    if np.sum(seeds) == 0:
        return seeds
    
    if direction == 'superior':
        target_normal = np.array([0, 0, 1])
        normal_threshold = 0.5
    else:
        target_normal = np.array([0, 0, -1])
        normal_threshold = 0.5
    
    max_thickness_mm = 6.0
    
    endplate = region_grow_constrained(
        seeds, cortical_mask, normal_x, normal_y, normal_z,
        target_normal, normal_threshold, max_thickness_mm,
        voxel_spacing, vertebra_mask
    )
    
    return endplate

def segment_endplates_by_normals(ct_data, vertebra_mask, cortical_mask, trabecular_mask, 
                                voxel_spacing, is_sacrum_flag=False, 
                                use_plane_pruning=True):
    """Robust endplate segmentation"""
    
    normal_x, normal_y, normal_z, _ = compute_sdf_normals(vertebra_mask, voxel_spacing)
    normal_x, normal_y, normal_z = orient_normals_outward(normal_x, normal_y, normal_z, vertebra_mask)
    
    superior_seeds, inferior_seeds = identify_endplate_surface_points(
        vertebra_mask, cortical_mask, ct_data, voxel_spacing,
        normal_x, normal_y, normal_z, is_sacrum_flag
    )
    
    if np.sum(superior_seeds) < 10:
        return None
    
    if not is_sacrum_flag and np.sum(inferior_seeds) < 10:
        return None
    
    superior_endplate = grow_endplate_constrained(
        superior_seeds, cortical_mask, normal_x, normal_y, normal_z,
        voxel_spacing, vertebra_mask, direction='superior'
    )
    
    if not is_sacrum_flag:
        inferior_endplate = grow_endplate_constrained(
            inferior_seeds, cortical_mask, normal_x, normal_y, normal_z,
            voxel_spacing, vertebra_mask, direction='inferior'
        )
    else:
        inferior_endplate = np.zeros_like(superior_endplate, dtype=bool)
    
    if use_plane_pruning and is_sacrum_flag and np.sum(superior_endplate) > 0:
        superior_endplate = prune_mask_to_best_fit_plane(
            superior_endplate, voxel_spacing,
            max_plane_dist_mm=3.5,
            min_size=80,
            use_ransac=True,
            ransac_threshold_mm=2.0
        )
    
    struct = generate_binary_structure(3, 1)
    superior_endplate = binary_closing(superior_endplate, structure=struct, iterations=1)
    if not is_sacrum_flag:
        inferior_endplate = binary_closing(inferior_endplate, structure=struct, iterations=1)
    
    superior_endplate = superior_endplate & vertebra_mask
    inferior_endplate = inferior_endplate & vertebra_mask
    
    if not is_sacrum_flag:
        overlap = superior_endplate & inferior_endplate
        if np.any(overlap):
            z_coords = np.where(vertebra_mask)[2]
            z_mid = (z_coords.min() + z_coords.max()) / 2
            overlap_coords = np.where(overlap)
            for i in range(len(overlap_coords[0])):
                z = overlap_coords[2][i]
                if z > z_mid:
                    inferior_endplate[overlap_coords[0][i], overlap_coords[1][i], z] = False
                else:
                    superior_endplate[overlap_coords[0][i], overlap_coords[1][i], z] = False
    
    sup_z = np.where(superior_endplate)[2]
    if len(sup_z) > 0:
        # VECTORIZED - much faster!
        any_xy = superior_endplate.any(axis=2)
        top_z_indices = superior_endplate.shape[2] - 1 - np.argmax(superior_endplate[:, :, ::-1], axis=2)
        
        outer_surface_superior = np.zeros_like(superior_endplate, dtype=bool)
        y_coords, x_coords = np.where(any_xy)
        outer_surface_superior[x_coords, y_coords, top_z_indices[x_coords, y_coords]] = True
        
        dist_from_outer_sup = distance_transform_edt(~outer_surface_superior, sampling=voxel_spacing)
        superior_thickness = np.max(dist_from_outer_sup[superior_endplate])
    else:
        superior_thickness = 0

    if not is_sacrum_flag:
        inf_z = np.where(inferior_endplate)[2]
        if len(inf_z) > 0:
            # VECTORIZED - much faster!
            any_xy_inf = inferior_endplate.any(axis=2)
            bottom_z_indices = np.argmax(inferior_endplate, axis=2)
            
            outer_surface_inferior = np.zeros_like(inferior_endplate, dtype=bool)
            y_coords_inf, x_coords_inf = np.where(any_xy_inf)
            outer_surface_inferior[x_coords_inf, y_coords_inf, bottom_z_indices[x_coords_inf, y_coords_inf]] = True
            
            dist_from_outer_inf = distance_transform_edt(~outer_surface_inferior, sampling=voxel_spacing)
            inferior_thickness = np.max(dist_from_outer_inf[inferior_endplate])
        else:
            inferior_thickness = 0
    else:
        inferior_thickness = 0
    
    z_coords = np.where(vertebra_mask)[2]
    vertebra_height_mm = (z_coords.max() - z_coords.min()) * voxel_spacing[2]
    
    return {
        'superior_endplate': superior_endplate,
        'inferior_endplate': inferior_endplate,
        'superior_thickness_mm': superior_thickness,
        'inferior_thickness_mm': inferior_thickness,
        'vertebra_height_mm': vertebra_height_mm
    }

# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================


def extract_vertebra_metrics(ct_data, vertebra_mask_input, ct_img_orig, ct_img, voxel_spacing, 
                            num_layers, is_sacrum_flag, use_plane_pruning):
    """
    Extract metrics for a single vertebra.
    OPTIMIZED: Now accepts ct_data directly instead of loading it.
    """
    try:
        # ct_data is now passed in - NO get_fdata() call!
        vertebra_mask = vertebra_mask_input
        
        if np.sum(vertebra_mask) < CONFIG['min_vertebra_volume']:
            return None
        
        # OPTIMIZATION 3: Crop to bounding box (speeds up everything)
        coords = np.array(np.where(vertebra_mask)).T
        mins = np.maximum(coords.min(axis=0) - 5, 0)
        maxs = np.minimum(coords.max(axis=0) + 6, vertebra_mask.shape)
        slc = tuple(slice(int(mins[d]), int(maxs[d])) for d in range(3))
        
        # Work on cropped data
        ct_data_crop = ct_data[slc]
        vertebra_mask_crop = vertebra_mask[slc]
        
        cortical_result = adaptive_segment_using_gradient(
            ct_data_crop, vertebra_mask_crop, voxel_spacing, num_layers
        )
        
        cortical_mask_crop = cortical_result['cortical']
        trabecular_mask_crop = cortical_result['trabecular']
        
        endplate_result = segment_endplates_by_normals(
            ct_data_crop, vertebra_mask_crop, cortical_mask_crop, trabecular_mask_crop, 
            voxel_spacing, is_sacrum_flag, use_plane_pruning
        )
        
        if endplate_result is None:
            return None
        
        superior_endplate_crop = endplate_result['superior_endplate']
        inferior_endplate_crop = endplate_result['inferior_endplate']
        
        # Paste back to full size
        superior_endplate = np.zeros_like(vertebra_mask, dtype=bool)
        inferior_endplate = np.zeros_like(vertebra_mask, dtype=bool)
        cortical_mask = np.zeros_like(vertebra_mask, dtype=bool)
        trabecular_mask = np.zeros_like(vertebra_mask, dtype=bool)
        
        superior_endplate[slc] = superior_endplate_crop
        inferior_endplate[slc] = inferior_endplate_crop
        cortical_mask[slc] = cortical_mask_crop
        trabecular_mask[slc] = trabecular_mask_crop
        
        cortical_final = cortical_mask & ~superior_endplate & ~inferior_endplate
        trabecular_final = trabecular_mask & ~superior_endplate & ~inferior_endplate
        
        cortical_hu = ct_data[cortical_final]
        trabecular_hu = ct_data[trabecular_final]
        superior_hu = ct_data[superior_endplate]
        inferior_hu = ct_data[inferior_endplate]
        
        metrics = {
            'success': True,
            'cortical_thickness_mm': float(cortical_result['transition_distance']),
            'cortical_volume_voxels': int(np.sum(cortical_final)),
            'cortical_volume_mm3': float(np.sum(cortical_final) * np.prod(voxel_spacing)),
            'cortical_mean_hu': float(np.mean(cortical_hu)) if len(cortical_hu) > 0 else 0,
            'cortical_std_hu': float(np.std(cortical_hu)) if len(cortical_hu) > 0 else 0,
            'cortical_min_hu': float(np.min(cortical_hu)) if len(cortical_hu) > 0 else 0,
            'cortical_max_hu': float(np.max(cortical_hu)) if len(cortical_hu) > 0 else 0,
            'trabecular_volume_voxels': int(np.sum(trabecular_final)),
            'trabecular_volume_mm3': float(np.sum(trabecular_final) * np.prod(voxel_spacing)),
            'trabecular_mean_hu': float(np.mean(trabecular_hu)) if len(trabecular_hu) > 0 else 0,
            'trabecular_std_hu': float(np.std(trabecular_hu)) if len(trabecular_hu) > 0 else 0,
            'trabecular_min_hu': float(np.min(trabecular_hu)) if len(trabecular_hu) > 0 else 0,
            'trabecular_max_hu': float(np.max(trabecular_hu)) if len(trabecular_hu) > 0 else 0,
            'superior_thickness_mm': float(endplate_result['superior_thickness_mm']),
            'superior_volume_voxels': int(np.sum(superior_endplate)),
            'superior_volume_mm3': float(np.sum(superior_endplate) * np.prod(voxel_spacing)),
            'superior_mean_hu': float(np.mean(superior_hu)) if len(superior_hu) > 0 else 0,
            'superior_std_hu': float(np.std(superior_hu)) if len(superior_hu) > 0 else 0,
            'superior_low_density_pct': float(100 * np.sum(superior_hu < 200) / len(superior_hu)) if len(superior_hu) > 0 else 0,
            'inferior_thickness_mm': float(endplate_result['inferior_thickness_mm']),
            'inferior_volume_voxels': int(np.sum(inferior_endplate)),
            'inferior_volume_mm3': float(np.sum(inferior_endplate) * np.prod(voxel_spacing)),
            'inferior_mean_hu': float(np.mean(inferior_hu)) if len(inferior_hu) > 0 else 0,
            'inferior_std_hu': float(np.std(inferior_hu)) if len(inferior_hu) > 0 else 0,
            'inferior_low_density_pct': float(100 * np.sum(inferior_hu < 200) / len(inferior_hu)) if len(inferior_hu) > 0 else 0,
            'total_volume_voxels': int(np.sum(cortical_final) + np.sum(trabecular_final) + 
                                      np.sum(superior_endplate) + np.sum(inferior_endplate)),
            'total_volume_mm3': float((np.sum(cortical_final) + np.sum(trabecular_final) + 
                                      np.sum(superior_endplate) + np.sum(inferior_endplate)) * np.prod(voxel_spacing)),
            'vertebra_height_mm': float(endplate_result['vertebra_height_mm']),
            '_masks_ras': {
                'cortical': cortical_final,
                'trabecular': trabecular_final,
                'superior_endplate': superior_endplate,
                'inferior_endplate': inferior_endplate
            },
            '_ct_img': ct_img,
            '_ct_img_orig': ct_img_orig
        }
        
        return metrics
        
    except Exception as e:
        return {'success': False, 'error': str(e)[:200]}


# ============================================================
# FILE MANAGEMENT
# ============================================================

def discover_scans(input_dir):
    """Find all CT scans"""
    input_path = Path(input_dir).resolve()
    scans = []
    
    for scan_path in sorted(input_path.glob('input.*.nii.gz')):
        scan_id = scan_path.stem.split('.')[1]
        scans.append({
            'patient_id': scan_id,
            'ct_path': str(scan_path.resolve()),
        })
    
    return scans

def setup_output_structure(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dirs = {
        'results': output_path / 'results',
        'segmentations': output_path / 'segmentations',
        'vertebrae_seg': output_path / 'vertebrae_segmentations'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs

def save_segmentation(masks_ras, ct_img, ct_img_orig, output_dir, patient_id, vertebra_label):
    """Save complete 5-label segmentation in original orientation"""
    combined = np.zeros_like(masks_ras['cortical'], dtype=np.uint8)
    combined[masks_ras['trabecular']] = 2
    combined[masks_ras['cortical']] = 1
    combined[masks_ras['superior_endplate']] = 3
    combined[masks_ras['inferior_endplate']] = 4
    
    combined_orig, orig_affine = reorient_back_to_original(combined, ct_img, ct_img_orig)
    
    output_img = nib.Nifti1Image(combined_orig.astype(np.uint8), orig_affine, ct_img_orig.header)
    output_path = Path(output_dir) / f'{patient_id}_{vertebra_label}_5labels.nii.gz'
    nib.save(output_img, output_path)

def save_combined_patient_segmentation(all_vertebrae_masks, ct_img, ct_img_orig, output_dir, patient_id):
    """Save one combined NIFTI per patient with all vertebrae labeled"""
    if not all_vertebrae_masks:
        return
    
    combined = np.zeros_like(all_vertebrae_masks[0]['masks_ras']['cortical'], dtype=np.uint16)
    
    for vert_data in all_vertebrae_masks:
        vertebra_num = vert_data['vertebra_num']
        masks = vert_data['masks_ras']
        
        base_label = vertebra_num * 10
        
        combined[masks['cortical']] = base_label + 1
        combined[masks['trabecular']] = base_label + 2
        combined[masks['superior_endplate']] = base_label + 3
        combined[masks['inferior_endplate']] = base_label + 4
    
    combined_orig, orig_affine = reorient_back_to_original(combined, ct_img, ct_img_orig)
    
    output_img = nib.Nifti1Image(combined_orig.astype(np.uint16), orig_affine, ct_img_orig.header)
    output_path = Path(output_dir) / f'{patient_id}_all_vertebrae_segmented.nii.gz'
    nib.save(output_img, output_path)
    
    label_info = {}
    for vert_data in all_vertebrae_masks:
        vertebra_num = vert_data['vertebra_num']
        anatomical_label = vert_data['anatomical_label']
        base_label = vertebra_num * 10
        
        label_info[anatomical_label] = {
            'vertebra_number': vertebra_num,
            'cortical': base_label + 1,
            'trabecular': base_label + 2,
            'superior_endplate': base_label + 3,
            'inferior_endplate': base_label + 4
        }
    
    label_json_path = Path(output_dir) / f'{patient_id}_label_mapping.json'
    with open(label_json_path, 'w') as f:
        json.dump(label_info, f, indent=2)

# ============================================================
# TOTALSEGMENTATOR - CANONICAL PATTERN
# ============================================================
# Replace the run_totalseg_single function with this corrected version:

# Replace the run_totalseg_single function with this corrected version:

def run_totalseg_single(scan_info):
    """
    Run TotalSegmentator following canonical output pattern:
    OUTPUT_ROOT/PATIENT_ID/vertebrae_body/vertebrae_body.nii.gz
    OUTPUT_ROOT/PATIENT_ID/total.nii.gz (combined multilabel file)
    """
    patient_id = scan_info['patient_id']
    ct_path = scan_info['ct_path']
    output_dir = scan_info['output_dir']
    use_gpu = scan_info.get('use_gpu', False)
    force_gpu = scan_info.get('force_gpu', False)
    
    patient_dir = Path(output_dir) / patient_id
    vertebrae_body_dir = patient_dir / 'vertebrae_body'
    
    vertebrae_body_file = vertebrae_body_dir / 'vertebrae_body.nii.gz'
    total_file = patient_dir / 'total.nii.gz'  # Combined file, not in subdirectory
    
    device = 'gpu' if use_gpu else 'cpu'
    
    env = os.environ.copy()
    if force_gpu:
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    try:
        import time
        
        # ===== TASK 1: vertebrae_body =====
        if not vertebrae_body_file.exists():
            print(f"\n  {patient_id}: Running vertebrae_body...", end='')
            sys.stdout.flush()
            
            cmd_body = [
                'TotalSegmentator', 
                '-i', ct_path, 
                '-o', str(vertebrae_body_dir),
                '--task', 'vertebrae_body',
                '--ml',
                '--device', device
            ]
            
            start_time = time.time()
            
            try:
                result = subprocess.run(cmd_body, capture_output=True, text=True, timeout=1800, env=env)
                elapsed = time.time() - start_time
            except subprocess.TimeoutExpired:
                print(f" TIMEOUT")
                return {'error': 'vertebrae_body timeout'}
            
            if result.returncode != 0:
                if device == 'gpu' and not force_gpu:
                    cmd_body[cmd_body.index('--device') + 1] = 'cpu'
                    result = subprocess.run(cmd_body, capture_output=True, text=True, timeout=1800, env=env)
                    elapsed = time.time() - start_time
                    if result.returncode != 0:
                        print(f" FAILED")
                        return {'error': f"vertebrae_body failed: {result.stderr[-200:]}"}
                else:
                    print(f" FAILED")
                    return {'error': f"vertebrae_body failed: {result.stderr[-200:]}"}
            
            # Check if file exists at expected location
            if not vertebrae_body_file.exists():
                # First, look in the vertebrae_body subdirectory
                found_files = list(vertebrae_body_dir.glob('*.nii*'))
                
                # If not found, look in the parent patient directory
                if not found_files:
                    found_files = list(patient_dir.glob('*.nii*'))
                
                if found_files:
                    vertebrae_body_file = found_files[0]
                    print(f" ⚠️  Using file from unexpected location: {vertebrae_body_file.relative_to(patient_dir)}")
                else:
                    print(f" NO OUTPUT")
                    return {'error': f'No output file in {vertebrae_body_dir} or {patient_dir}'}
            
            # Verify the file actually exists now
            if vertebrae_body_file.exists():
                size_mb = vertebrae_body_file.stat().st_size / (1024*1024)
                print(f" ✓ {elapsed:.1f}s ({size_mb:.1f}MB)")
            else:
                print(f" FILE NOT FOUND")
                return {'error': f'File does not exist: {vertebrae_body_file}'}
        else:
            print(f"  {patient_id}: ✓ Existing vertebrae_body")
            # Still need to verify the file exists
            if not vertebrae_body_file.exists():
                # Look in vertebrae_body subdirectory first
                found_files = list(vertebrae_body_dir.glob('*.nii*'))
                
                # If not found, look in parent patient directory
                if not found_files:
                    found_files = list(patient_dir.glob('*.nii*'))
                
                if found_files:
                    vertebrae_body_file = found_files[0]
                else:
                    return {'error': f'vertebrae_body file not found in {vertebrae_body_dir} or {patient_dir}'}
        
        # ===== TASK 2: total (for combined multilabel segmentation) =====
        total_result = None
        
        if not total_file.exists():
            print(f"  {patient_id}: Running total task...", end='')
            sys.stdout.flush()
            
            # Use --ml flag to get multilabel output (single combined file)
            cmd_total = [
                'TotalSegmentator', 
                '-i', ct_path, 
                '-o', str(total_file),  # Direct output file path
                '--task', 'total',
                '--ml',  # CRITICAL: multilabel output (single file)
                '--device', device
            ]
            
            start_time = time.time()
            
            try:
                result = subprocess.run(cmd_total, capture_output=True, text=True, timeout=1800, env=env)
                elapsed = time.time() - start_time
            except subprocess.TimeoutExpired:
                print(f" TIMEOUT (skipping)")
                total_result = None
            else:
                if result.returncode == 0:
                    if total_file.exists():
                        size_mb = total_file.stat().st_size / (1024*1024)
                        print(f" ✓ {elapsed:.1f}s ({size_mb:.1f}MB)")
                        total_result = str(total_file)
                    else:
                        print(f" NO OUTPUT (skipping)")
                else:
                    print(f" FAILED (skipping)")
                    print(f"    Error: {result.stderr[-200:]}")
        else:
            print(f"  {patient_id}: ✓ Existing total")
            total_result = str(total_file)
        
        return {
            'vertebrae_body': str(vertebrae_body_file),
            'vertebrae_levels': total_result
        }
        
    except Exception as e:
        import traceback
        print(f"\n  {patient_id}: ❌ Exception: {str(e)}")
        traceback.print_exc()
        return {'error': str(e)[:200]}


    if not CONFIG.get('skip_totalsegmentator', False):
        scans = run_totalseg_parallel(scans, output_dirs['vertebrae_seg'], CONFIG['totalseg_workers'])
    else:
        print("\n[2/4] Skipping TotalSegmentator (using existing segmentations)...")
        
        for scan in scans:
            patient_id = scan['patient_id']
            patient_dir = Path(output_dirs['vertebrae_seg']) / patient_id
            
            vertebrae_body_file = patient_dir / 'vertebrae_body' / 'vertebrae_body.nii.gz'
            total_file = patient_dir / 'total.nii.gz'
            
            # Look for vertebrae_body file
            if vertebrae_body_file.exists():
                scan['vertebrae_seg_path'] = str(vertebrae_body_file)
            else:
                vertebrae_body_dir = patient_dir / 'vertebrae_body'
                # Check in vertebrae_body subdirectory
                found = list(vertebrae_body_dir.glob('*.nii*'))
                if not found:
                    # Check in patient directory directly
                    found = list(patient_dir.glob('*.nii*'))
                    # Filter out total.nii.gz if it exists
                    found = [f for f in found if f.name != 'total.nii.gz']
                
                if found:
                    scan['vertebrae_seg_path'] = str(found[0])
            
            # Look for total file
            if total_file.exists():
                scan['vertebrae_levels_path'] = str(total_file)
        
        found_body = sum(1 for s in scans if s.get('vertebrae_seg_path'))
        found_total = sum(1 for s in scans if s.get('vertebrae_levels_path'))
        print(f"  Found vertebrae_body: {found_body}/{len(scans)}")
        print(f"  Found total: {found_total}/{len(scans)}")
        if found_body == 0:
            print("  ⚠️  WARNING: No existing vertebrae_body segmentations found!")
        print()
        
def run_totalseg_parallel(scans, output_dir, num_workers=4):
    device_mode = 'GPU' if CONFIG.get('use_gpu', False) else 'CPU'
    
    print(f"\n[2/4] Running TotalSegmentator...")
    print(f"  Device: {device_mode}")
    print(f"  Estimated time: 2-5 min per scan\n")
    
    for scan in scans:
        scan['output_dir'] = output_dir
        scan['use_gpu'] = CONFIG.get('use_gpu', False)
        scan['force_gpu'] = CONFIG.get('force_gpu', False)
    
    results = []
    errors = []
    
    with tqdm(total=len(scans), desc="Segmenting scans", unit="scan") as pbar:
        for scan in scans:
            result = run_totalseg_single(scan)
            results.append(result)
            if result and 'error' in result:
                errors.append({'patient_id': scan['patient_id'], 'error': result['error']})
            pbar.update(1)
    
    for scan, result in zip(scans, results):
        if result and 'vertebrae_body' in result:
            scan['vertebrae_seg_path'] = result.get('vertebrae_body')
            scan['vertebrae_levels_path'] = result.get('vertebrae_levels')
    
    successful = sum(1 for r in results if r and 'vertebrae_body' in r and r['vertebrae_body'])
    print(f"\nTotalSegmentator: {successful}/{len(scans)} successful\n")
    
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:3]:
            print(f"  - {err['patient_id']}: {err['error'][:80]}")
    
    return scans

# ============================================================
# PARALLEL PROCESSING
# ============================================================

def process_single_patient(scan_info, output_dirs, use_plane_pruning):
    patient_id = scan_info['patient_id']
    ct_path = scan_info['ct_path']
    vertebrae_body_path = scan_info.get('vertebrae_seg_path')
    total_path = scan_info.get('vertebrae_levels_path')
    
    if not vertebrae_body_path or not os.path.exists(vertebrae_body_path):
        return {'patient_id': patient_id, 'success': False, 'error': 'No vertebrae_body segmentation'}
    
    if not total_path or not os.path.exists(total_path):
        return {'patient_id': patient_id, 'success': False, 'error': 'No total segmentation'}
    
    try:
        # Load CT image once - OPTIMIZED
        ct_img_orig = nib.load(ct_path)
        ct_img = reorient_to_ras(ct_img_orig)
        ct_data = np.asanyarray(ct_img.dataobj).astype(np.float32, copy=False)  # Load once!
        voxel_spacing = ct_img.header.get_zooms()
        
        # Split vertebrae_body into individual vertebrae using total labels
        individual_vertebrae, vb_img = split_vertebrae_body_by_total_labels(
            vertebrae_body_path, 
            total_path
        )
        
        if len(individual_vertebrae) == 0:
            return {'patient_id': patient_id, 'success': False, 'error': 'No vertebrae found after splitting'}
        
        vertebrae_list = sorted(individual_vertebrae.items())
        print(f"  {patient_id}: {len(vertebrae_list)} confident, 0 fallback")
        
        results = []
        all_vertebrae_masks = []
        
        # Process each individual vertebra
        for idx, (total_label, vert_data) in enumerate(vertebrae_list, start=1):
            try:
                anatomical_label = vert_data['anatomical_label']
                vertebra_mask_ras = vert_data['mask']
                confidence = vert_data['confidence']
                is_sacrum_flag = (total_label == 26)
                
                # Extract metrics - NOW PASSING CT_DATA directly
                metrics = extract_vertebra_metrics(
                    ct_data, vertebra_mask_ras, ct_img_orig, ct_img, voxel_spacing,
                    CONFIG['num_layers'], is_sacrum_flag, use_plane_pruning
                )
                
                if metrics is None or not metrics.get('success'):
                    continue
                
                metrics.update({
                    'patient_id': patient_id,
                    'vertebra_label': int(total_label),
                    'vertebra_level': anatomical_label,
                    'has_anatomical_label': True,
                    'is_sacrum': is_sacrum_flag,
                    'label_confidence': True,
                    'label_method': 'total_overlay',
                    'overlap_fraction': confidence
                })
                
                # Save segmentations
                if metrics.get('_masks_ras') and CONFIG['save_segmentations']:
                    if CONFIG.get('save_individual_vertebrae', True):
                        save_segmentation(
                            metrics['_masks_ras'], 
                            metrics['_ct_img'],
                            metrics['_ct_img_orig'],
                            output_dirs['segmentations'],
                            patient_id, 
                            anatomical_label
                        )
                    
                    if CONFIG.get('save_combined_patient', True):
                        all_vertebrae_masks.append({
                            'vertebra_num': idx,
                            'anatomical_label': anatomical_label,
                            'masks_ras': metrics['_masks_ras']
                        })
                    
                    del metrics['_masks_ras']
                    del metrics['_ct_img']
                    del metrics['_ct_img_orig']
                
                results.append(metrics)
                
            except Exception as e:
                error_info = {
                    'patient_id': patient_id,
                    'vertebra_label': int(total_label),
                    'vertebra_level': vert_data['anatomical_label'],
                    'success': False,
                    'error': str(e)[:200]
                }
                results.append(error_info)
        
        # Save combined patient segmentation
        if CONFIG.get('save_combined_patient', True) and all_vertebrae_masks:
            save_combined_patient_segmentation(
                all_vertebrae_masks,
                ct_img,
                ct_img_orig,
                output_dirs['segmentations'],
                patient_id
            )
        
        return results
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        return {'patient_id': patient_id, 'success': False, 'error': error_details}

def process_all_patients_parallel(scans, output_dirs, num_workers, use_plane_pruning):
    """
    Process all patients - with Windows multiprocessing fix.
    On Windows, multiprocessing with CUDA doesn't work well, so we force sequential processing.
    """
    print(f"[3/4] Analyzing vertebrae...")
    print(f"  Workers: {num_workers}\n")
    
    scans_with_seg = [s for s in scans if s.get('vertebrae_seg_path')]
    if len(scans_with_seg) == 0:
        print("⚠️  WARNING: No scans have segmentation paths!")
        return []
    
    print(f"  Processing {len(scans_with_seg)}/{len(scans)} scans\n")
    
    all_results = []
    failed_patients = []
    
    # Check if we're on Windows or if num_workers is 1
    import platform
    is_windows = platform.system() == 'Windows'
    
    # Force sequential processing on Windows or if num_workers=1
    if is_windows or num_workers == 1:
        print("  Using sequential processing (Windows or single worker mode)")
        with tqdm(total=len(scans_with_seg), desc="Analyzing patients", unit="patient") as pbar:
            for scan in scans_with_seg:
                result = process_single_patient(scan, output_dirs, use_plane_pruning)
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, dict) and not result.get('success', False):
                    failed_patients.append(result)
                pbar.update(1)
    else:
        # Unix/Linux parallel processing
        process_func = partial(process_single_patient, 
                              output_dirs=output_dirs,
                              use_plane_pruning=use_plane_pruning)
        
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(scans_with_seg), desc="Analyzing patients", unit="patient") as pbar:
                for result in pool.imap_unordered(process_func, scans_with_seg, chunksize=1):
                    if isinstance(result, list):
                        all_results.extend(result)
                    elif isinstance(result, dict) and not result.get('success', False):
                        failed_patients.append(result)
                    pbar.update(1)
    
    print(f"\nAnalysis: {len(all_results)} vertebrae processed")
    if failed_patients:
        print(f"Failed: {len(failed_patients)} patients")
    print()
    
    return all_results

# ============================================================
# INTERACTIVE INPUT
# ============================================================

def get_input_directory():
    """Prompt user for input directory"""
    while True:
        print("\n" + "=" * 60)
        print("INPUT DIRECTORY")
        print("=" * 60)
        input_path = input("Enter the path to your CT scans directory: ").strip()
        input_path = input_path.strip('"').strip("'")
        input_dir = Path(input_path)
        
        if not input_dir.exists():
            print(f"\n❌ ERROR: Directory does not exist: {input_dir}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                sys.exit(1)
            continue
        
        if not input_dir.is_dir():
            print(f"\n❌ ERROR: Path is not a directory: {input_dir}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                sys.exit(1)
            continue
        
        nii_files = list(input_dir.glob('*.nii.gz'))
        if len(nii_files) == 0:
            print(f"\n⚠️  WARNING: No .nii.gz files found in {input_dir}")
            proceed = input("Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                retry = input("Try different directory? (y/n): ").strip().lower()
                if retry != 'y':
                    sys.exit(1)
                continue
        else:
            print(f"\n✓ Found {len(nii_files)} .nii.gz files")
        
        return str(input_dir)

def get_output_directory():
    """Prompt user for output directory"""
    print("\n" + "=" * 60)
    print("OUTPUT DIRECTORY")
    print("=" * 60)
    print("Options:")
    print("  1. Press ENTER to auto-generate (timestamp-based)")
    print("  2. Enter a custom path")
    
    output_path = input("\nEnter output directory path (or press ENTER for auto): ").strip()
    
    if not output_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path.cwd() / f'bone_analysis_results_{timestamp}'
        print(f"\n✓ Output will be saved to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)
    
    output_path = output_path.strip('"').strip("'")
    output_dir = Path(output_path)
    
    if output_dir.exists():
        print(f"\n⚠️  WARNING: Directory already exists: {output_dir}")
        overwrite = input("Use this directory anyway? (y/n): ").strip().lower()
        if overwrite != 'y':
            return get_output_directory()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Output will be saved to: {output_dir}")
    return str(output_dir)

def get_configuration_options():
    """Prompt user for additional configuration"""
    print("\n" + "=" * 60)
    print("CONFIGURATION OPTIONS")
    print("=" * 60)
    
    print("\n1. Processing device:")
    print("   a. GPU (faster, recommended)")
    print("   b. CPU (slower, more compatible)")
    device_choice = input("Choose device (a/b) [default: a]: ").strip().lower()
    use_gpu = device_choice != 'b'
    
    print("\n2. TotalSegmentator:")
    skip_totalseg = input("Skip TotalSegmentator? (y/n) [default: n]: ").strip().lower() == 'y'
    
    print("\n3. Number of workers:")
    totalseg_workers_input = input("TotalSegmentator workers [default: 4]: ").strip()
    totalseg_workers = int(totalseg_workers_input) if totalseg_workers_input else 4
    
    analysis_workers_input = input("Analysis workers [default: 16]: ").strip()
    analysis_workers = int(analysis_workers_input) if analysis_workers_input else 16
    
    verbose = input("\n4. Enable verbose output? (y/n) [default: n]: ").strip().lower() == 'y'
    
    return {
        'use_gpu': use_gpu,
        'skip_totalsegmentator': skip_totalseg,
        'totalseg_workers': totalseg_workers,
        'analysis_workers': analysis_workers,
        'verbose_workers': verbose
    }

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BONE ANALYSIS PIPELINE V10")
    print("Robust Vertebrae Segmentation and Analysis")
    print("=" * 60)
    
    input_dir = get_input_directory()
    output_dir = get_output_directory()
    
    print("\n" + "=" * 60)
    config_choice = input("Configure advanced settings? (y/n) [default: n]: ").strip().lower()
    
    if config_choice == 'y':
        config_options = get_configuration_options()
    else:
        config_options = {
            'use_gpu': True,
            'skip_totalsegmentator': False,
            'totalseg_workers': 4,
            'analysis_workers': 16,
            'verbose_workers': False
        }
        print("\n✓ Using default configuration")
    
    CONFIG.update({
        'input_dir': input_dir,
        'output_dir': output_dir,
        **config_options
    })
    
    print("=" * 60)
    print("STARTING ANALYSIS")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {CONFIG['input_dir']}")
    print(f"Output: {CONFIG['output_dir']}")
    
    output_dirs = setup_output_structure(CONFIG['output_dir'])
    
    print("\nOutput structure:")
    print(f"  Results: {output_dirs['results']}")
    print(f"  Segmentations: {output_dirs['segmentations']}")
    print(f"  TotalSegmentator: {output_dirs['vertebrae_seg']}")
    
    print("\n[1/4] Finding CT scans...")
    scans = discover_scans(CONFIG['input_dir'])
    
    if len(scans) == 0:
        print("ERROR: No scans found!")
        print(f"Looking for files matching pattern: input.*.nii.gz")
        sys.exit(1)
    
    print(f"  Found: {len(scans)} scans")
    
    if not CONFIG.get('skip_totalsegmentator', False):
        scans = run_totalseg_parallel(scans, output_dirs['vertebrae_seg'], CONFIG['totalseg_workers'])
    else:
        print("\n[2/4] Skipping TotalSegmentator (using existing segmentations)...")
        
        for scan in scans:
            patient_id = scan['patient_id']
            patient_dir = Path(output_dirs['vertebrae_seg']) / patient_id
            
            # Look for vertebrae_body file (flexible - checks multiple locations/names)
            vertebrae_body_file = None
            possible_vb_files = [
                patient_dir / 'vertebrae_body.nii',
                patient_dir / 'vertebrae_body.nii.gz',
                patient_dir / 'vertebrae_body' / 'vertebrae_body.nii',
                patient_dir / 'vertebrae_body' / 'vertebrae_body.nii.gz',
            ]
            
            for vb_file in possible_vb_files:
                if vb_file.exists():
                    scan['vertebrae_seg_path'] = str(vb_file)
                    break
    
            # Look for total file (flexible)
            for total_name in ['total.nii.gz', 'total.nii']:
                total_file = patient_dir / total_name
                if total_file.exists():
                    scan['vertebrae_levels_path'] = str(total_file)
                    break
        
        found_body = sum(1 for s in scans if s.get('vertebrae_seg_path'))
        found_total = sum(1 for s in scans if s.get('vertebrae_levels_path'))
        print(f"  Found vertebrae_body: {found_body}/{len(scans)}")
        print(f"  Found total: {found_total}/{len(scans)}")
        if found_body == 0:
            print("  ⚠️  WARNING: No existing vertebrae_body segmentations found!")
        print()
    
    all_results = process_all_patients_parallel(
        scans, output_dirs,
        CONFIG['analysis_workers'],
        CONFIG['use_plane_pruning']
    )
    
    print("[4/4] Saving results...")
    
    successful_results = [r for r in all_results if r.get('success', False)]
    failed_results = [r for r in all_results if not r.get('success', False)]
    
    print(f"\n  Successful: {len(successful_results)}")
    print(f"  Failed: {len(failed_results)}")
    
    if len(failed_results) > 0 and len(failed_results) <= 10:
        print("\n  Failed vertebrae:")
        for fail in failed_results[:10]:
            patient = fail.get('patient_id', 'unknown')
            vertebra = fail.get('vertebra_level', fail.get('vertebra_label', 'unknown'))
            error = fail.get('error', 'no error')
            print(f"    - {patient}/{vertebra}: {error[:80]}")
    
    df = pd.DataFrame(successful_results)
    
    if len(df) > 0:
        output_csv = output_dirs['results'] / 'all_results_v10.csv'
        df.to_csv(output_csv, index=False)
        
        total_vertebrae = len(df)
        confident_labels = df['label_confidence'].sum() if 'label_confidence' in df.columns else 0
        
        summary = {
            'total_scans': len(scans),
            'total_vertebrae': total_vertebrae,
            'confident_labels': int(confident_labels),
            'fallback_labels': total_vertebrae - int(confident_labels),
            'pipeline_version': '10.0',
            'mean_cortical_thickness_mm': float(df.cortical_thickness_mm.mean()),
            'std_cortical_thickness_mm': float(df.cortical_thickness_mm.std()),
            'mean_cortical_hu': float(df.cortical_mean_hu.mean()),
            'mean_trabecular_hu': float(df.trabecular_mean_hu.mean()),
            'mean_superior_thickness_mm': float(df.superior_thickness_mm.mean()),
            'std_superior_thickness_mm': float(df.superior_thickness_mm.std()),
            'mean_inferior_thickness_mm': float(df.inferior_thickness_mm.mean()),
            'std_inferior_thickness_mm': float(df.inferior_thickness_mm.std()),
            'mean_superior_hu': float(df.superior_mean_hu.mean()),
            'mean_inferior_hu': float(df.inferior_mean_hu.mean()),
            'sacrum_count': int(df['is_sacrum'].sum() if 'is_sacrum' in df.columns else 0)
        }
        
        with open(output_dirs['results'] / 'summary_v10.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 60)
        print("COMPLETE!")
        print("=" * 60)
        print(f"Scans: {len(scans)}")
        print(f"Vertebrae: {summary['total_vertebrae']}")
        
        if 'label_confidence' in df.columns:
            print(f"Labeling: {confident_labels} confident, {summary['fallback_labels']} fallback")
        
        if summary['sacrum_count'] > 0:
            print(f"Sacrum: {summary['sacrum_count']}")
        
        print(f"\nCortical:")
        print(f"  Thickness: {summary['mean_cortical_thickness_mm']:.2f}±{summary['std_cortical_thickness_mm']:.2f} mm")
        print(f"  HU: {summary['mean_cortical_hu']:.0f}")
        
        print(f"\nTrabecular:")
        print(f"  HU: {summary['mean_trabecular_hu']:.0f}")
        
        print(f"\nEndplates:")
        print(f"  Superior: {summary['mean_superior_thickness_mm']:.2f}±{summary['std_superior_thickness_mm']:.2f} mm, HU: {summary['mean_superior_hu']:.0f}")
        print(f"  Inferior: {summary['mean_inferior_thickness_mm']:.2f}±{summary['std_inferior_thickness_mm']:.2f} mm, HU: {summary['mean_inferior_hu']:.0f}")
        
        print(f"\nOutputs:")
        print(f"  CSV: {output_csv}")
        print(f"  Segmentations: {output_dirs['segmentations']}")
        print(f"  Labels: 0=bg, 1=cortical, 2=trabecular, 3=superior, 4=inferior")
    else:
        print("\n" + "=" * 60)
        print("ERROR: No successful results to save!")
        print("=" * 60)
        
        print("\nDiagnostic Information:")
        print(f"  Scans found: {len(scans)}")
        
        scans_with_seg = sum(1 for s in scans if s.get('vertebrae_seg_path'))
        print(f"  Scans with segmentation: {scans_with_seg}/{len(scans)}")
        
        if len(all_results) > 0:
            print(f"  Total processing attempts: {len(all_results)}")
            print(f"\n  Common errors:")
            error_counts = {}
            for r in all_results:
                if not r.get('success', False):
                    error_msg = r.get('error', 'Unknown error')
                    error_key = error_msg[:50]
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    - ({count}x) {error}...")
        else:
            print("  No processing attempts were made")
        
        print("\n  Troubleshooting:")
        print("    1. Check TotalSegmentator output files exist")
        print("    2. Verify file naming: input.*.nii.gz")
        print("    3. Try running with verbose mode")
        
        sys.exit(1)
    
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")