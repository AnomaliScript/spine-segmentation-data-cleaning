#!/usr/bin/env python3
"""
Create Path Planning Sanctuary (/pthpl) with Optimal Spacings
Selective dimension-wise resampling: only fix dimensions > 1mm

Strategy:
  - X, Y, Z analyzed independently
  - If dimension > 1mm → resample to 1mm
  - If dimension ≤ 1mm → keep original (preserve detail!)
  
Examples:
  3.0×3.0×0.5mm → 1.0×1.0×0.5mm  (fix X,Y; keep Z)
  1.0×1.0×3.0mm → 1.0×1.0×1.0mm  (keep X,Y; fix Z)
  0.5×2.0×0.5mm → 0.5×1.0×0.5mm  (keep X,Z; fix Y)
  0.7×0.7×0.8mm → 0.7×0.7×0.8mm  (already good, just copy!)

Result: "Sanctuary of good spacings" - all dimensions ≤ 1mm
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil


# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths (MODIFY THESE)
DATASETS = [
    {
        'name': 'CTSpine1K',
        'base_path': Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\CTSpine1K"),
        'volumes_dir': 'clean_volumes',
        'labels_dir': 'clean_labels'
    },
    {
        'name': 'RSNA_clean_v3',
        'base_path': Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\RSNA_clean_v3"),
        'volumes_dir': 'clean_volumes',
        'labels_dir': 'clean_labels'
    }
]

# Spacing threshold (any dimension > this gets resampled to this value)
SPACING_THRESHOLD_MM = 1.0

# Interpolation orders
INTERPOLATION_ORDER_VOLUME = 3  # Cubic (sinc-like) for CT volumes
INTERPOLATION_ORDER_LABEL = 0   # Nearest neighbor for segmentation labels


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_spacing_from_affine(affine):
    """Extract voxel spacing from NIfTI affine matrix."""
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return spacing


def calculate_selective_zoom_factors(original_spacing, threshold=1.0):
    """
    Calculate zoom factors for selective dimension-wise resampling.
    Only resample dimensions that exceed threshold.
    
    Args:
        original_spacing: (x, y, z) spacing in mm
        threshold: Maximum allowed spacing (mm)
        
    Returns:
        zoom_factors: (x, y, z) zoom factors
        target_spacing: (x, y, z) target spacing after resampling
        needs_resampling: Boolean indicating if any resampling is needed
    """
    zoom_factors = []
    target_spacing = []
    
    for dim_spacing in original_spacing:
        if dim_spacing > threshold:
            # This dimension exceeds threshold - resample to threshold
            zoom_factor = dim_spacing / threshold
            target_spacing.append(threshold)
        else:
            # This dimension is already good - keep original
            zoom_factor = 1.0
            target_spacing.append(dim_spacing)
        
        zoom_factors.append(zoom_factor)
    
    zoom_factors = np.array(zoom_factors)
    target_spacing = np.array(target_spacing)
    needs_resampling = not np.allclose(zoom_factors, 1.0, atol=0.01)
    
    return zoom_factors, target_spacing, needs_resampling


def selective_resample(img_data, zoom_factors, order=3):
    """
    Perform selective resampling with specified interpolation order.
    
    Args:
        img_data: 3D numpy array
        zoom_factors: (x, y, z) zoom factors
        order: Interpolation order (0=nearest, 3=cubic)
        
    Returns:
        resampled_data: Resampled 3D array
    """
    if np.allclose(zoom_factors, 1.0, atol=0.01):
        # No resampling needed
        return img_data
    
    # Perform resampling
    resampled_data = zoom(img_data, zoom_factors, order=order, mode='nearest')
    
    return resampled_data


def create_new_affine(original_affine, original_spacing, target_spacing):
    """
    Create new affine matrix with updated spacing.
    Preserves orientation but updates voxel dimensions.
    """
    new_affine = original_affine.copy()
    
    # Scale rotation matrix columns by new spacing
    for i in range(3):
        direction = original_affine[:3, i] / original_spacing[i]
        new_affine[:3, i] = direction * target_spacing[i]
    
    return new_affine


def process_file_to_sanctuary(input_path, output_path, threshold=1.0, 
                              interpolation_order=3, file_type="volume"):
    """
    Process a single file with selective resampling to sanctuary folder.
    
    Args:
        input_path: Path to input .nii.gz file
        output_path: Path to output .nii.gz file in /pthpl
        threshold: Spacing threshold (mm)
        interpolation_order: 0=nearest, 3=cubic
        file_type: "volume" or "label"
        
    Returns:
        stats: Dictionary with processing statistics
    """
    # Load NIfTI file
    img = nib.load(input_path)
    img_data = img.get_fdata()
    original_affine = img.affine
    
    # Get original spacing
    original_spacing = get_spacing_from_affine(original_affine)
    
    # Calculate selective zoom factors
    zoom_factors, target_spacing, needs_resampling = calculate_selective_zoom_factors(
        original_spacing, threshold
    )
    
    stats = {
        'filename': input_path.name,
        'file_type': file_type,
        'original_spacing': original_spacing,
        'target_spacing': target_spacing,
        'original_shape': img_data.shape,
        'needs_resampling': needs_resampling,
        'dimensions_fixed': []
    }
    
    # Determine which dimensions were fixed
    for i, (orig, targ) in enumerate(zip(original_spacing, target_spacing)):
        if not np.isclose(orig, targ, atol=0.01):
            dim_name = ['X', 'Y', 'Z'][i]
            stats['dimensions_fixed'].append(f"{dim_name}: {orig:.2f}→{targ:.2f}mm")
    
    if not needs_resampling:
        # Already meets criteria - just copy
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        stats['action'] = 'COPIED (already good)'
        stats['new_shape'] = img_data.shape
        return stats
    
    # Perform selective resampling
    resampled_data = selective_resample(img_data, zoom_factors, order=interpolation_order)
    
    # Create new affine
    new_affine = create_new_affine(original_affine, original_spacing, target_spacing)
    
    # Create new NIfTI image
    new_img = nib.Nifti1Image(resampled_data.astype(img_data.dtype), new_affine)
    new_img.header['descrip'] = f"Selective resample: dims>{threshold}mm to {threshold}mm"
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_img, output_path)
    
    stats['action'] = 'RESAMPLED'
    stats['new_shape'] = resampled_data.shape
    
    return stats


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def create_sanctuary_for_dataset(dataset_config, dry_run=True):
    """
    Create /pthpl sanctuary folder for a single dataset.
    
    Args:
        dataset_config: Dictionary with dataset paths
        dry_run: If True, analyze without creating files
    """
    name = dataset_config['name']
    base_path = dataset_config['base_path']
    volumes_dir = base_path / dataset_config['volumes_dir']
    labels_dir = base_path / dataset_config['labels_dir']
    
    # Output directories
    sanctuary_path = base_path / "pthpl"
    sanctuary_volumes = sanctuary_path / "volumes"
    sanctuary_labels = sanctuary_path / "labels"
    
    print("\n" + "="*80)
    print(f"PROCESSING DATASET: {name}")
    print("="*80)
    print(f"Source volumes:  {volumes_dir}")
    print(f"Source labels:   {labels_dir}")
    print(f"Sanctuary path:  {sanctuary_path}")
    print(f"Threshold:       ≤{SPACING_THRESHOLD_MM}mm per dimension")
    print(f"Mode:            {'DRY RUN (preview)' if dry_run else 'LIVE RUN (creating files)'}")
    print("="*80 + "\n")
    
    # Find all volume files
    volume_files = sorted(volumes_dir.glob("*.nii.gz"))
    
    if not volume_files:
        print(f"⚠️  No volume files found in {volumes_dir}")
        return None
    
    print(f"Found {len(volume_files)} volume files\n")
    
    if dry_run:
        print("DRY RUN - Analyzing first 5 files...\n")
        
        for vol_file in volume_files[:5]:
            img = nib.load(vol_file)
            spacing = get_spacing_from_affine(img.affine)
            zoom_factors, target_spacing, needs_resampling = calculate_selective_zoom_factors(
                spacing, SPACING_THRESHOLD_MM
            )
            
            print(f"📁 {vol_file.name}")
            print(f"   Original: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm")
            print(f"   Target:   {target_spacing[0]:.2f} × {target_spacing[1]:.2f} × {target_spacing[2]:.2f} mm")
            
            if needs_resampling:
                dims_fixed = []
                for i, (orig, targ) in enumerate(zip(spacing, target_spacing)):
                    if not np.isclose(orig, targ):
                        dim_name = ['X', 'Y', 'Z'][i]
                        dims_fixed.append(f"{dim_name}: {orig:.2f}→{targ:.2f}")
                print(f"   Action:   RESAMPLE ({', '.join(dims_fixed)})")
            else:
                print(f"   Action:   COPY (already ≤{SPACING_THRESHOLD_MM}mm)")
            
            print()
        
        if len(volume_files) > 5:
            print(f"... and {len(volume_files) - 5} more files\n")
        
        # Statistics
        needs_work = 0
        already_good = 0
        for vol_file in volume_files:
            img = nib.load(vol_file)
            spacing = get_spacing_from_affine(img.affine)
            _, _, needs_resampling = calculate_selective_zoom_factors(spacing, SPACING_THRESHOLD_MM)
            if needs_resampling:
                needs_work += 1
            else:
                already_good += 1
        
        print("-"*80)
        print(f"SUMMARY for {name}:")
        print(f"  Files already ≤{SPACING_THRESHOLD_MM}mm: {already_good} ({already_good/len(volume_files)*100:.1f}%)")
        print(f"  Files needing work: {needs_work} ({needs_work/len(volume_files)*100:.1f}%)")
        print("-"*80)
        
        return None
    
    # LIVE RUN - Actually create sanctuary
    print("LIVE RUN - Creating sanctuary files...\n")
    
    all_stats = []
    
    for vol_file in tqdm(volume_files, desc=f"Processing {name}"):
        # Get case identifier from filename
        # Handle both formats: CTS1K_001_0000.nii.gz or VerSe_001.nii.gz
        stem = vol_file.stem.replace('.nii', '')  # Remove .nii from .nii.gz
        
        # Try to find corresponding label file
        # Look for matching pattern in labels directory
        possible_label_names = [
            labels_dir / vol_file.name,  # Same name
            labels_dir / f"{stem}.nii.gz",  # Without _0000
            labels_dir / stem.replace('_0000', '.nii.gz'),  # Replace _0000
        ]
        
        label_file = None
        for possible_name in possible_label_names:
            if possible_name.exists():
                label_file = possible_name
                break
        
        # Output paths
        output_vol = sanctuary_volumes / vol_file.name
        if label_file:
            output_label = sanctuary_labels / label_file.name
        
        # Process volume
        vol_stats = process_file_to_sanctuary(
            vol_file,
            output_vol,
            threshold=SPACING_THRESHOLD_MM,
            interpolation_order=INTERPOLATION_ORDER_VOLUME,
            file_type="volume"
        )
        
        # Process label if exists
        if label_file:
            label_stats = process_file_to_sanctuary(
                label_file,
                output_label,
                threshold=SPACING_THRESHOLD_MM,
                interpolation_order=INTERPOLATION_ORDER_LABEL,
                file_type="label"
            )
        else:
            label_stats = None
        
        all_stats.append({
            'volume': vol_stats,
            'label': label_stats
        })
    
    # Summary report
    print("\n" + "="*80)
    print(f"SANCTUARY CREATION COMPLETE: {name}")
    print("="*80)
    
    copied = sum(1 for s in all_stats if s['volume']['action'] == 'COPIED (already good)')
    resampled = sum(1 for s in all_stats if s['volume']['action'] == 'RESAMPLED')
    
    print(f"Location: {sanctuary_path}")
    print(f"Total files: {len(all_stats)}")
    print(f"  Copied (already good): {copied}")
    print(f"  Resampled: {resampled}")
    print()
    
    # Dimension-specific statistics
    x_fixed = sum(1 for s in all_stats if any('X:' in d for d in s['volume']['dimensions_fixed']))
    y_fixed = sum(1 for s in all_stats if any('Y:' in d for d in s['volume']['dimensions_fixed']))
    z_fixed = sum(1 for s in all_stats if any('Z:' in d for d in s['volume']['dimensions_fixed']))
    
    print("Dimensions fixed:")
    print(f"  X-dimension: {x_fixed} files")
    print(f"  Y-dimension: {y_fixed} files")
    print(f"  Z-dimension: {z_fixed} files")
    print()
    
    # Save detailed statistics
    stats_file = sanctuary_path / "sanctuary_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write(f"PATH PLANNING SANCTUARY STATISTICS: {name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Threshold: <={SPACING_THRESHOLD_MM}mm per dimension\n")
        f.write(f"Strategy: Selective dimension-wise resampling\n")
        f.write(f"Volume interpolation: Order {INTERPOLATION_ORDER_VOLUME} (cubic/sinc-like)\n")
        f.write(f"Label interpolation: Order {INTERPOLATION_ORDER_LABEL} (nearest neighbor)\n\n")
        f.write(f"Total files: {len(all_stats)}\n")
        f.write(f"Copied (already good): {copied}\n")
        f.write(f"Resampled: {resampled}\n\n")
        f.write(f"Dimensions fixed:\n")
        f.write(f"  X: {x_fixed} files\n")
        f.write(f"  Y: {y_fixed} files\n")
        f.write(f"  Z: {z_fixed} files\n\n")
        f.write("="*80 + "\n")
        f.write("DETAILED FILE STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        for stat in all_stats:
            vol = stat['volume']
            f.write(f"File: {vol['filename']}\n")
            f.write(f"  Original spacing: {vol['original_spacing'][0]:.2f} × {vol['original_spacing'][1]:.2f} × {vol['original_spacing'][2]:.2f} mm\n")
            f.write(f"  Target spacing:   {vol['target_spacing'][0]:.2f} × {vol['target_spacing'][1]:.2f} × {vol['target_spacing'][2]:.2f} mm\n")
            f.write(f"  Action: {vol['action']}\n")
            if vol['dimensions_fixed']:
                f.write(f"  Fixed: {', '.join(vol['dimensions_fixed'])}\n")
            f.write(f"  Shape: {vol['original_shape']} -> {vol['new_shape']}\n\n")
    
    print(f"Statistics saved: {stats_file}")
    print("="*80 + "\n")
    
    return all_stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create /pthpl sanctuary with optimal spacings (≤1mm per dimension)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_sanctuary.py --dry-run    # Preview what will be created
  python create_sanctuary.py              # Create sanctuary folders
  
Strategy:
  - Analyze each dimension (X, Y, Z) independently
  - If dimension > 1mm → resample to 1mm (using sinc-like interpolation)
  - If dimension <= 1mm → keep original (preserve detail)
  
Result:
  All files in /pthpl have spacing <=1mm in ALL dimensions
  Original high-quality detail is preserved where it exists
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview analysis without creating files'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PATH PLANNING SANCTUARY CREATOR")
    print("="*80)
    print(f"Threshold: ≤{SPACING_THRESHOLD_MM}mm per dimension")
    print(f"Strategy: Selective dimension-wise resampling")
    print(f"Mode: {'DRY RUN (preview)' if args.dry_run else 'LIVE RUN (creating files)'}")
    print("="*80)
    
    # Process each dataset
    for dataset_config in DATASETS:
        if not dataset_config['base_path'].exists():
            print(f"\n⚠️  Dataset not found: {dataset_config['base_path']}")
            print("   Update DATASETS configuration in script")
            continue
        
        create_sanctuary_for_dataset(dataset_config, dry_run=args.dry_run)
    
    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN COMPLETE")
        print("="*80)
        print("To create sanctuary folders, run without --dry-run flag")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("ALL SANCTUARIES CREATED!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Use files in /pthpl for path planning algorithm development")
        print("  2. All files guaranteed to have spacing ≤1mm in every dimension")
        print("  3. Original high-resolution detail preserved where it existed")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()