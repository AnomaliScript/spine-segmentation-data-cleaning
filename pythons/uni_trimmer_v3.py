import nibabel as nib
import numpy as np
from pathlib import Path
import shutil
import sys

def remove_labels_above_threshold(nii_in_path, nii_out_path, threshold=7):
    """
    Remove all labels above threshold from segmentation file.
    Preserves original data type for segmentation integrity.
    """
    img = nib.load(nii_in_path)
    data = np.asarray(img.dataobj)  # Preserves original dtype
    
    # Zero out labels above threshold
    data[data > threshold] = 0
    
    # Save modified data as new NIfTI
    new_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
    nib.save(new_img, nii_out_path)
    print(f"  ✓ Cleaned segmentation: {nii_out_path.name}")


def find_matching_pairs(volumes_path: Path, labels_path: Path) -> list[tuple]:
    """
    Find matching volume/label pairs from two directories.
    Handles various naming conventions (with/without _seg suffix, etc.)
    Returns list of (volume_file, label_file) tuples.
    """
    # Get all volume and label files
    vol_files = sorted(list(volumes_path.glob("*.nii.gz")) + list(volumes_path.glob("*.nii")))
    label_files = sorted(list(labels_path.glob("*.nii.gz")) + list(labels_path.glob("*.nii")))
    
    if not vol_files:
        print(f"ERROR: No volume files found in {volumes_path}")
        return []
    
    if not label_files:
        print(f"ERROR: No label files found in {labels_path}")
        return []
    
    # Create mappings for easy lookup
    vol_dict = {}
    for vf in vol_files:
        # Get base name without extension
        base = vf.stem.replace('.nii', '')
        # Also try without common suffixes
        clean_base = base.replace('_0000', '').replace('_volume', '').replace('_vol', '').replace('_image', '')
        vol_dict[base] = vf
        vol_dict[clean_base] = vf
    
    label_dict = {}
    for lf in label_files:
        base = lf.stem.replace('.nii', '')
        clean_base = base.replace('_seg', '').replace('_label', '').replace('_mask', '').replace('_segmentation', '')
        label_dict[base] = lf
        label_dict[clean_base] = lf
    
    # Match pairs
    pairs = []
    matched_labels = set()
    
    for label_file in label_files:
        if label_file in matched_labels:
            continue
            
        label_base = label_file.stem.replace('.nii', '')
        label_clean = label_base.replace('_seg', '').replace('_label', '').replace('_mask', '').replace('_segmentation', '')
        
        # Try to find matching volume
        vol_file = None
        
        # Strategy 1: Direct match
        if label_base in vol_dict:
            vol_file = vol_dict[label_base]
        # Strategy 2: Clean name match
        elif label_clean in vol_dict:
            vol_file = vol_dict[label_clean]
        # Strategy 3: Label name + _0000
        elif f"{label_base}_0000" in vol_dict:
            vol_file = vol_dict[f"{label_base}_0000"]
        # Strategy 4: Clean name + _0000
        elif f"{label_clean}_0000" in vol_dict:
            vol_file = vol_dict[f"{label_clean}_0000"]
        # Strategy 5: Volume name + _seg
        else:
            for vol_base, vf in vol_dict.items():
                vol_clean = vol_base.replace('_0000', '').replace('_volume', '').replace('_vol', '')
                if vol_clean == label_clean or f"{vol_clean}_seg" == label_base:
                    vol_file = vf
                    break
        
        if vol_file:
            pairs.append((vol_file, label_file))
            matched_labels.add(label_file)
        else:
            print(f"  ⚠ WARNING: No matching volume found for: {label_file.name}")
    
    return pairs


def process_cervical_dataset(base_path, dataset_prefix="DATASET", threshold=7):
    """
    Universal processor for any spine dataset with /volumes and /labels folders.
    
    Args:
        base_path: Path to dataset folder containing /volumes and /labels
        dataset_prefix: Prefix for renamed files (e.g., "CTS1K", "VERSE", "RSNA")
        threshold: Maximum label to keep (default 7 for cervical vertebrae C1-C7)
    """
    base_path = Path(base_path)
    volumes_path = base_path / "volumes"
    labels_path = base_path / "labels"
    
    # Validate input directories
    if not base_path.exists():
        print(f"ERROR: Base path does not exist: {base_path}")
        return
    
    if not volumes_path.exists():
        print(f"ERROR: Volumes directory not found: {volumes_path}")
        return
    
    if not labels_path.exists():
        print(f"ERROR: Labels directory not found: {labels_path}")
        return
    
    # Create output directories
    renamed_vol_path = base_path / "renamed_vol"
    cervical_labels_path = base_path / "labels_cervical_only"
    
    renamed_vol_path.mkdir(exist_ok=True)
    cervical_labels_path.mkdir(exist_ok=True)
    
    print("="*80)
    print(f"PROCESSING DATASET: {dataset_prefix}")
    print("="*80)
    print(f"Base directory: {base_path}")
    print(f"Source folders:")
    print(f"  - Volumes: {volumes_path}")
    print(f"  - Labels:  {labels_path}")
    print(f"Output folders:")
    print(f"  - Volumes: {renamed_vol_path}")
    print(f"  - Labels:  {cervical_labels_path}")
    print(f"Label threshold: keeping labels 1-{threshold} (cervical vertebrae)")
    print()
    
    # Find matching pairs
    print("Finding matching volume/label pairs...")
    pairs = find_matching_pairs(volumes_path, labels_path)
    
    if not pairs:
        print("ERROR: No matching pairs found!")
        print("\nPlease check:")
        print("  1. Files exist in both volumes/ and labels/ directories")
        print("  2. Volume and label files have matching names")
        return
    
    print(f"Found {len(pairs)} matching volume/label pairs")
    print()
    
    # Process each pair
    processed_count = 0
    skipped_count = 0
    
    for idx, (vol_file, label_file) in enumerate(pairs, start=1):
        # Generate output names
        output_id = f"{idx:03d}"
        output_vol_name = f"{dataset_prefix}_{output_id}_0000.nii.gz"
        output_seg_name = f"{dataset_prefix}_{output_id}.nii.gz"
        
        out_vol_path = renamed_vol_path / output_vol_name
        out_seg_path = cervical_labels_path / output_seg_name
        
        print(f"Processing pair {idx:03d}:")
        print(f"  Volume: {vol_file.name} → {output_vol_name}")
        print(f"  Label:  {label_file.name} → {output_seg_name}")
        
        try:
            # Copy and rename volume (no modification needed)
            shutil.copy2(vol_file, out_vol_path)
            print(f"  ✓ Copied volume: {output_vol_name}")
            
            # Process and save cleaned segmentation (remove non-cervical labels)
            remove_labels_above_threshold(label_file, out_seg_path, threshold=threshold)
            
            processed_count += 1
            print()
            
        except Exception as e:
            print(f"  ✗ ERROR processing pair: {e}")
            skipped_count += 1
            print()
    
    # Summary
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {processed_count} pairs")
    print(f"Skipped: {skipped_count} pairs")
    print()
    print(f"Output locations:")
    print(f"  Volumes:       {renamed_vol_path.absolute()}")
    print(f"  Segmentations: {cervical_labels_path.absolute()}")
    print()


def main():
    """
    Main function - edit these variables to process different datasets.
    """
    # ==================== CONFIGURATION ====================
    # Edit these variables for your specific dataset:
    
    # Path to dataset folder (must contain /volumes and /labels subdirectories)
    base_path = r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\VerSe_clean_v3"

    # Prefix for output files (e.g., "CTS1K", "VERSE", "RSNA", etc.)
    dataset_prefix = "VerSe"
    
    # Label threshold (keep labels 1-7 for cervical vertebrae C1-C7)
    cervical_threshold = 7
    
    # ==================== PROCESSING ====================
    process_cervical_dataset(
        base_path=base_path,
        dataset_prefix=dataset_prefix,
        threshold=cervical_threshold
    )


if __name__ == "__main__":
    # Check if command line arguments provided
    if len(sys.argv) >= 3:
        # Usage: python script.py <base_path> <prefix> [threshold]
        base_path = sys.argv[1]
        dataset_prefix = sys.argv[2]
        threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 7
        
        process_cervical_dataset(base_path, dataset_prefix, threshold)
    else:
        # Use configuration from main()
        main()