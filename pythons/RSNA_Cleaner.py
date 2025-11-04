from pathlib import Path
import shutil
import nibabel as nib
import numpy as np


def check_cervical_to_t3_labels(seg_path: Path) -> dict:
    """
    Check if segmentation contains ONLY labels 1-10 (C1-C7, T1-T3) and background (0).
    
    Returns {'is_valid': bool, 'labels_found': list, 'reason': str}
    """
    try:
        img = nib.load(seg_path)
        data = img.get_fdata()
        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
        
        if len(unique_labels) == 0:
            return {'is_valid': False, 'labels_found': [], 'reason': 'No labels found'}
        
        # Check if ALL labels are in range 1-10 (C1-C7 = 1-7, T1-T3 = 8-10)
        all_valid = all(1 <= label <= 10 for label in unique_labels)
        
        if not all_valid:
            invalid_labels = [label for label in unique_labels if label < 1 or label > 10]
            return {
                'is_valid': False,
                'labels_found': list(unique_labels),
                'reason': f'Contains labels outside C1-T3 range: {invalid_labels}'
            }
        
        return {
            'is_valid': True,
            'labels_found': list(unique_labels),
            'reason': 'Valid C1-T3 scan'
        }
        
    except Exception as e:
        return {'is_valid': False, 'labels_found': [], 'reason': f'Error reading file: {e}'}


def filter_rsna_dataset(input_dir: Path, output_dir: Path):
    """
    Filter RSNA dataset to only include cases with C1-T3 labels (1-10).
    """
    volumes_dir = input_dir / "volumes"
    segmentations_dir = input_dir / "segmentations"
    
    out_volumes_dir = output_dir / "imagesTr"
    out_segmentations_dir = output_dir / "labelsTr"
    
    # Create output directories
    out_volumes_dir.mkdir(parents=True, exist_ok=True)
    out_segmentations_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all segmentation files
    seg_files = sorted(segmentations_dir.glob("case_*.nii"))
    
    print(f"Found {len(seg_files)} segmentation files")
    print(f"\n{'='*60}")
    print("Filtering dataset for C1-T3 only...")
    print(f"{'='*60}\n")
    
    valid_count = 0
    filtered_count = 0
    
    for seg_file in seg_files:
        case_id = seg_file.stem  # e.g., "case_0000"
        vol_file = volumes_dir / f"{case_id}.nii"
        
        # Check if corresponding volume exists
        if not vol_file.exists():
            print(f"  WARNING: Volume not found for {case_id}")
            continue
        
        # Check segmentation labels
        label_check = check_cervical_to_t3_labels(seg_file)
        
        if label_check['is_valid']:
            # Copy both files with nnUNet naming
            # Volume: {case_id}_0000.nii.gz
            # Segmentation: {case_id}.nii.gz
            shutil.copy2(vol_file, out_volumes_dir / f"{case_id}_0000.nii.gz")
            shutil.copy2(seg_file, out_segmentations_dir / f"{case_id}.nii.gz")
            
            valid_count += 1
            print(f"  ✓ {case_id} - Labels: {label_check['labels_found']}")
        else:
            filtered_count += 1
            if filtered_count <= 10:  # Only print first 10 filtered to avoid spam
                print(f"  ✗ {case_id} - {label_check['reason']}")
                if label_check['labels_found']:
                    print(f"      Labels: {label_check['labels_found']}")
    
    if filtered_count > 10:
        print(f"  ... and {filtered_count - 10} more filtered cases")
    
    print(f"\n{'='*60}")
    print(f"Valid cases (C1-T3 only): {valid_count}")
    print(f"Filtered cases: {filtered_count}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")
    
    return valid_count


def main():
    input_dir = Path("C:\\Users\\anoma\\Downloads\\surgipath-datasets\\RSNA")
    output_dir = Path("RSNA_out")
    
    # Validate input structure
    if not (input_dir / "volumes").exists():
        print("ERROR: 'volumes' directory not found in current directory")
        return
    
    if not (input_dir / "segmentations").exists():
        print("ERROR: 'segmentations' directory not found in current directory")
        return
    
    # Filter dataset
    num_valid = filter_rsna_dataset(input_dir, output_dir)
    
    if num_valid == 0:
        print("\nWARNING: No valid cases found!")
    else:
        print(f"\nSUCCESS: {num_valid} cases copied to {output_dir}/")


if __name__ == "__main__":
    main()