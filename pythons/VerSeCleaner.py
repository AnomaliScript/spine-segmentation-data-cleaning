from pathlib import Path
import shutil
import json
import re
import nibabel as nib
import numpy as np


def extract_subject_info(filename: str) -> dict:
    """
    Extract subject ID and split ID from various VerSe naming conventions.
    Returns {'subject_id': str, 'split_id': str or None}
    """
    # Standard format: sub-{id}_split-{split}_ct.nii or sub-{id}_ct.nii
    match = re.search(r'sub-(gl\d+|verse\d+)', filename)
    if match:
        subject_id = match.group(1)
        
        # Check for split
        split_match = re.search(r'split-verse(\d+)', filename)
        split_id = split_match.group(1) if split_match else None
        
        return {'subject_id': subject_id, 'split_id': split_id}
    
    # Legacy format: GL{number}_CT or verse{number}_CT
    legacy_match = re.search(r'(GL|verse)(\d+)', filename, re.IGNORECASE)
    if legacy_match:
        prefix = legacy_match.group(1).lower()
        number = legacy_match.group(2)
        subject_id = f"{prefix}{number}"
        return {'subject_id': subject_id, 'split_id': None}
    
    return {'subject_id': None, 'split_id': None}


def check_relevant_labels(seg_path: Path) -> dict:
    """
    Check if segmentation contains ONLY relevant labels (1-7) and background (0).
    
    Vertebrae label mapping (standard VerSe):
    C1-C7: labels 1-7
    T1-T12: labels 8-19
    L1-L6: labels 20-25
    
    Returns {'is_valid': bool, 'labels_found': list, 'reason': str}
    """
    try:
        img = nib.load(seg_path) # type: ignore
        data = img.get_fdata() # type: ignore
        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
        
        if len(unique_labels) == 0:
            return {'is_valid': False, 'labels_found': [], 'reason': 'No labels found'}

        # Check if ALL labels are relevant (1-10) (Cervial + Thoratic up to T3 to get more data validated because very little files have only C1-C7)
        all_relevant = all(1 <= label <= 10 for label in unique_labels)
        
        if not all_relevant:
            non_relevant = [label for label in unique_labels if label < 1 or label > 7]
            return {
                'is_valid': False,
                'labels_found': list(unique_labels),
                'reason': f'Contains non-relevant labels: {non_relevant}'
            }
        
        return {
            'is_valid': True,
            'labels_found': list(unique_labels),
            'reason': 'Valid relevant-only scan'
        }
        
    except Exception as e:
        return {'is_valid': False, 'labels_found': [], 'reason': f'Error reading file: {e}'}

def find_nii_files(directory: Path) -> list[tuple[Path, dict]]:
    """
    Find all .nii files and extract their subject info.
    Returns list of (filepath, subject_info_dict)
    """
    files_info = []
    
    for subject_dir in directory.iterdir():
        if not subject_dir.is_dir():
            continue
        
        for file in subject_dir.iterdir():
            if file.suffix == '.nii':
                info = extract_subject_info(file.name)
                if info['subject_id']:
                    files_info.append((file, info))
    
    return files_info


def match_volumes_to_segmentations(volumes_dir: Path, segmentations_dir: Path) -> list[dict]:
    """
    Match volume files to their corresponding segmentation files.
    Only includes pairs where segmentation has ONLY relevant labels (1-7).
    Returns list of {'volume': Path, 'segmentation': Path, 'case_id': str}
    """
    # Get all volume and segmentation files with their metadata
    volume_files = find_nii_files(volumes_dir)
    seg_files = find_nii_files(segmentations_dir)
    
    print(f"Found {len(volume_files)} volume files")
    print(f"Found {len(seg_files)} segmentation files")
    
    # Create lookup dict for segmentations
    seg_lookup = {}
    for seg_path, seg_info in seg_files:
        subject_id = seg_info['subject_id']
        split_id = seg_info['split_id']
        key = (subject_id, split_id)
        seg_lookup[key] = seg_path
    
    # Match volumes to segmentations and filter by relevant-only criteria
    matched_pairs = []
    filtered_count = 0
    
    for vol_path, vol_info in volume_files:
        subject_id = vol_info['subject_id']
        split_id = vol_info['split_id']
        key = (subject_id, split_id)
        
        if key in seg_lookup:
            seg_path = seg_lookup[key]
            
            # Check if segmentation has ONLY relevant labels
            print(f"\n  Checking: {subject_id}" + (f"_split{split_id}" if split_id else ""))
            label_check = check_relevant_labels(seg_path)
            
            if label_check['is_valid']:
                # Create case ID
                if split_id:
                    case_id = f"{subject_id}_split{split_id}"
                else:
                    case_id = subject_id
                
                matched_pairs.append({
                    'volume': vol_path,
                    'segmentation': seg_path,
                    'case_id': case_id
                })
                print(f"    ✓ VALID - Labels: {label_check['labels_found']}")
            else:
                filtered_count += 1
                print(f"    ✗ FILTERED - {label_check['reason']}")
                if label_check['labels_found']:
                    print(f"      Labels found: {label_check['labels_found']}")
        else:
            print(f"  WARNING: No segmentation found for volume {vol_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Matched pairs: {len(matched_pairs)}")
    print(f"Filtered out (contains non-relevant labels): {filtered_count}")
    print(f"{'='*60}")
    
    return matched_pairs


def organize_to_nnunet(matched_pairs: list[dict], output_dir: Path):
    """
    Copy and rename files to nnUNetv2 format.
    """
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Organizing {len(matched_pairs)} cases to nnUNet format")
    print(f"{'='*60}")
    
    for pair in matched_pairs:
        case_id = pair['case_id']
        
        # Copy volume
        volume_out = images_dir / f"{case_id}_0000.nii.gz"
        shutil.copy2(pair['volume'], volume_out)
        print(f"  Copied volume: {case_id}_0000.nii.gz")
        
        # Copy segmentation
        seg_out = labels_dir / f"{case_id}.nii.gz"
        shutil.copy2(pair['segmentation'], seg_out)
        print(f"  Copied segmentation: {case_id}.nii.gz")
    
    return len(matched_pairs)


def create_nnunet_dataset_json(output_dir: Path, num_cases: int):
    """
    Creates dataset.json for nnUNetv2.
    """
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "0": "background",
            "1": "C1",
            "2": "C2",
            "3": "C3",
            "4": "C4",
            "5": "C5",
            "6": "C6",
            "7": "C7"
        },
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
    }
    
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"\nCreated dataset.json with {num_cases} training cases")


def main():
    # Set your paths here
    volumes_dir = Path("C:\\Users\\anoma\\Downloads\\surgipath-datasets\\VerSe\\volumes")
    segmentations_dir = Path("C:\\Users\\anoma\\Downloads\\surgipath-datasets\\VerSe\\segmentations")
    output_dir = Path("out_verse")
    
    # Validate input directories
    if not volumes_dir.exists():
        print(f"ERROR: Volumes directory not found: {volumes_dir}")
        return
    
    if not segmentations_dir.exists():
        print(f"ERROR: Segmentations directory not found: {segmentations_dir}")
        return
    
    # Match volumes to segmentations
    matched_pairs = match_volumes_to_segmentations(volumes_dir, segmentations_dir)
    
    if not matched_pairs:
        print("\nNo matched pairs found. Check your data structure.")
        return
    
    # Organize to nnUNet format
    num_processed = organize_to_nnunet(matched_pairs, output_dir)
    
    # Create dataset.json
    create_nnunet_dataset_json(output_dir, num_processed)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: Processed {num_processed} cases")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()