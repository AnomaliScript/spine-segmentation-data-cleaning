from pathlib import Path
import shutil
import json
import re
import nibabel as nib
import numpy as np
from typing import Optional


def check_vertebrae_ratio(labels: list[int]) -> dict:
    """
    Check if at least 2/3 (66.7%) of vertebrae are cervical.
    
    Cervical (C1-C7): labels 1-7
    Thoracic (T1-T12): labels 8-19
    
    Rule: cervical must be ≥ 2/3 of total vertebrae (laser focus on cervical)
    Examples:
        - C1-C7 (7C, 0T) → 100% cervical ✓
        - C1-T3 (7C, 3T) → 70% cervical ✓
        - C5-T3 (3C, 3T) → 50% cervical ✗ (needs ≥66.7%)
        - C6-T2 (2C, 2T) → 50% cervical ✗
        - C3-T1 (5C, 1T) → 83% cervical ✓
    
    Returns {'is_valid': bool, 'cervical_count': int, 'thoracic_count': int, 'cervical_percentage': float, 'reason': str}
    """
    cervical = [l for l in labels if 1 <= l <= 7]
    thoracic = [l for l in labels if 8 <= l <= 19]
    
    cervical_count = len(cervical)
    thoracic_count = len(thoracic)
    total_count = cervical_count + thoracic_count
    
    if total_count == 0:
        return {
            'is_valid': False,
            'cervical_count': 0,
            'thoracic_count': 0,
            'cervical_percentage': 0.0,
            'reason': 'No vertebrae found'
        }
    
    # Calculate percentage of cervical vertebrae
    cervical_percentage = (cervical_count / total_count) * 100
    
    # Must be at least 66.7% cervical
    if cervical_percentage >= 66.67:
        return {
            'is_valid': True,
            'cervical_count': cervical_count,
            'thoracic_count': thoracic_count,
            'cervical_percentage': cervical_percentage,
            'reason': f'Valid: {cervical_percentage:.1f}% cervical ({cervical_count}C/{total_count} total)'
        }
    else:
        return {
            'is_valid': False,
            'cervical_count': cervical_count,
            'thoracic_count': thoracic_count,
            'cervical_percentage': cervical_percentage,
            'reason': f'Only {cervical_percentage:.1f}% cervical ({cervical_count}C/{total_count} total), need ≥66.7%'
        }


def analyze_segmentation(seg_path: Path, max_label: int = 19) -> dict:
    """
    Analyze segmentation file for labels and vertebrae ratio.
    
    Args:
        seg_path: Path to segmentation file
        max_label: Maximum valid label (19 for C1-T12, can be higher for full spine)
    
    Returns dict with validation results
    """
    try:
        img = nib.load(seg_path) # type: ignore
        data = img.get_fdata() # type: ignore
        unique_labels = np.unique(data)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        if len(unique_labels) == 0:
            return {
                'is_valid': False,
                'labels_found': [],
                'cervical_count': 0,
                'thoracic_count': 0,
                'ratio': 0.0,
                'reason': 'No labels found'
            }
        
        # Check for invalid labels (beyond max_label)
        invalid_labels = [l for l in unique_labels if l > max_label]
        if invalid_labels:
            return {
                'is_valid': False,
                'labels_found': list(unique_labels),
                'cervical_count': 0,
                'thoracic_count': 0,
                'ratio': 0.0,
                'reason': f'Contains labels beyond T12: {invalid_labels}'
            }
        
        # Check vertebrae ratio
        ratio_check = check_vertebrae_ratio(list(unique_labels))
        
        return {
            'is_valid': ratio_check['is_valid'],
            'labels_found': list(unique_labels),
            'cervical_count': ratio_check['cervical_count'],
            'thoracic_count': ratio_check['thoracic_count'],
            'cervical_percentage': ratio_check['cervical_percentage'],
            'reason': ratio_check['reason']
        }
        
    except Exception as e:
        return {
            'is_valid': False,
            'labels_found': [],
            'cervical_count': 0,
            'thoracic_count': 0,
            'cervical_percentage': 0.0,
            'reason': f'Error reading file: {e}'
        }


# ==================== VerSe Dataset Handler ====================

def extract_verse_subject_info(filename: str) -> dict:
    """Extract subject ID and split ID from VerSe naming conventions."""
    match = re.search(r'sub-(gl\d+|verse\d+)', filename)
    if match:
        subject_id = match.group(1)
        split_match = re.search(r'split-verse(\d+)', filename)
        split_id = split_match.group(1) if split_match else None
        return {'subject_id': subject_id, 'split_id': split_id}
    
    legacy_match = re.search(r'(GL|verse)(\d+)', filename, re.IGNORECASE)
    if legacy_match:
        prefix = legacy_match.group(1).lower()
        number = legacy_match.group(2)
        subject_id = f"{prefix}{number}"
        return {'subject_id': subject_id, 'split_id': None}
    
    return {'subject_id': None, 'split_id': None}


def find_verse_nii_files(directory: Path) -> list[tuple[Path, dict]]:
    """Find all .nii files in VerSe directory structure."""
    files_info = []
    for subject_dir in directory.iterdir():
        if not subject_dir.is_dir():
            continue
        for file in subject_dir.iterdir():
            if file.suffix == '.nii':
                info = extract_verse_subject_info(file.name)
                if info['subject_id']:
                    files_info.append((file, info))
    return files_info


def process_verse_dataset(volumes_dir: Path, segmentations_dir: Path):
    """Process VerSe dataset and return valid matched pairs."""
    volume_files = find_verse_nii_files(volumes_dir)
    seg_files = find_verse_nii_files(segmentations_dir)
    
    print(f"  Found {len(volume_files)} volume files")
    print(f"  Found {len(seg_files)} segmentation files")
    
    # Create lookup for segmentations
    seg_lookup = {}
    for seg_path, seg_info in seg_files:
        key = (seg_info['subject_id'], seg_info['split_id'])
        seg_lookup[key] = seg_path
    
    matched_pairs = []
    stats = {'valid': 0, 'filtered_ratio': 0, 'filtered_labels': 0, 'no_match': 0}
    
    for vol_path, vol_info in volume_files:
        key = (vol_info['subject_id'], vol_info['split_id'])
        
        if key in seg_lookup:
            seg_path = seg_lookup[key]
            case_id = f"{vol_info['subject_id']}_split{vol_info['split_id']}" if vol_info['split_id'] else vol_info['subject_id']
            
            # Analyze segmentation
            analysis = analyze_segmentation(seg_path, max_label=19)
            
            if analysis['is_valid']:
                matched_pairs.append({
                    'volume': vol_path,
                    'segmentation': seg_path,
                    'case_id': case_id,
                    'labels': analysis['labels_found'],
                    'cervical_count': analysis['cervical_count'],
                    'thoracic_count': analysis['thoracic_count']
                })
                stats['valid'] += 1
                print(f"    ✓ {case_id}: {analysis['cervical_percentage']:.1f}% cervical ({analysis['cervical_count']}C/{analysis['cervical_count']+analysis['thoracic_count']}T total)")
            else:
                if 'ratio' in analysis['reason'].lower():
                    stats['filtered_ratio'] += 1
                else:
                    stats['filtered_labels'] += 1
                print(f"    ✗ {case_id}: {analysis['reason']}")
        else:
            stats['no_match'] += 1
    
    return matched_pairs, stats


# ==================== RSNA Dataset Handler ====================

def process_rsna_dataset(volumes_dir: Path, segmentations_dir: Path) -> tuple[list[dict], dict[str, int]]:
    """Process RSNA dataset and return valid matched pairs."""
    seg_files = sorted(segmentations_dir.glob("case_*.nii"))
    
    print(f"  Found {len(seg_files)} segmentation files")
    
    matched_pairs = []
    stats = {'valid': 0, 'filtered_ratio': 0, 'filtered_labels': 0, 'no_match': 0}
    
    for seg_file in seg_files:
        case_id = seg_file.stem  # e.g., "case_0000"
        vol_file = volumes_dir / f"{case_id}.nii"
        
        if not vol_file.exists():
            stats['no_match'] += 1
            continue
        
        # Analyze segmentation
        analysis = analyze_segmentation(seg_file, max_label=19)
        
        if analysis['is_valid']:
            matched_pairs.append({
                'volume': vol_file,
                'segmentation': seg_file,
                'case_id': case_id,
                'labels': analysis['labels_found'],
                'cervical_count': analysis['cervical_count'],
                'thoracic_count': analysis['thoracic_count']
            })
            stats['valid'] += 1
            print(f"    ✓ {case_id}: {analysis['cervical_percentage']:.1f}% cervical ({analysis['cervical_count']}C/{analysis['cervical_count']+analysis['thoracic_count']}T total)")
        else:
            if 'ratio' in analysis['reason'].lower():
                stats['filtered_ratio'] += 1
            else:
                stats['filtered_labels'] += 1
            print(f"    ✗ {case_id}: {analysis['reason']}")
    
    return matched_pairs, stats


# ==================== Main Processing ====================

def organize_to_nnunet(matched_pairs: list[dict], output_dir: Path, dataset_name: str):
    """Copy and rename files to nnUNetv2 format."""
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for i, pair in enumerate(matched_pairs):
        # Use original case_id with dataset prefix
        case_id = f"{dataset_name}_{pair['case_id']}"
        
        # Copy volume
        volume_out = images_dir / f"{case_id}_0000.nii.gz"
        shutil.copy2(pair['volume'], volume_out)
        
        # Copy segmentation
        seg_out = labels_dir / f"{case_id}.nii.gz"
        shutil.copy2(pair['segmentation'], seg_out)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(matched_pairs)} files...")


def create_dataset_json(output_dir: Path, num_cases: int, max_label: int = 19):
    """Create dataset.json for nnUNetv2."""
    labels_dict = {"0": "background"}
    
    # Cervical
    for i in range(1, 8):
        labels_dict[str(i)] = f"C{i}"
    
    # Thoracic (if needed)
    if max_label >= 8:
        for i in range(8, min(max_label + 1, 20)):
            labels_dict[str(i)] = f"T{i - 7}"
    
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels_dict,
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
    }
    
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)


def main():
    # ==================== CONFIGURATION ====================
    # Set your dataset paths here
    verse_dir = Path("C:\\Users\\anoma\\Downloads\\surgipath-datasets\\VerSe")
    rsna_dir = Path("C:\\Users\\anoma\\Downloads\\surgipath-datasets\\RSNA")
    output_dir = Path("C:\\Users\\anoma\\Downloads\\surgipath-datasets\\v2\\cleaned-backup")
    
    # Process which datasets?
    process_verse = True
    process_rsna = True
    
    # ==================== PROCESSING ====================
    all_pairs = []
    combined_stats = {
        'verse': {'valid': 0, 'filtered_ratio': 0, 'filtered_labels': 0, 'no_match': 0},
        'rsna': {'valid': 0, 'filtered_ratio': 0, 'filtered_labels': 0, 'no_match': 0}
    }
    
    # Process VerSe
    if process_verse and verse_dir.exists():
        print(f"\n{'='*70}")
        print("Processing VerSe Dataset")
        print(f"{'='*70}")
        
        verse_vols = verse_dir / "volumes"
        verse_segs = verse_dir / "segmentations"
        
        if verse_vols.exists() and verse_segs.exists():
            verse_pairs, verse_stats = process_verse_dataset(verse_vols, verse_segs)
            all_pairs.extend(verse_pairs)
            combined_stats['verse'] = verse_stats
            print(f"\n  VerSe Summary: {verse_stats['valid']} valid cases")
        else:
            print("  ERROR: VerSe volumes/segmentations directories not found")
    
    # Process RSNA
    if process_rsna and rsna_dir.exists():
        print(f"\n{'='*70}")
        print("Processing RSNA Dataset")
        print(f"{'='*70}")
        
        rsna_vols = rsna_dir / "volumes"
        rsna_segs = rsna_dir / "segmentations"
        
        if rsna_vols.exists() and rsna_segs.exists():
            rsna_pairs, rsna_stats = process_rsna_dataset(rsna_vols, rsna_segs)
            all_pairs.extend(rsna_pairs)
            combined_stats['rsna'] = rsna_stats
            print(f"\n  RSNA Summary: {rsna_stats['valid']} valid cases")
        else:
            print("  ERROR: RSNA volumes/segmentations directories not found")
    
    # ==================== OUTPUT ====================
    if not all_pairs:
        print("\n❌ No valid cases found!")
        return
    
    print(f"\n{'='*70}")
    print("Organizing to nnUNet format...")
    print(f"{'='*70}")
    
    # Organize files
    verse_count = len([p for p in all_pairs if 'verse' in str(p['volume']) or 'gl' in p['case_id'].lower()])
    rsna_count = len(all_pairs) - verse_count
    
    if verse_count > 0:
        verse_pairs_only = [p for p in all_pairs if 'verse' in str(p['volume']) or 'gl' in p['case_id'].lower()]
        organize_to_nnunet(verse_pairs_only, output_dir, "verse")
    
    if rsna_count > 0:
        rsna_pairs_only = [p for p in all_pairs if 'verse' not in str(p['volume']) and 'gl' not in p['case_id'].lower()]
        organize_to_nnunet(rsna_pairs_only, output_dir, "rsna")
    
    # Create dataset.json
    create_dataset_json(output_dir, len(all_pairs), max_label=19)
    
    # ==================== FINAL SUMMARY ====================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nVerSe Dataset:")
    print(f"  ✓ Valid:              {combined_stats['verse']['valid']}")
    print(f"  ✗ Filtered (ratio):   {combined_stats['verse']['filtered_ratio']}")
    print(f"  ✗ Filtered (labels):  {combined_stats['verse']['filtered_labels']}")
    print(f"  ✗ No match:           {combined_stats['verse']['no_match']}")
    
    print(f"\nRSNA Dataset:")
    print(f"  ✓ Valid:              {combined_stats['rsna']['valid']}")
    print(f"  ✗ Filtered (ratio):   {combined_stats['rsna']['filtered_ratio']}")
    print(f"  ✗ Filtered (labels):  {combined_stats['rsna']['filtered_labels']}")
    print(f"  ✗ No match:           {combined_stats['rsna']['no_match']}")
    
    print(f"\n{'='*70}")
    print(f"✅ TOTAL VALID CASES: {len(all_pairs)}")
    print(f"   - VerSe: {verse_count}")
    print(f"   - RSNA:  {rsna_count}")
    print(f"\n📁 Output: {output_dir.absolute()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()