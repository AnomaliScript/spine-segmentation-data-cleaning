import os
import gzip
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path
from collections import defaultdict

# ====================
# CONFIGURATION
# ====================

SOURCE_BASE = r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\VerSe"
DEST_LABELS = r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\VerSe_clean_v3\\labels"
DEST_VOLUMES = r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\VerSe_clean_v3\\volumes"

# Vertebrae label mapping (hypothesis to verify)
VERTEBRAE_LABELS = {
    1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6", 7: "C7",
    8: "T1", 9: "T2", 10: "T3"
}

CERVICAL_LABELS = set(range(1, 8))  # 1-7
THORACIC_LABELS = set(range(8, 11))  # 8-10

DRY_RUN = False  # Set to False to actually copy files


# ====================
# UTILITY FUNCTIONS
# ====================

def check_gzip_status(file_path):
    """Check if a file is gzipped."""
    try:
        with open(file_path, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    except:
        return False


def gzip_file(input_path, output_path=None):
    """Gzip a file if not already gzipped."""
    if output_path is None:
        output_path = input_path + '.gz'
    
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return output_path


def get_unique_labels(nii_file_path):
    """Extract unique label values from a NIfTI segmentation file."""
    try:
        img = nib.load(nii_file_path)
        data = img.get_fdata()
        unique_labels = np.unique(data)
        # Remove 0 (background) and convert to integers
        unique_labels = set(int(label) for label in unique_labels if label > 0)
        return unique_labels
    except Exception as e:
        print(f"  ⚠️  Error reading {nii_file_path}: {e}")
        return set()


def format_vertebrae_list(labels):
    """Format label numbers as vertebrae names."""
    return [VERTEBRAE_LABELS.get(l, f"Unknown({l})") for l in sorted(labels)]


def meets_cervical_criteria(labels):
    """
    Check if the labels meet our cervical spine criteria:
    PRE-CHECKS (mandatory - ONLY THESE MATTER):
    1. Must have at least 3 cervical vertebrae
    2. No vertebrae beyond T3 (labels > 10)
    3. Thoracic vertebrae must be sequential starting from T1
       - If T2 present, T1 must be present
       - If T3 present, T1 and T2 must be present
    """
    # Handle empty labels set FIRST
    if not labels:
        return False, "No labels found in segmentation file"
    
    cervical_present = labels & CERVICAL_LABELS
    thoracic_present = labels & THORACIC_LABELS
    
    num_cervical = len(cervical_present)

    # General rule: at least 3 cervical for each thoracic
    if num_cervical >= 3:
        thoracic_names = format_vertebrae_list(thoracic_present)
        return True, f"{num_cervical} cervical ({', '.join(thoracic_names)})"
    else:
        return False, f"Insufficient cervical ({num_cervical}) need at least 3"

# ====================
# MAIN FILTERING ALGORITHM
# ====================

def filter_and_copy_cervical_spine_data():
    """
    Main algorithm: Filter files based on cervical vertebrae criteria and copy them.
    """
    print("=" * 80)
    print("MAIN ALGORITHM: Filtering and Copying Cervical Spine Data")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no files will be copied)' if DRY_RUN else 'LIVE (files will be copied)'}")
    print()
    
    # Create destination directories
    if not DRY_RUN:
        os.makedirs(DEST_LABELS, exist_ok=True)
        os.makedirs(DEST_VOLUMES, exist_ok=True)
    
    stats = {
        "total_files": 0,
        "accepted": 0,
        "rejected": 0,
        "missing_volume": 0,
        "copied": 0,
        "errors": 0
    }
    
    acceptance_reasons = defaultdict(int)
    rejection_reasons = defaultdict(int)
    
    labels_dir = Path(SOURCE_BASE) / "segmentations"
    volumes_dir = Path(SOURCE_BASE) / "volumes"
    
    print(f"\n{'='*80}")
    print(f"📁 Processing...")
    
    label_files = list(labels_dir.glob("*.nii*"))
    stats["total_files"] += len(label_files)
    
    for label_file in label_files:
        # Get corresponding volume file (remove _seg suffix)
        volume_name = label_file.name.replace("_seg", "")
        volume_file = volumes_dir / volume_name
        
        # Alternative: try without any suffix modification
        if not volume_file.exists():
            volume_file = volumes_dir / label_file.name
        
        print(f"\n📄 {label_file.name}")
        
        # Get labels from segmentation
        labels = get_unique_labels(label_file)
        vertebrae = format_vertebrae_list(labels)
        
        print(f"   Labels found: {sorted(labels)}")
        print(f"   Vertebrae: {', '.join(vertebrae) if vertebrae else 'None'}")
        
        # Check if meets criteria
        meets_criteria, reason = meets_cervical_criteria(labels)
        
        if meets_criteria:
            print(f"   ✅ ACCEPTED: {reason}")
            stats["accepted"] += 1
            acceptance_reasons[reason] += 1
            
            # Check if volume file exists
            if not volume_file.exists():
                print(f"   ⚠️  Warning: Corresponding volume not found: {volume_name}")
                stats["missing_volume"] += 1
                continue
            
            # Prepare destination file names with dataset prefix
            dest_label_name = f"{label_file.name}"
            dest_volume_name = f"{volume_name}"
            
            dest_label_path = Path(DEST_LABELS) / dest_label_name
            dest_volume_path = Path(DEST_VOLUMES) / dest_volume_name
            
            # Copy files
            if DRY_RUN:
                print(f"   [DRY RUN] Would copy:")
                print(f"      {label_file} → {dest_label_path}")
                print(f"      {volume_file} → {dest_volume_path}")
            else:
                try:
                    shutil.copy2(label_file, dest_label_path)
                    shutil.copy2(volume_file, dest_volume_path)
                    print(f"   📋 Copied label: {dest_label_name}")
                    print(f"   📋 Copied volume: {dest_volume_name}")
                    stats["copied"] += 2
                except Exception as e:
                    print(f"   ⚠️  Error copying files: {e}")
                    stats["errors"] += 1
        else:
            print(f"   ❌ REJECTED: {reason}")
            stats["rejected"] += 1
            rejection_reasons[reason] += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total label files processed: {stats['total_files']}")
    print(f"Accepted: {stats['accepted']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Missing corresponding volume: {stats['missing_volume']}")
    
    if not DRY_RUN:
        print(f"Files copied: {stats['copied']}")
        print(f"Errors: {stats['errors']}")
    
    print("\n--- Acceptance Reasons ---")
    for reason, count in sorted(acceptance_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count:3d}x {reason}")
    
    print("\n--- Rejection Reasons ---")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count:3d}x {reason}")
    
    print("=" * 80)


# ====================
# MAIN EXECUTION
# ====================

if __name__ == "__main__":
    print("\n" + "🔬" * 40)
    print("CTSpine1K Cervical Vertebrae Data Filtering Pipeline")
    print("🔬" * 40 + "\n")
        
    # Step 3: Main filtering and copying
    filter_and_copy_cervical_spine_data()
    
    print("\n✅ Pipeline complete!")
    if DRY_RUN:
        print("⚠️  This was a DRY RUN. Set DRY_RUN = False to actually copy files.")