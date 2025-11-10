import nibabel as nib
import numpy as np
from pathlib import Path
import shutil

def count_cervical_vertebrae(seg_file: Path) -> int:
    """
    Count how many cervical vertebrae (labels 1-7) are present in segmentation.
    Returns the count of unique cervical labels found.
    """
    try:
        img = nib.load(str(seg_file))
        data = np.asarray(img.dataobj)
        unique_labels = np.unique(data)
        
        # Count cervical vertebrae (labels 1-7)
        cervical_labels = [label for label in unique_labels if 1 <= label <= 7]
        return len(cervical_labels)
    except Exception as e:
        print(f"  ERROR reading {seg_file.name}: {e}")
        return 0


def copy_or_gzip_file(input_path: Path, output_path: Path) -> bool:
    """
    Copy file to output. If already .gz, just copy. If .nii, copy as-is with .gz extension.
    This is MUCH faster than actually gzipping.
    Note: Output will always be .nii.gz but may not be compressed if input wasn't compressed.
    """
    try:
        if input_path.suffix == '.gz':
            # Already gzipped, just copy
            shutil.copy2(input_path, output_path)
        else:
            # Just copy the .nii file and name it .nii.gz
            # This is MUCH faster than actually compressing
            # nnUNet will handle compression during preprocessing anyway
            shutil.copy2(input_path, output_path)
        return True
    except Exception as e:
        print(f"  ERROR copying {input_path.name}: {e}")
        return False


def find_verse_pairs(base_path: Path) -> list[dict]:
    """
    Find all volume/segmentation pairs in VerSe dataset structure.
    Returns list of {'volume': Path, 'segmentation': Path, 'subject': str}
    """
    base_path = Path(base_path)
    volumes_dir = base_path / "volumes"
    segs_dir = base_path / "segmentations"
    
    if not volumes_dir.exists() or not segs_dir.exists():
        print(f"ERROR: Required directories not found!")
        print(f"  Volumes: {volumes_dir}")
        print(f"  Segmentations: {segs_dir}")
        return []
    
    pairs = []
    
    # Get all subject subdirectories in segmentations
    seg_subjects = [d for d in segs_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    
    for seg_subject_dir in sorted(seg_subjects):
        subject_id = seg_subject_dir.name  # e.g., "sub-gl003"
        
        # Find corresponding volume directory
        vol_subject_dir = volumes_dir / subject_id
        
        if not vol_subject_dir.exists():
            print(f"  WARNING: No volume directory found for {subject_id}")
            continue
        
        # Find segmentation files in this subject's directory
        seg_files = list(seg_subject_dir.glob("*.nii")) + list(seg_subject_dir.glob("*.nii.gz"))
        # Filter for mask files (usually contain 'msk' or 'seg')
        seg_files = [f for f in seg_files if 'msk' in f.name or 'seg' in f.name]
        
        # Find volume files
        vol_files = list(vol_subject_dir.glob("*.nii")) + list(vol_subject_dir.glob("*.nii.gz"))
        # Filter out segmentation files if they're in volumes directory
        vol_files = [f for f in vol_files if 'msk' not in f.name and 'seg' not in f.name]
        
        if not seg_files:
            print(f"  WARNING: No segmentation files found in {subject_id}")
            continue
        
        if not vol_files:
            print(f"  WARNING: No volume files found in {subject_id}")
            continue
        
        # Match segmentation with volume
        # Typically there's one volume and one segmentation per subject
        for seg_file in seg_files:
            # Try to find matching volume
            # Remove segmentation-specific parts from filename
            seg_base = seg_file.stem.replace('.nii', '')
            seg_base_clean = seg_base.replace('_seg-vert_msk', '').replace('_seg', '').replace('_msk', '')
            
            # Look for volume with similar name
            vol_file = None
            for vf in vol_files:
                vol_base = vf.stem.replace('.nii', '')
                if seg_base_clean in vol_base or vol_base in seg_base_clean:
                    vol_file = vf
                    break
            
            # If no match found, just take the first volume file (usually only one per subject)
            if vol_file is None and len(vol_files) == 1:
                vol_file = vol_files[0]
            
            if vol_file:
                pairs.append({
                    'volume': vol_file,
                    'segmentation': seg_file,
                    'subject': subject_id
                })
            else:
                print(f"  WARNING: Could not match volume for {seg_file.name} in {subject_id}")
    
    return pairs


def process_verse_dataset(base_path: Path, min_cervical_count: int = 3):
    """
    Process VerSe dataset:
    1. Find all volume/segmentation pairs
    2. Filter for cases with 3+ cervical vertebrae
    3. Gzip files if needed (NO!)
    4. Rename to VerSe_xxx_0000.nii.gz and VerSe_xxx.nii.gz format
    """
    base_path = Path(base_path)
    
    # Create output directories
    output_volumes = Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\VerSe_clean_v3") / "renamed_vol"
    output_labels = Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\VerSe_clean_v3") / "labels_cervical_only"

    output_volumes.mkdir(exist_ok=True)
    output_labels.mkdir(exist_ok=True)
    
    print("="*80)
    print("PROCESSING VerSe DATASET")
    print("="*80)
    print(f"Base directory: {base_path}")
    print(f"Output volumes: {output_volumes}")
    print(f"Output labels:  {output_labels}")
    print(f"Minimum cervical vertebrae required: {min_cervical_count}")
    print()
    
    # Step 1: Find all pairs
    print("Step 1: Finding volume/segmentation pairs...")
    pairs = find_verse_pairs(base_path)
    
    if not pairs:
        print("ERROR: No pairs found!")
        return
    
    print(f"  Found {len(pairs)} volume/segmentation pairs")
    print()
    
    # Step 2: Filter by cervical count and process
    print("Step 2: Filtering and processing files...")
    print()
    
    accepted_pairs = []
    rejected_pairs = []
    
    for pair in pairs:
        seg_file = pair['segmentation']
        subject = pair['subject']
        
        # Count cervical vertebrae
        cervical_count = count_cervical_vertebrae(seg_file)
        
        print(f"Checking {subject}:")
        print(f"  Segmentation: {seg_file.name}")
        print(f"  Cervical vertebrae found: {cervical_count}")
        
        if cervical_count >= min_cervical_count:
            print(f"  ✓ ACCEPTED (has {cervical_count} >= {min_cervical_count})")
            accepted_pairs.append(pair)
        else:
            print(f"  ✗ REJECTED (has {cervical_count} < {min_cervical_count})")
            rejected_pairs.append(pair)
        print()
    
    print()
    print("="*80)
    print(f"Filtering complete:")
    print(f"  Accepted: {len(accepted_pairs)} pairs")
    print(f"  Rejected: {len(rejected_pairs)} pairs")
    print("="*80)
    print()
    
    if not accepted_pairs:
        print("No files met the criteria. Exiting.")
        return
    
    # Step 3: Copy and rename accepted pairs
    print("Step 3: Copying and renaming accepted files...")
    print()
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for idx, pair in enumerate(accepted_pairs, start=1):
        vol_file = pair['volume']
        seg_file = pair['segmentation']
        subject = pair['subject']
        
        # Generate new names
        new_vol_name = f"VerSe_{idx:03d}_0000.nii"
        new_seg_name = f"VerSe_{idx:03d}.nii"
        
        out_vol_path = output_volumes / new_vol_name
        out_seg_path = output_labels / new_seg_name
        
        # Skip if already processed
        if out_vol_path.exists() and out_seg_path.exists():
            print(f"Skipping pair {idx:03d} ({subject}) - already processed")
            skipped_count += 1
            continue
        
        print(f"Processing pair {idx:03d}/{len(accepted_pairs)} ({subject}):")
        print(f"  Volume: {vol_file.name} → {new_vol_name}")
        print(f"  Segmentation: {seg_file.name} → {new_seg_name}")
        
        try:
            # Copy volume (fast copy, no compression)
            if copy_or_gzip_file(vol_file, out_vol_path):
                print(f"  ✓ Volume copied")
            else:
                raise Exception("Failed to process volume")
            
            # Copy segmentation (fast copy, no compression)
            if copy_or_gzip_file(seg_file, out_seg_path):
                print(f"  ✓ Segmentation copied")
            else:
                raise Exception("Failed to process segmentation")
            
            success_count += 1
            print(f"  Progress: {success_count}/{len(accepted_pairs)} complete")
            print()
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            error_count += 1
            print()
    
    # Final summary
    print("="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Successfully processed: {success_count} pairs")
    print(f"Skipped (already done): {skipped_count} pairs")
    print(f"Errors: {error_count} pairs")
    print()
    print(f"Output locations:")
    print(f"  Volumes:       {output_volumes.absolute()}")
    print(f"  Segmentations: {output_labels.absolute()}")
    print()


if __name__ == "__main__":
    # Configuration
    base_path = Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\VerSe")
    
    # Process dataset (only accept cases with 3+ cervical vertebrae)
    process_verse_dataset(base_path, min_cervical_count=3)