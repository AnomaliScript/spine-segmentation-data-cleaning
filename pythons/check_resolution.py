import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd
import shutil

SLICE_THICKNESS_THRESHOLD_MM = 0.625  # mm
DRY_RUN = True  # Set to False to actually delete files
PREFIX = "VerSe" # Dataset prefix used in filenames

def get_nifti_metadata(file_path):
    """Extract key metadata from NIfTI file."""
    img = nib.load(file_path)
    header = img.header
    # Use header.get_data_shape() to obtain the image shape without requiring
    # the high-level attribute .shape on the image object.
    shape = header.get_data_shape()
    # Ensure we have at least 3 dimensions for downstream indexing
    if len(shape) < 3:
        shape = tuple(list(shape) + [1] * (3 - len(shape)))
    zooms = header.get_zooms()[:3]
    
    return {
        'filename': file_path.name,
        'shape': shape,
        'voxel_size': zooms,  # mm
        'slice_thickness': zooms[2],  # z-axis resolution
        'dimensions': f"{shape[0]}x{shape[1]}x{shape[2]}",
        'resolution': f"{zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f}mm",
        'file_size_mb': file_path.stat().st_size / (1024 * 1024)
    }

def purge_and_renumber_dataset(base_path):
    """
    Remove low-quality files (thick slices) and renumber remaining files sequentially.
    
    Args:
        base_path: Path to {PREFIX} folder
        SLICE_THICKNESS_THRESHOLD_MM: Remove files with slice thickness > this (mm)
        DRY_RUN: If True, only show what would be deleted without actually deleting
    """
    base_path = Path(base_path)
    vol_folder = base_path / "clean_volumes"
    seg_folder = base_path / "clean_labels"
    
    print("=" * 100)
    print("PURGE AND RENUMBER LOW-QUALITY FILES")
    print("=" * 100)
    print(f"Slice thickness threshold: >{SLICE_THICKNESS_THRESHOLD_MM}mm will be REMOVED")
    print(f"Mode: {'DRY RUN (no files will be deleted)' if DRY_RUN else 'LIVE RUN (files WILL be deleted)'}")
    print()
    
    # Collect all volume files with metadata
    vol_files = sorted(vol_folder.glob(f"{PREFIX}_*_0000.nii.gz"))
    
    if not vol_files:
        print("ERROR: No volume files found!")
        # Return empty structures so callers can safely unpack the result
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    print(f"Found {len(vol_files)} volume files")
    print()
    
    # Analyze each file
    file_data = []
    for vol_file in vol_files:
        # Extract case number from filename (e.g., {PREFIX}_001_0000.nii.gz -> 001)
        case_num = vol_file.stem.split('_')[1]
        seg_file = seg_folder / f"{PREFIX}_{case_num}.nii.gz"
        
        # Get metadata
        metadata = get_nifti_metadata(vol_file)
        metadata['case_num'] = case_num
        metadata['vol_path'] = vol_file
        metadata['seg_path'] = seg_file
        metadata['seg_exists'] = seg_file.exists()
        
        file_data.append(metadata)
    
    df = pd.DataFrame(file_data)
    
    # Identify files to remove
    files_to_remove = df[df['slice_thickness'] > SLICE_THICKNESS_THRESHOLD_MM]
    files_to_keep = df[df['slice_thickness'] <= SLICE_THICKNESS_THRESHOLD_MM]
    
    print("ANALYSIS:")
    print("-" * 100)
    print(f"Total files: {len(df)}")
    print(f"Files to REMOVE (slice thickness >{SLICE_THICKNESS_THRESHOLD_MM}mm): {len(files_to_remove)}")
    print(f"Files to KEEP: {len(files_to_keep)}")
    print()
    
    if len(files_to_remove) > 0:
        print("FILES TO BE REMOVED:")
        print("-" * 100)
        for idx, row in files_to_remove.iterrows():
            print(f"  ❌ {row['filename']} - Resolution: {row['resolution']}")
        print()
    
    if len(files_to_keep) > 0:
        print("FILES TO BE KEPT & RENUMBERED:")
        print("-" * 100)
        for new_idx, (old_idx, row) in enumerate(files_to_keep.iterrows(), start=1):
            old_name = row['filename']
            new_vol_name = f"{PREFIX}_{new_idx:03d}_0000.nii.gz"
            new_seg_name = f"{PREFIX}_{new_idx:03d}.nii.gz"
            print(f"  ✓ {old_name} → {new_vol_name} (Resolution: {row['resolution']})")
        print()
    
    # Stop here if dry run
    if DRY_RUN:
        print("=" * 100)
        print("DRY RUN COMPLETE - No files were modified")
        print("=" * 100)
        print("To actually delete and renumber files, run with DRY_RUN=False")
        print()
        return df, files_to_remove, files_to_keep
    
    # === ACTUAL DELETION AND RENAMING ===
    print("=" * 100)
    print("EXECUTING DELETION AND RENUMBERING...")
    print("=" * 100)
    print()
    
    # Step 1: Delete files
    print("Step 1: Deleting low-quality files...")
    deleted_count = 0
    for idx, row in files_to_remove.iterrows():
        vol_path = row['vol_path']
        seg_path = row['seg_path']
        
        try:
            if vol_path.exists():
                vol_path.unlink()
                print(f"  🗑️  Deleted: {vol_path.name}")
                deleted_count += 1
            
            if seg_path.exists():
                seg_path.unlink()
                print(f"  🗑️  Deleted: {seg_path.name}")
        except Exception as e:
            print(f"  ❌ Error deleting {vol_path.name}: {e}")
    
    print(f"\nDeleted {deleted_count} volume files and their segmentations")
    print()
    
    # Step 2: Rename remaining files to temporary names first (to avoid conflicts)
    print("Step 2: Renaming files to temporary names...")
    temp_mapping = []
    for idx, row in files_to_keep.iterrows():
        vol_path = row['vol_path']
        seg_path = row['seg_path']
        
        temp_vol_name = f"TEMP_{row['case_num']}_0000.nii.gz"
        temp_seg_name = f"TEMP_{row['case_num']}.nii.gz"
        
        temp_vol_path = vol_folder / temp_vol_name
        temp_seg_path = seg_folder / temp_seg_name
        
        try:
            if vol_path.exists():
                shutil.move(str(vol_path), str(temp_vol_path))
            if seg_path.exists():
                shutil.move(str(seg_path), str(temp_seg_path))
            
            temp_mapping.append({
                'temp_vol': temp_vol_path,
                'temp_seg': temp_seg_path
            })
        except Exception as e:
            print(f"  ❌ Error creating temp file: {e}")
    
    print(f"  ✓ Moved {len(temp_mapping)} pairs to temporary names")
    print()
    
    # Step 3: Rename to final sequential numbers
    print("Step 3: Renumbering to sequential format...")
    renamed_count = 0
    for new_idx, temp_files in enumerate(temp_mapping, start=1):
        new_vol_name = f"{PREFIX}_{new_idx:03d}_0000.nii.gz"
        new_seg_name = f"{PREFIX}_{new_idx:03d}.nii.gz"

        final_vol_path = vol_folder / new_vol_name
        final_seg_path = seg_folder / new_seg_name
        
        try:
            if temp_files['temp_vol'].exists():
                shutil.move(str(temp_files['temp_vol']), str(final_vol_path))
            if temp_files['temp_seg'].exists():
                shutil.move(str(temp_files['temp_seg']), str(final_seg_path))
            
            renamed_count += 1
            print(f"  ✓ Renamed to: {new_vol_name}")
        except Exception as e:
            print(f"  ❌ Error renaming: {e}")
    
    print()
    print("=" * 100)
    print("PURGE AND RENUMBER COMPLETE!")
    print("=" * 100)
    print(f"Deleted: {deleted_count} low-quality pairs")
    print(f"Renumbered: {renamed_count} high-quality pairs")
    print(f"Final dataset size: {renamed_count} cases")
    print()
    
    return df, files_to_remove, files_to_keep

if __name__ == "__main__":
    base_path = Path("C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\VerSe_clean_v3")
    
    # STEP 1: DRY RUN (preview what will be deleted)
    print("\n🔍 RUNNING DRY RUN - NO FILES WILL BE DELETED")
    print()
    df, to_remove, to_keep = purge_and_renumber_dataset(base_path)
    
    # STEP 2: Ask for confirmation
    print()
    print("=" * 100)
    response = input("Do you want to proceed with deletion and renumbering? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print()
        print("🚨 PROCEEDING WITH LIVE RUN - FILES WILL BE DELETED")
        print()
        purge_and_renumber_dataset(base_path)
    else:
        print()
        print("❌ Operation cancelled. No files were modified.")
        print()