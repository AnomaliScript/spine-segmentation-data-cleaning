import nibabel as nib
import numpy as np
from pathlib import Path

def diagnose_segmentations(labels_dir: Path, num_samples: int = 5):
    """
    Diagnose what's wrong with segmentation files.
    """
    labels_dir = Path(labels_dir)
    
    print("="*80)
    print("SEGMENTATION DIAGNOSTIC")
    print("="*80)
    print(f"Checking: {labels_dir}")
    print()
    
    label_files = sorted(list(labels_dir.glob("*.nii.gz")) + list(labels_dir.glob("*.nii")))
    
    if not label_files:
        print("ERROR: No label files found!")
        return
    
    print(f"Found {len(label_files)} label files")
    print()
    
    # Check first few files
    sample_files = label_files[:num_samples]
    
    for label_file in sample_files:
        print("="*80)
        print(f"FILE: {label_file.name}")
        print("="*80)
        
        # File size
        size_bytes = label_file.stat().st_size
        size_kb = size_bytes / 1024
        size_mb = size_kb / 1024
        print(f"File size: {size_bytes:,} bytes ({size_kb:.2f} KB, {size_mb:.2f} MB)")
        
        try:
            # Load and analyze
            img = nib.load(str(label_file))
            data = np.asarray(img.dataobj)
            
            print(f"Shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Total voxels: {data.size:,}")
            
            # Check unique values
            unique_labels = np.unique(data)
            print(f"Unique labels: {unique_labels.tolist()}")
            
            # Count non-zero voxels
            non_zero = np.count_nonzero(data)
            percent_labeled = (non_zero / data.size) * 100
            print(f"Non-zero voxels: {non_zero:,} ({percent_labeled:.2f}%)")
            
            # Check if it's all zeros
            if non_zero == 0:
                print("⚠ WARNING: This file is ALL ZEROS - empty segmentation!")
            
            # Check label distribution
            if len(unique_labels) > 1:
                print("\nLabel distribution:")
                for label in unique_labels:
                    if label == 0:
                        continue
                    count = np.sum(data == label)
                    print(f"  Label {int(label)}: {count:,} voxels")
            
            # Check affine/spacing
            print(f"\nVoxel spacing: {img.header.get_zooms()[:3]}")
            
        except Exception as e:
            print(f"ERROR loading file: {e}")
        
        print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    total_size = sum(f.stat().st_size for f in label_files)
    avg_size_kb = (total_size / len(label_files)) / 1024
    avg_size_mb = avg_size_kb / 1024
    
    print(f"Average file size: {avg_size_kb:.2f} KB ({avg_size_mb:.2f} MB)")
    
    # Check for suspiciously small files
    tiny_files = [f for f in label_files if f.stat().st_size < 50000]  # < 50 KB
    
    if tiny_files:
        print(f"\n⚠ Found {len(tiny_files)} suspiciously small files (< 50 KB):")
        for f in tiny_files[:10]:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name}: {size_kb:.2f} KB")
        if len(tiny_files) > 10:
            print(f"  ... and {len(tiny_files) - 10} more")
    
    print()
    print("DIAGNOSIS:")
    print("-" * 80)
    if avg_size_kb < 100:
        print("• Files are VERY small - likely empty or corrupted")
        print("• Possible causes:")
        print("  1. Segmentation files are empty/all zeros")
        print("  2. Files got corrupted during copying")
        print("  3. Wrong files were copied (not the actual segmentations)")
        print("\nRECOMMENDATION:")
        print("  - Check the ORIGINAL segmentation files in DukeCSpineSeg_segmentation/")
        print("  - Compare their sizes with what's in labels/ folder")
    else:
        print("• File sizes look reasonable")
    print()


if __name__ == "__main__":
    # Check the labels folder
    labels_dir = Path(r"C:\Users\anoma\Downloads\spine-segmentation-data-cleaning\DukeCSS\labels")
    diagnose_segmentations(labels_dir, num_samples=5)
    
    print("\n" + "="*80)
    print("NOW CHECKING ORIGINAL SEGMENTATIONS FOR COMPARISON")
    print("="*80)
    print()
    
    # Also check original segmentations
    orig_seg_dir = Path(r"C:\Users\anoma\Downloads\spine-segmentation-data-cleaning\DukeCSS\DukeCSpineSeg_segmentation")
    
    if orig_seg_dir.exists():
        orig_files = sorted(orig_seg_dir.glob("*.nii.gz"))[:3]
        for f in orig_files:
            size_kb = f.stat().st_size / 1024
            print(f"{f.name}: {size_kb:.2f} KB")
    else:
        print("Original segmentation directory not found")