import nibabel as nib
import numpy as np
from pathlib import Path
import shutil

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


def process_spine_dataset(base_path, threshold=7):
    """
    Process CTSpine1K dataset:
    1. Find volume/segmentation pairs
    2. Rename to nnUNet format (CTS1K_xxx_0000.nii.gz and CTS1K_xxx.nii.gz)
    3. Filter segmentations to keep only labels ≤ threshold
    4. Save to renamed_vol/ and labels_cervical_only/ folders
    """
    base_path = Path(base_path)
    volumes_path = base_path / "volumes"
    labels_path = base_path / "labels"
    
    # Create output directories
    renamed_vol_path = base_path / "renamed_vol"
    cervical_labels_path = base_path / "labels_cervical_only"
    
    renamed_vol_path.mkdir(exist_ok=True)
    cervical_labels_path.mkdir(exist_ok=True)
    
    print(f"Processing dataset at: {base_path}")
    print(f"Output folders:")
    print(f"  - Volumes: {renamed_vol_path}")
    print(f"  - Labels:  {cervical_labels_path}")
    print()
    
    # Find all segmentation files
    seg_files = sorted(labels_path.glob("*_seg.nii.gz"))
    
    if not seg_files:
        print("ERROR: No segmentation files found matching pattern '*_seg.nii.gz'")
        print(f"Please check if files exist in: {labels_path}")
        return
    
    print(f"Found {len(seg_files)} segmentation files")
    print()
    
    # Process each segmentation file
    processed_count = 0
    skipped_count = 0
    
    for idx, seg_file in enumerate(seg_files, start=1):
        # Derive volume filename by removing _seg suffix
        vol_name = seg_file.name.replace("_seg.nii.gz", ".nii.gz")
        vol_file = volumes_path / vol_name
        
        # Check if corresponding volume exists
        if not vol_file.exists():
            print(f"⚠ SKIPPED [{idx:03d}]: Volume not found for {seg_file.name}")
            print(f"  Expected: {vol_file}")
            skipped_count += 1
            continue
        
        # Generate nnUNet format names
        nnunet_id = f"{idx:03d}"
        nnunet_vol_name = f"CTS1K_{nnunet_id}_0000.nii.gz"
        nnunet_seg_name = f"CTS1K_{nnunet_id}.nii.gz"
        
        out_vol_path = renamed_vol_path / nnunet_vol_name
        out_seg_path = cervical_labels_path / nnunet_seg_name
        
        print(f"Processing pair {idx:03d}:")
        print(f"  Volume: {vol_name} → {nnunet_vol_name}")
        print(f"  Segmentation: {seg_file.name} → {nnunet_seg_name}")
        
        try:
            # Copy and rename volume (no modification needed)
            shutil.copy2(vol_file, out_vol_path)
            print(f"  ✓ Copied volume: {nnunet_vol_name}")
            
            # Process and save cleaned segmentation
            remove_labels_above_threshold(seg_file, out_seg_path, threshold=threshold)
            
            processed_count += 1
            print()
            
        except Exception as e:
            print(f"  ✗ ERROR processing pair: {e}")
            skipped_count += 1
            print()
    
    # Summary
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {processed_count} pairs")
    print(f"Skipped: {skipped_count} pairs")
    print()
    print(f"Output locations:")
    print(f"  Volumes:       {renamed_vol_path.absolute()}")
    print(f"  Segmentations: {cervical_labels_path.absolute()}")


if __name__ == "__main__":
    # Set your base path here
    base_path = r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\CTSpine1K"
    
    # Process dataset (keeping labels 1-7 for cervical vertebrae)
    process_spine_dataset(base_path, threshold=7)