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


def process_rsna_dataset(base_path, threshold=7):
    """
    Process RSNA dataset:
    1. Find volume/segmentation pairs in volumes/ and labels/ folders
    2. Rename to nnUNet format (RSNA_xxx_0000.nii.gz and RSNA_xxx.nii.gz)
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
    
    print(f"Processing RSNA dataset at: {base_path}")
    print(f"Source folders:")
    print(f"  - Volumes: {volumes_path}")
    print(f"  - Labels:  {labels_path}")
    print(f"Output folders:")
    print(f"  - Volumes: {renamed_vol_path}")
    print(f"  - Labels:  {cervical_labels_path}")
    print()
    
    # Find all volume files (assuming .nii.gz format)
    vol_files = sorted(volumes_path.glob("*.nii.gz"))
    
    if not vol_files:
        print("ERROR: No volume files found matching pattern '*.nii.gz'")
        print(f"Please check if files exist in: {volumes_path}")
        return
    
    print(f"Found {len(vol_files)} volume files")
    print()
    
    # Find all label files
    label_files = sorted(labels_path.glob("*.nii.gz"))
    
    if not label_files:
        print("ERROR: No label files found matching pattern '*.nii.gz'")
        print(f"Please check if files exist in: {labels_path}")
        return
    
    print(f"Found {len(label_files)} label files")
    print()
    
    # Create a mapping between volumes and labels
    # Assuming they have similar base names (e.g., case1.nii.gz and case1_seg.nii.gz)
    # or identical names in different folders
    
    vol_basenames = {}
    for vol_file in vol_files:
        # Remove common suffixes to find base name
        basename = vol_file.stem.replace('.nii', '')
        # Store both with and without potential suffixes
        vol_basenames[basename] = vol_file
        # Also try without common volume suffixes
        clean_name = basename.replace('_volume', '').replace('_vol', '').replace('_image', '')
        vol_basenames[clean_name] = vol_file
    
    label_basenames = {}
    for label_file in label_files:
        basename = label_file.stem.replace('.nii', '')
        label_basenames[basename] = label_file
        # Also try without common label suffixes
        clean_name = basename.replace('_seg', '').replace('_label', '').replace('_mask', '').replace('_segmentation', '')
        label_basenames[clean_name] = label_file
    
    # Find matching pairs
    print("Matching volume-label pairs...")
    pairs = []
    
    for vol_file in vol_files:
        vol_basename = vol_file.stem.replace('.nii', '')
        vol_clean = vol_basename.replace('_volume', '').replace('_vol', '').replace('_image', '')
        
        # Try to find matching label
        label_file = None
        
        # Strategy 1: Exact match
        if vol_basename in label_basenames:
            label_file = label_basenames[vol_basename]
        # Strategy 2: Volume name + _seg
        elif f"{vol_basename}_seg" in label_basenames:
            label_file = label_basenames[f"{vol_basename}_seg"]
        # Strategy 3: Volume name + _label
        elif f"{vol_basename}_label" in label_basenames:
            label_file = label_basenames[f"{vol_basename}_label"]
        # Strategy 4: Clean name match
        elif vol_clean in label_basenames:
            label_file = label_basenames[vol_clean]
        # Strategy 5: Check if any label starts with volume basename
        else:
            for label_basename, lf in label_basenames.items():
                if label_basename.startswith(vol_clean) or vol_clean.startswith(label_basename.replace('_seg', '').replace('_label', '')):
                    label_file = lf
                    break
        
        if label_file:
            pairs.append((vol_file, label_file))
        else:
            print(f"  ⚠ No matching label found for volume: {vol_file.name}")
    
    print(f"Found {len(pairs)} matching volume-label pairs")
    print()
    
    if len(pairs) == 0:
        print("ERROR: No matching pairs found! Check your file naming convention.")
        print("\nExample volume files:")
        for vf in vol_files[:3]:
            print(f"  {vf.name}")
        print("\nExample label files:")
        for lf in label_files[:3]:
            print(f"  {lf.name}")
        return
    
    # Process each pair
    processed_count = 0
    skipped_count = 0
    
    for idx, (vol_file, label_file) in enumerate(pairs, start=1):
        # Generate RSNA format names
        nnunet_id = f"{idx:03d}"
        nnunet_vol_name = f"RSNA_{nnunet_id}_0000.nii.gz"
        nnunet_seg_name = f"RSNA_{nnunet_id}.nii.gz"
        
        out_vol_path = renamed_vol_path / nnunet_vol_name
        out_seg_path = cervical_labels_path / nnunet_seg_name
        
        print(f"Processing pair {idx:03d}:")
        print(f"  Volume: {vol_file.name} → {nnunet_vol_name}")
        print(f"  Label:  {label_file.name} → {nnunet_seg_name}")
        
        try:
            # Copy and rename volume (no modification needed)
            shutil.copy2(vol_file, out_vol_path)
            print(f"  ✓ Copied volume: {nnunet_vol_name}")
            
            # Process and save cleaned segmentation
            remove_labels_above_threshold(label_file, out_seg_path, threshold=threshold)
            
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
    base_path = r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\RSNA_clean_v3"
    
    # Process dataset (keeping labels 1-7 for cervical vertebrae)
    process_rsna_dataset(base_path, threshold=7)