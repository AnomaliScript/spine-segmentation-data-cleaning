import os
import gzip
import shutil

def get_nii_files(folder):
    """Collect all .nii and .nii.gz files in the folder recursively."""
    nii_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return sorted(nii_files)

def get_file_pairs(images_folder, labels_folder):
    """Identify corresponding image-segmentation pairs by base filename."""
    images_files = get_nii_files(images_folder)
    labels_files = get_nii_files(labels_folder)
    
    # Extract base names (remove _0000.nii.gz, .nii.gz, or .nii)
    images_base = {os.path.basename(f).replace('_0000.nii.gz', '').replace('.nii.gz', ''): f for f in images_files}
    labels_base = {os.path.basename(f).replace('.nii.gz', '').replace('.nii', ''): f for f in labels_files}
    
    # Find matching pairs and unpaired files
    pairs = []
    unpaired_images = []
    unpaired_labels = []
    
    for base_name in images_base:
        if base_name in labels_base:
            pairs.append((images_base[base_name], labels_base[base_name]))
        else:
            unpaired_images.append(images_base[base_name])
    
    for base_name in labels_base:
        if base_name not in images_base:
            unpaired_labels.append(labels_base[base_name])
    
    return pairs, unpaired_images, unpaired_labels

def rename_and_compress_nii_files(images_folder, labels_folder):
    """Rename files in imagesTs to CVPP_XXXX_0000.nii.gz and labelsTs to CVPP_XXXX.nii.gz, starting IDs at 0071."""
    images_folder = r"C://Users//anoma//Downloads//surgipath-datasets//collective//imagesTs"
    labels_folder = r"C://Users//anoma//Downloads//surgipath-datasets//collective//labelsTs"
    
    # Get file pairs and unpaired files
    pairs, unpaired_images, unpaired_labels = get_file_pairs(images_folder, labels_folder)
    
    if not (pairs or unpaired_images or unpaired_labels):
        print("No .nii or .nii.gz files found in either folder.")
        return
    
    # Assign IDs to pairs and unpaired files, starting at 0071
    new_names = {}
    used_ids = set()
    next_id = 71  # Start at 071
    
    # Process paired files
    for image_path, label_path in pairs:
        while f"{next_id:03d}" in used_ids:
            next_id += 1
        id_num = f"{next_id:03d}"
        used_ids.add(id_num)
        
        # Image file (always .nii.gz, with _0000)
        new_image_name = f"CVPP_{id_num}_0000.nii.gz"
        new_image_path = os.path.join(os.path.dirname(image_path), new_image_name)
        new_names[image_path] = new_image_path
        
        # Label file (could be .nii or .nii.gz, no _0000)
        ext = '.nii.gz' if label_path.endswith('.nii.gz') or '13089.nii' in label_path else '.nii'
        new_label_name = f"CVPP_{id_num}{ext}"
        new_label_path = os.path.join(os.path.dirname(label_path), new_label_name)
        new_names[label_path] = new_label_path
        
        next_id += 1
    
    # Process unpaired images
    for image_path in unpaired_images:
        while f"{next_id:03d}" in used_ids:
            next_id += 1
        id_num = f"{next_id:03d}"
        used_ids.add(id_num)
        new_image_name = f"CVPP_{id_num}_0000.nii.gz"
        new_image_path = os.path.join(os.path.dirname(image_path), new_image_name)
        new_names[image_path] = new_image_path
        next_id += 1
    
    # Process unpaired labels — FIXED: was 'label_label_path'
    for label_path in unpaired_labels:
        while f"{next_id:03d}" in used_ids:
            next_id += 1
        id_num = f"{next_id:03d}"
        used_ids.add(id_num)
        ext = '.nii.gz' if label_path.endswith('.nii.gz') or '13089.nii' in label_path else '.nii'
        new_label_name = f"CVPP_{id_num}{ext}"
        new_label_path = os.path.join(os.path.dirname(label_path), new_label_name)
        new_names[label_path] = new_label_path
        next_id += 1
    
    # Rename files
    for old_path, new_path in new_names.items():
        if old_path != new_path:
            if os.path.exists(new_path):
                print(f"Warning: {new_path} already exists, skipping rename for {old_path}")
                continue
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
    
    # Compress the single .nii file (13089.nii)
    for old_path, new_path in new_names.items():
        if new_path.endswith('.nii') and '13089.nii' in old_path:
            compressed_path = new_path + '.gz'
            with open(new_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Compressed: {new_path} -> {compressed_path}")
            os.remove(new_path)
            print(f"Deleted original: {new_path}")

if __name__ == "__main__":
    rename_and_compress_nii_files(
        r"C://Users//anoma//Downloads//surgipath-datasets//collective//imagesTs",
        r"C://Users//anoma//Downloads//surgipath-datasets//collective//labelsTs"
    )
    print("Processing completed.")