from pathlib import Path
import shutil

def copy_corresponding_labels(images_dir: Path, labels_source_dirs: list[Path], labels_dest_dir: Path):
    """
    For each image file in images_dir, find and copy its corresponding label file.
    """
    labels_dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(images_dir.glob("*_0000.nii.gz"))
    
    print(f"Found {len(image_files)} image files in {images_dir.name}")
    print(f"Searching for corresponding labels...\n")
    
    found_count = 0
    missing_count = 0
    
    for img_file in image_files:
        # Get case ID by removing _0000.nii.gz suffix
        case_id = img_file.name.replace("_0000.nii.gz", "")
        label_name = f"{case_id}.nii.gz"
        
        # Search for label in all source directories
        found = False
        for labels_source in labels_source_dirs:
            label_path = labels_source / label_name
            
            if label_path.exists():
                shutil.copy2(label_path, labels_dest_dir / label_name)
                found_count += 1
                found = True
                print(f"  ✓ Copied: {label_name}")
                break
        
        if not found:
            missing_count += 1
            print(f"  ✗ Missing: {label_name}")
    
    print(f"\n{'='*60}")
    print(f"Labels found and copied: {found_count}")
    print(f"Labels missing: {missing_count}")
    print(f"{'='*60}")


def main():
    # Define source directories for labels
    labels_sources = [
        Path("C:/Users/anoma/Downloads/surgipath-datasets/VerSe_clean/labelsTr"),
        Path("C:/Users/anoma/Downloads/surgipath-datasets/RSNA_clean/labelsTr")
    ]
    
    # Training labels
    print("=" * 60)
    print("Processing imagesTr → labelsTr")
    print("=" * 60)
    
    train_images = Path("C:/Users/anoma/Downloads/surgipath-datasets/collective/imagesTr")
    train_labels_dest = Path("C:/Users/anoma/Downloads/surgipath-datasets/collective/labelsTr")
    
    copy_corresponding_labels(train_images, labels_sources, train_labels_dest)
    
    # Test labels
    print("\n" + "=" * 60)
    print("Processing imagesTs → labelsTs")
    print("=" * 60)
    
    test_images = Path("C:/Users/anoma/Downloads/surgipath-datasets/collective/imagesTs")
    test_labels_dest = Path("C:/Users/anoma/Downloads/surgipath-datasets/collective/labelsTs")
    
    copy_corresponding_labels(test_images, labels_sources, test_labels_dest)


if __name__ == "__main__":
    main()