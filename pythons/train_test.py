from pathlib import Path
import shutil
import random

def split_dataset(source_dir: Path, train_dir: Path, test_dir: Path, test_ratio: float = 0.2):
    # Randomly split files from source into train (80%) and test (20%) directories.
    # Get all _0000.nii.gz files (volumes)
    files = sorted(source_dir.glob("*_0000.nii.gz"))
    
    # Shuffle randomly
    random.shuffle(files)
    
    # Calculate split point
    num_test = int(len(files) * test_ratio)
    
    test_files = files[:num_test]
    train_files = files[num_test:]
    
    print(f"  Total: {len(files)} | Test: {len(test_files)} | Train: {len(train_files)}")
    
    # Copy files
    for f in test_files:
        shutil.copy2(f, test_dir / f.name)
    
    for f in train_files:
        shutil.copy2(f, train_dir / f.name)
    
    return test_files, train_files


def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    verse_images = Path("C:/Users/anoma/Downloads/surgipath-datasets/VerSe_clean/imagesTr")
    rsna_images = Path("C:/Users/anoma/Downloads/surgipath-datasets/RSNA_clean/imagesTr")
    
    collective_train = Path("C:/Users/anoma/Downloads/surgipath-datasets/collective/imagesTr")
    collective_test = Path("C:/Users/anoma/Downloads/surgipath-datasets/collective/imagesTs")
    
    # Create output directories
    collective_train.mkdir(parents=True, exist_ok=True)
    collective_test.mkdir(parents=True, exist_ok=True)
    
    print("Splitting VerSe dataset...")
    verse_test, verse_train = split_dataset(verse_images, collective_train, collective_test)
    
    print("\nSplitting RSNA dataset...")
    rsna_test, rsna_train = split_dataset(rsna_images, collective_train, collective_test)
    
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"Training images: {len(verse_train) + len(rsna_train)}")
    print(f"Testing images: {len(verse_test) + len(rsna_test)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()