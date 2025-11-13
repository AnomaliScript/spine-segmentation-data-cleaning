import nibabel as nib
import numpy as np
from pathlib import Path
from collections import Counter

def check_label_schema(labels_dir: Path, sample_size: int = 5):
    """
    Check the labeling schema of segmentation files.
    Verifies if labels follow the expected convention (1=C1, 2=C2, ..., 7=C7).
    
    Args:
        labels_dir: Directory containing label/segmentation files
        sample_size: Number of files to sample for checking
    """
    labels_dir = Path(labels_dir)
    
    print("="*80)
    print("LABEL SCHEMA CHECKER")
    print("="*80)
    print(f"Checking labels in: {labels_dir}")
    print()
    
    # Get all label files
    label_files = sorted(list(labels_dir.glob("*.nii.gz")) + list(labels_dir.glob("*.nii")))
    
    if not label_files:
        print("ERROR: No label files found!")
        return
    
    print(f"Found {len(label_files)} label files")
    print(f"Sampling {min(sample_size, len(label_files))} files for analysis")
    print()
    
    # Sample files evenly distributed
    if len(label_files) <= sample_size:
        sample_files = label_files
    else:
        step = len(label_files) // sample_size
        sample_files = [label_files[i * step] for i in range(sample_size)]
    
    # Analyze each sample
    all_labels = []
    label_counts = Counter()
    
    for label_file in sample_files:
        print(f"Analyzing: {label_file.name}")
        
        try:
            img = nib.load(str(label_file))
            data = np.asarray(img.dataobj)
            
            unique_labels = np.unique(data)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
            
            print(f"  Unique labels found: {sorted(unique_labels.tolist())}")
            
            # Count voxels per label
            for label in unique_labels:
                count = np.sum(data == label)
                print(f"    Label {int(label)}: {count:,} voxels")
                label_counts[int(label)] += 1
            
            all_labels.extend(unique_labels.tolist())
            print()
            
        except Exception as e:
            print(f"  ERROR: Could not read file: {e}")
            print()
    
    # Summary analysis
    print("="*80)
    print("SCHEMA ANALYSIS SUMMARY")
    print("="*80)
    print()
    
    # Get all unique labels across all samples
    unique_all = sorted(set(int(x) for x in all_labels))
    print(f"All unique labels across samples: {unique_all}")
    print()
    
    # Count how many files contain each label
    print("Label frequency across sampled files:")
    for label in sorted(label_counts.keys()):
        frequency = label_counts[label]
        percentage = (frequency / len(sample_files)) * 100
        print(f"  Label {label}: appears in {frequency}/{len(sample_files)} files ({percentage:.1f}%)")
    print()
    
    # Check if schema matches expected cervical convention
    print("SCHEMA VERIFICATION:")
    print("-" * 80)
    
    expected_cervical = set(range(1, 8))  # C1-C7 should be labels 1-7
    found_labels = set(unique_all)
    
    cervical_labels = found_labels & expected_cervical
    extra_labels = found_labels - expected_cervical
    
    print(f"Expected cervical labels (1-7): {sorted(expected_cervical)}")
    print(f"Found cervical labels (1-7):    {sorted(cervical_labels)}")
    
    if cervical_labels == expected_cervical:
        print("✓ All expected cervical labels found!")
    else:
        missing = expected_cervical - cervical_labels
        if missing:
            print(f"⚠ Missing cervical labels: {sorted(missing)}")
    
    if extra_labels:
        print(f"\n⚠ Additional labels found (non-cervical): {sorted(extra_labels)}")
        print("  These labels likely represent:")
        
        # Common vertebrae labeling schemes
        if any(8 <= l <= 19 for l in extra_labels):
            print("    - Labels 8-19: Thoracic vertebrae (T1-T12)")
        if any(20 <= l <= 24 for l in extra_labels):
            print("    - Labels 20-24: Lumbar vertebrae (L1-L5)")
        if any(l >= 25 for l in extra_labels):
            print("    - Labels 25+: Sacral/other vertebrae")
    else:
        print("\n✓ No extra labels found - dataset appears to be cervical-only!")
    
    print()
    print("="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    if extra_labels:
        print("• Dataset contains non-cervical vertebrae")
        print("• Proceed with cervical cleaning script (filters labels 1-7)")
        print("• Check if you want to require minimum 3 cervical vertebrae")
    else:
        print("• Dataset appears to be cervical-only already")
        print("• You may still want to run cervical cleaning to:")
        print("    - Standardize naming convention")
        print("    - Filter cases with too few cervical vertebrae")
    
    print()
    
    # Detailed mapping suggestion
    print("SUGGESTED LABEL MAPPING:")
    print("-" * 80)
    for label in sorted(cervical_labels):
        print(f"  Label {label} → C{label} (Cervical vertebra {label})")
    
    print()


if __name__ == "__main__":
    # Check the labels directory after running the DICOM converter
    labels_dir = Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\DukeCSS\\labels")
    
    # Check more files if you want a more thorough analysis
    check_label_schema(labels_dir, sample_size=10)