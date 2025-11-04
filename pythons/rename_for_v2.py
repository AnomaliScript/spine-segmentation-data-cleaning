from pathlib import Path
import re


def extract_case_number(filename: str) -> int:
    """
    Extract a numeric identifier from various filename patterns.
    Tries to find any number in the filename to use as case number.
    """
    # Remove extension and _0000 suffix
    name = filename.replace('_0000', '').replace('.nii.gz', '').replace('.nii', '')
    
    # Try to find any numbers in the filename
    numbers = re.findall(r'\d+', name)
    
    if numbers:
        # Use the first substantial number found (prefer longer numbers)
        numbers_by_length = sorted(numbers, key=len, reverse=True)
        return int(numbers_by_length[0])
    
    # Fallback: use hash of filename if no numbers found
    return abs(hash(name)) % 10000


def get_file_pairs(images_dir: Path, labels_dir: Path) -> list[dict]:
    """
    Match image files with their corresponding label files.
    Returns list of {'image': Path, 'label': Path, 'base_name': str}
    """
    # Get all label files (they don't have _0000)
    label_files = sorted(labels_dir.glob("*.nii.gz")) + sorted(labels_dir.glob("*.nii"))
    
    pairs = []
    unmatched_images = []
    
    for label_file in label_files:
        # Find corresponding image file
        # Try with _0000 suffix first
        image_name_with_suffix = label_file.stem.replace('.nii', '') + '_0000' + label_file.suffix
        image_file = images_dir / image_name_with_suffix
        
        if not image_file.exists():
            # Try without .gz if it has it
            if label_file.suffix == '.gz':
                image_name_with_suffix = label_file.stem.replace('.nii', '') + '_0000.nii.gz'
                image_file = images_dir / image_name_with_suffix
        
        if image_file.exists():
            pairs.append({
                'image': image_file,
                'label': label_file,
                'base_name': label_file.stem.replace('.nii', '')
            })
        else:
            print(f"  WARNING: No matching image found for label: {label_file.name}")
    
    # Check for orphaned images
    all_image_files = set(images_dir.glob("*_0000.nii.gz")) | set(images_dir.glob("*_0000.nii"))
    matched_images = {pair['image'] for pair in pairs}
    orphaned = all_image_files - matched_images
    
    if orphaned:
        print(f"\n  WARNING: Found {len(orphaned)} orphaned image files (no matching label):")
        for img in sorted(orphaned)[:5]:  # Show first 5
            print(f"    - {img.name}")
        if len(orphaned) > 5:
            print(f"    ... and {len(orphaned) - 5} more")
    
    return pairs


def generate_new_names(pairs: list[dict]) -> list[dict]:
    """
    Generate CVPP_XXX format names for all pairs.
    Returns list of {'old_image': Path, 'old_label': Path, 'new_base': str}
    """
    # Sort pairs by extracted case number to maintain some ordering
    sorted_pairs = sorted(pairs, key=lambda p: extract_case_number(p['base_name']))
    
    rename_map = []
    for idx, pair in enumerate(sorted_pairs, start=1):
        new_base = f"CVPP_{idx:03d}"  # CVPP_001, CVPP_002, etc.
        rename_map.append({
            'old_image': pair['image'],
            'old_label': pair['label'],
            'new_base': new_base,
            'old_base': pair['base_name']
        })
    
    return rename_map


def preview_renames(rename_map: list[dict]) -> None:
    """Show preview of what will be renamed."""
    print(f"\n{'='*80}")
    print("PREVIEW OF RENAMES")
    print(f"{'='*80}\n")
    
    print(f"Total pairs to rename: {len(rename_map)}\n")
    
    # Show first 10 and last 5
    preview_count = min(10, len(rename_map))
    
    print("First few renames:")
    for item in rename_map[:preview_count]:
        print(f"\n  {item['old_base']}")
        print(f"    Image: {item['old_image'].name}")
        print(f"      →  {item['new_base']}_0000{item['old_image'].suffix}")
        print(f"    Label: {item['old_label'].name}")
        print(f"      →  {item['new_base']}{item['old_label'].suffix}")
    
    if len(rename_map) > preview_count:
        print(f"\n  ... {len(rename_map) - preview_count} more pairs ...")
        
        if len(rename_map) > preview_count + 5:
            print("\nLast few renames:")
            for item in rename_map[-5:]:
                print(f"\n  {item['old_base']}")
                print(f"    Image: {item['old_image'].name} → {item['new_base']}_0000{item['old_image'].suffix}")
                print(f"    Label: {item['old_label'].name} → {item['new_base']}{item['old_label'].suffix}")


def rename_files(rename_map: list[dict], dry_run: bool = True) -> None:
    """
    Perform the actual renaming.
    
    Args:
        rename_map: List of rename operations
        dry_run: If True, only simulate (don't actually rename)
    """
    if dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN - No files will be renamed")
        print(f"{'='*80}\n")
        return
    
    print(f"\n{'='*80}")
    print("RENAMING FILES...")
    print(f"{'='*80}\n")
    
    success_count = 0
    error_count = 0
    
    for item in rename_map:
        try:
            # Rename image file
            new_image_name = f"{item['new_base']}_0000{item['old_image'].suffix}"
            new_image_path = item['old_image'].parent / new_image_name
            item['old_image'].rename(new_image_path)
            
            # Rename label file
            new_label_name = f"{item['new_base']}{item['old_label'].suffix}"
            new_label_path = item['old_label'].parent / new_label_name
            item['old_label'].rename(new_label_path)
            
            success_count += 1
            
            if success_count % 10 == 0:
                print(f"  Renamed {success_count}/{len(rename_map)} pairs...")
                
        except Exception as e:
            error_count += 1
            print(f"  ERROR renaming {item['old_base']}: {e}")
    
    print(f"\n{'='*80}")
    print(f"COMPLETE: {success_count} pairs renamed successfully")
    if error_count > 0:
        print(f"ERRORS: {error_count} pairs failed")
    print(f"{'='*80}")


def main():
    # ==================== CONFIGURATION ====================
    base_dir = Path(r"C:\\Users\\anoma\\Downloads\\surgipath-datasets\\v2\\cleaned-backup")
    images_dir = base_dir / "imagesTr"
    labels_dir = base_dir / "labelsTr"
    
    # Set to False to actually perform the rename
    DRY_RUN = False
    
    # ==================== VALIDATION ====================
    if not base_dir.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        return
    
    if not images_dir.exists():
        print(f"ERROR: imagesTr directory not found: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"ERROR: labelsTr directory not found: {labels_dir}")
        return
    
    # ==================== PROCESSING ====================
    print(f"{'='*80}")
    print("STANDARDIZING FILENAMES TO CVPP_XXX FORMAT")
    print(f"{'='*80}\n")
    print(f"Base directory: {base_dir}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}\n")
    
    # Step 1: Find all matching pairs
    print("Step 1: Finding image-label pairs...")
    pairs = get_file_pairs(images_dir, labels_dir)
    
    if not pairs:
        print("\nERROR: No matching image-label pairs found!")
        return
    
    print(f"  Found {len(pairs)} matching pairs\n")
    
    # Step 2: Generate new names
    print("Step 2: Generating CVPP_XXX names...")
    rename_map = generate_new_names(pairs)
    print(f"  Generated {len(rename_map)} new names\n")
    
    # Step 3: Preview
    preview_renames(rename_map)
    
    # Step 4: Rename (or dry run)
    rename_files(rename_map, dry_run=DRY_RUN)
    
    if DRY_RUN:
        print("\n" + "="*80)
        print("DRY RUN MODE - No files were actually renamed")
        print("Set DRY_RUN = False in the code to perform actual renaming")
        print("="*80)


if __name__ == "__main__":
    main()