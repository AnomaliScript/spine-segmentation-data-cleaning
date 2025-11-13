from pathlib import Path

def explore_duke_structure(base_path: Path, case_limit: int = 3):
    """
    Explore the Duke dataset structure to understand where DICOMs actually are.
    """
    base_path = Path(base_path)
    imaging_dir = base_path / "DukeCSpineSeg_imaging_files" / "case_image"
    
    print("="*80)
    print("DUKE DATASET STRUCTURE EXPLORER")
    print("="*80)
    print(f"Exploring: {imaging_dir}")
    print()
    
    if not imaging_dir.exists():
        print(f"ERROR: Directory not found: {imaging_dir}")
        return
    
    # Get case directories
    case_dirs = sorted([d for d in imaging_dir.iterdir() if d.is_dir()])[:case_limit]
    
    print(f"Examining first {len(case_dirs)} cases...\n")
    
    for case_dir in case_dirs:
        print("="*80)
        print(f"CASE: {case_dir.name}")
        print("="*80)
        
        # Level 1: Direct children
        level1_items = list(case_dir.iterdir())
        print(f"\nLevel 1 ({len(level1_items)} items):")
        for item in level1_items:
            item_type = "DIR" if item.is_dir() else "FILE"
            size = f"{item.stat().st_size:,} bytes" if item.is_file() else ""
            print(f"  [{item_type}] {item.name} {size}")
        
        # Level 2: Explore subdirectories
        subdirs = [d for d in level1_items if d.is_dir()]
        for subdir in subdirs:
            level2_items = list(subdir.iterdir())
            print(f"\n  Level 2 in '{subdir.name}' ({len(level2_items)} items):")
            for item in level2_items:
                item_type = "DIR" if item.is_dir() else "FILE"
                size = f"{item.stat().st_size:,} bytes" if item.is_file() else ""
                ext = item.suffix if item.is_file() else ""
                print(f"    [{item_type}] {item.name} {ext} {size}")
            
            # Level 3: Go deeper if directories
            level2_subdirs = [d for d in level2_items if d.is_dir()]
            for subdir2 in level2_subdirs:
                level3_items = list(subdir2.iterdir())
                print(f"\n      Level 3 in '{subdir2.name}' ({len(level3_items)} items):")
                for item in level3_items[:10]:  # Limit to first 10
                    item_type = "DIR" if item.is_dir() else "FILE"
                    size = f"{item.stat().st_size:,} bytes" if item.is_file() else ""
                    ext = item.suffix if item.is_file() else ""
                    print(f"        [{item_type}] {item.name} {ext} {size}")
                
                if len(level3_items) > 10:
                    print(f"        ... and {len(level3_items) - 10} more items")
                
                # Check what file types are present
                if level3_items:
                    file_types = {}
                    for item in level3_items:
                        if item.is_file():
                            ext = item.suffix if item.suffix else "no_extension"
                            file_types[ext] = file_types.get(ext, 0) + 1
                    
                    if file_types:
                        print(f"\n      File type summary:")
                        for ext, count in sorted(file_types.items()):
                            print(f"        {ext}: {count} files")
        
        print("\n")
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print("Please check the output above to see:")
    print("  1. Are there .dcm files? If so, at which level?")
    print("  2. Are there .zip files? If so, at which level?")
    print("  3. Are there files with no extension? (might be DICOMs)")
    print("  4. What's the actual structure?")
    print()


if __name__ == "__main__":
    base_path = Path(r"C:\Users\anoma\Downloads\spine-segmentation-data-cleaning\DukeCSS")
    explore_duke_structure(base_path, case_limit=3)