from pathlib import Path
import gzip
import shutil


def is_actually_gzipped(filepath: Path) -> bool:
    """Check if file is really gzipped by reading magic bytes."""
    try:
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    except Exception:
        return False


def fix_and_compress_file(filepath: Path, dry_run: bool = True) -> dict:
    """
    Fix a fake .gz file:
    1. Rename to remove .gz (it's not actually compressed)
    2. Properly compress it to .nii.gz
    """
    result = {
        'original': filepath.name,
        'is_fake_gz': False,
        'action': None,
        'final_name': None,
        'error': None
    }
    
    # Check if it has .gz extension but isn't actually gzipped
    if filepath.suffix == '.gz' and not is_actually_gzipped(filepath):
        result['is_fake_gz'] = True
        
        if dry_run:
            # Determine what the filename should be
            if filepath.stem.endswith('.nii'):
                # Like CVPP_001_0000.nii.gz
                result['action'] = 'rename to .nii → compress to .nii.gz'
                result['final_name'] = filepath.name  # Keep same name but actually compress
            else:
                # Like CVPP_001_0000.gz (missing .nii)
                result['action'] = 'rename to .nii → compress to .nii.gz'
                result['final_name'] = f"{filepath.stem}.nii.gz"
            return result
        
        # Actually fix it
        try:
            # Step 1: Rename to .nii (remove fake .gz)
            if filepath.stem.endswith('.nii'):
                nii_path = filepath.parent / filepath.stem
            else:
                nii_path = filepath.parent / f"{filepath.stem}.nii"
            
            print(f"    Renaming: {filepath.name} → {nii_path.name}")
            filepath.rename(nii_path)
            
            # Step 2: Actually compress it
            gz_path = Path(str(nii_path) + '.gz')
            print(f"    Compressing: {nii_path.name} → {gz_path.name}")
            
            with open(nii_path, 'rb') as f_in:
                with gzip.open(gz_path, 'wb', compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Step 3: Remove uncompressed file
            nii_path.unlink()
            
            result['action'] = 'SUCCESS'
            result['final_name'] = gz_path.name
            
            # Show compression ratio
            original_size = filepath.stat().st_size
            compressed_size = gz_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100
            print(f"    Compressed: {original_size:,} → {compressed_size:,} bytes ({ratio:.1f}% reduction)")
            
        except Exception as e:
            result['error'] = str(e)
            result['action'] = 'FAILED'
        
        return result
    
    elif filepath.suffix == '.gz' and is_actually_gzipped(filepath):
        result['action'] = 'Already properly gzipped'
        result['final_name'] = filepath.name
        return result
    
    else:
        result['action'] = 'Not a .gz file, skipping'
        return result


def process_directory(directory: Path, dry_run: bool = True):
    """Process all .gz files in directory."""
    
    gz_files = sorted(directory.glob("*.gz"))
    
    if not gz_files:
        print(f"  No .gz files found in {directory.name}")
        return
    
    print(f"  Found {len(gz_files)} .gz files")
    
    if dry_run:
        print("\n  DRY RUN - Analyzing files:\n")
        
        fake_count = 0
        real_count = 0
        
        for filepath in gz_files[:15]:  # Show first 15
            result = fix_and_compress_file(filepath, dry_run=True)
            
            if result['is_fake_gz']:
                fake_count += 1
                print(f"    ⚠️  FAKE: {result['original']}")
                print(f"       → {result['action']}")
            elif result['action'] == 'Already properly gzipped':
                real_count += 1
                print(f"    ✓  OK: {result['original']}")
        
        if len(gz_files) > 15:
            print(f"\n    ... and {len(gz_files) - 15} more files")
        
        # Count all files
        total_fake = sum(1 for f in gz_files if not is_actually_gzipped(f))
        total_real = len(gz_files) - total_fake
        
        print(f"\n  Summary:")
        print(f"    Fake .gz files (need fixing): {total_fake}")
        print(f"    Real .gz files (already OK): {total_real}")
        print(f"    Total: {len(gz_files)}")
        
    else:
        print("\n  Processing files...\n")
        
        success_count = 0
        already_ok_count = 0
        error_count = 0
        
        for i, filepath in enumerate(gz_files, 1):
            print(f"  [{i}/{len(gz_files)}] {filepath.name}")
            
            result = fix_and_compress_file(filepath, dry_run=False)
            
            if result['action'] == 'SUCCESS':
                success_count += 1
                print(f"    ✓ Fixed: {result['final_name']}\n")
            elif result['action'] == 'Already properly gzipped':
                already_ok_count += 1
                print(f"    ✓ Already OK\n")
            elif result['action'] == 'FAILED':
                error_count += 1
                print(f"    ✗ Error: {result['error']}\n")
        
        print(f"\n  Results:")
        print(f"    Fixed: {success_count}")
        print(f"    Already OK: {already_ok_count}")
        print(f"    Errors: {error_count}")


def main():
    # ==================== CONFIGURATION ====================
    base_dir = Path(r"C:\Users\anoma\Downloads\surgipath-datasets\v2\cleaned-backup")
    
    # Set to False to actually fix the files
    DRY_RUN = False
    
    # ==================== VALIDATION ====================
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return
    
    print(f"{'='*80}")
    print("FIX FAKE .GZ FILES AND PROPERLY COMPRESS")
    print(f"{'='*80}\n")
    print("Problem: Files have .gz extension but are NOT actually compressed")
    print("Solution: Rename to .nii, then properly gzip compress them\n")
    
    # Process imagesTr
    images_dir = base_dir / "imagesTr"
    if images_dir.exists():
        print(f"Processing imagesTr: {images_dir}")
        print("-" * 80)
        process_directory(images_dir, dry_run=DRY_RUN)
    else:
        print(f"WARNING: imagesTr not found")
    
    # Process labelsTr
    labels_dir = base_dir / "labelsTr"
    if labels_dir.exists():
        print(f"\n\nProcessing labelsTr: {labels_dir}")
        print("-" * 80)
        process_directory(labels_dir, dry_run=DRY_RUN)
    else:
        print(f"WARNING: labelsTr not found")
    
    # Final message
    print(f"\n{'='*80}")
    if DRY_RUN:
        print("⚠️  DRY RUN MODE - No files were modified")
        print("\nWhat will happen:")
        print("  1. Rename fake .gz files to .nii")
        print("  2. Properly gzip compress them")
        print("  3. Result: Actual .nii.gz files (much smaller!)")
        print("\nSet DRY_RUN = False to fix the files")
    else:
        print("✅ PROCESSING COMPLETE")
        print("All files are now properly compressed .nii.gz files")
        print("File sizes should be MUCH smaller now!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()