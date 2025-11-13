import pandas as pd
from pathlib import Path
import subprocess
import tempfile
import shutil
import zipfile

def extract_dicoms_from_case(case_dir: Path, temp_dir: Path) -> bool:
    """
    Extract DICOMs from Duke's nested zip structure.
    Returns True if extraction successful.
    """
    subdirs = [d for d in case_dir.iterdir() if d.is_dir()]
    
    extracted_any = False
    for subdir in subdirs:
        zip_files = list(subdir.glob("*.zip"))
        
        for zip_file in zip_files:
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                extracted_any = True
            except Exception as e:
                print(f"      Warning: Could not extract {zip_file.name}: {e}")
    
    return extracted_any


def convert_duke_with_dcm2niix(base_path: Path):
    """
    Convert Duke dataset using dcm2niix CLI tool.
    Extracts from zips, converts, and renames sequentially.
    """
    base_path = Path(base_path)
    imaging_dir = base_path / "DukeCSpineSeg_imaging_files" / "case_image"
    seg_dir = base_path / "DukeCSpineSeg_segmentation"
    
    # Create output directories
    volumes_dir = base_path / "volumes_2"
    labels_dir = base_path / "labels_2"
    
    volumes_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("DUKE DATASET: DICOM TO NIFTI CONVERSION (using dcm2niix)")
    print("="*80)
    print(f"Base directory: {base_path}")
    print(f"Imaging source: {imaging_dir}")
    print(f"Segmentation source: {seg_dir}")
    print(f"Output volumes: {volumes_dir}")
    print(f"Output labels: {labels_dir}")
    print()
    
    # Validate directories
    if not imaging_dir.exists():
        print(f"ERROR: Imaging directory not found: {imaging_dir}")
        return
    
    if not seg_dir.exists():
        print(f"ERROR: Segmentation directory not found: {seg_dir}")
        return
    
    # Check dcm2niix is installed
    try:
        result = subprocess.run(['dcm2niix', '-v'], capture_output=True, text=True)
        print(f"Using dcm2niix: {result.stdout.split()[0] if result.stdout else 'installed'}")
    except FileNotFoundError:
        print("ERROR: dcm2niix not found. Please install it:")
        print("  Windows: Download from https://github.com/rordenlab/dcm2niix/releases")
        print("  Linux: sudo apt install dcm2niix")
        print("  Mac: brew install dcm2niix")
        return
    print()
    
    # Get all case directories
    case_dirs = sorted([d for d in imaging_dir.iterdir() if d.is_dir()])
    
    # Get all segmentation files
    seg_files = sorted(seg_dir.glob("*.nii.gz"))
    seg_dict = {}
    for seg_file in seg_files:
        case_id = seg_file.name.split('_')[0]
        seg_dict[case_id] = seg_file
    
    print(f"Found {len(case_dirs)} case directories")
    print(f"Found {len(seg_files)} segmentation files")
    print()
    
    # Check resume status
    already_done = sum(1 for d in case_dirs 
                      if d.name in seg_dict 
                      and (volumes_dir / f"{d.name}_0000.nii.gz").exists()
                      and (labels_dir / f"{d.name}.nii.gz").exists())
    
    remaining = len([d for d in case_dirs if d.name in seg_dict]) - already_done
    
    print("="*80)
    print("RESUME STATUS:")
    print(f"  Total cases: {len([d for d in case_dirs if d.name in seg_dict])}")
    print(f"  Already processed: {already_done}")
    print(f"  Remaining: {remaining}")
    print("="*80)
    print()
    
    if remaining == 0:
        print("All cases already processed!")
        return
    
    # Process each case
    success_count = 0
    error_count = 0
    skipped_count = 0
    case_number = 0
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        
        if case_id not in seg_dict:
            continue
        
        case_number += 1
        
        print(f"[{case_number}/{len([d for d in case_dirs if d.name in seg_dict])}] Processing: {case_id}")
        
        seg_file = seg_dict[case_id]
        
        # Output paths with _0000 suffix for volumes
        vol_output = volumes_dir / f"{case_id}_0000.nii.gz"
        label_output = labels_dir / f"{case_id}.nii.gz"
        
        # Skip if already processed
        if vol_output.exists() and label_output.exists():
            print(f"  ✓ Already processed, skipping")
            skipped_count += 1
            print()
            continue
        
        # Handle partial processing
        if vol_output.exists() and not label_output.exists():
            try:
                shutil.copy2(seg_file, label_output)
                print(f"  ✓ Partial: copied segmentation only")
                skipped_count += 1
                print()
                continue
            except Exception as e:
                print(f"  ✗ ERROR copying label: {e}")
                error_count += 1
                print()
                continue
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract DICOMs
            print(f"  Extracting DICOMs...")
            if not extract_dicoms_from_case(case_dir, Path(temp_dir)):
                raise Exception("No DICOMs extracted")
            
            # Convert using dcm2niix
            print(f"  Converting with dcm2niix...")
            
            # dcm2niix options:
            # -b n: don't create BIDS sidecar
            # -z y: compress to .nii.gz
            # -f %i: filename pattern (use instance UID - we'll rename after)
            # -o: output directory
            # -s y: single file mode (merge 2D slices)
            
            cmd = [
                'dcm2niix',
                '-b', 'n',           # No BIDS sidecar
                '-z', 'y',           # Compress
                '-f', '%p_%s',       # Patient_Series naming
                '-o', str(volumes_dir),  # Output to volumes dir
                '-s', 'y',           # Single file
                str(temp_dir)        # Input directory
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"  ✗ dcm2niix failed: {result.stderr}")
                raise Exception("dcm2niix conversion failed")
            
            # dcm2niix creates files with various names - find the newest .nii.gz
            nifti_files = sorted(volumes_dir.glob("*.nii.gz"), key=lambda x: x.stat().st_mtime)
            
            if not nifti_files:
                raise Exception("No NIfTI file created by dcm2niix")
            
            # Get the most recently created file
            newest_nifti = nifti_files[-1]
            
            # Rename to our standard naming
            if newest_nifti != vol_output:
                shutil.move(str(newest_nifti), str(vol_output))
            
            print(f"  ✓ Volume converted: {vol_output.name}")
            
            # Copy segmentation
            shutil.copy2(seg_file, label_output)
            print(f"  ✓ Segmentation copied: {label_output.name}")
            
            success_count += 1
            print()
            
        except subprocess.TimeoutExpired:
            print(f"  ✗ ERROR: dcm2niix timeout")
            error_count += 1
            print()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            error_count += 1
            print()
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
            # Cleanup any intermediate files dcm2niix might have created
            for f in volumes_dir.glob("*_*_*.nii.gz"):
                if f != vol_output:
                    try:
                        f.unlink()
                    except:
                        pass
    
    # Summary
    print("="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"Successfully processed: {success_count} cases")
    print(f"Already completed (skipped): {skipped_count} cases")
    print(f"Errors: {error_count} cases")
    print(f"Total completed: {success_count + skipped_count} cases")
    print()
    print(f"Output locations:")
    print(f"  Volumes: {volumes_dir.absolute()}")
    print(f"  Labels:  {labels_dir.absolute()}")
    print()
    print("NEXT STEPS:")
    print("  Run universal_cervical_cleaner.py with dataset_prefix='DCSS'")
    print()


if __name__ == "__main__":
    base_path = Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\DukeCSS")
    convert_duke_with_dcm2niix(base_path)