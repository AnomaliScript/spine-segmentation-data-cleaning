import dicom2nifti
import dicom2nifti.settings as settings
from pathlib import Path
import shutil
import tempfile
import zipfile

def extract_dicoms_from_nested_structure(case_dir: Path, temp_extract_dir: Path) -> Path:
    """
    Navigate Duke's nested folder structure and extract DICOM files from zip archives.
    Returns path to temp directory containing extracted DICOMs.
    
    Duke structure: case_dir/long-name-folder/zipfile.zip
    """
    extracted_count = 0
    
    # Look for subdirectories with long names (e.g., 1.2.826.0.1.3680043...)
    subdirs = [d for d in case_dir.iterdir() if d.is_dir()]
    print(f"    Found {len(subdirs)} subdirectories")
    
    for subdir in subdirs:
        # Look for zip files directly in this subdirectory
        zip_files = list(subdir.glob("*.zip"))
        
        if zip_files:
            print(f"    Found {len(zip_files)} zip file(s) in {subdir.name}")
            
            for zip_file in zip_files:
                try:
                    # Extract to temp directory
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        # List what's in the zip
                        file_list = zip_ref.namelist()
                        print(f"      Extracting {zip_file.name} ({len(file_list)} files inside)")
                        
                        zip_ref.extractall(temp_extract_dir)
                        extracted_count += len(file_list)
                    
                    print(f"      ✓ Extracted: {zip_file.name}")
                    
                except Exception as e:
                    print(f"      ✗ Warning: Could not extract {zip_file.name}: {e}")
    
    if extracted_count == 0:
        raise Exception("No files extracted from zip archives")
    
    print(f"    Total files extracted: {extracted_count}")
    
    # Check for DICOM files - they might not have .dcm extension!
    # Try multiple patterns
    dicom_files = []
    dicom_files.extend(list(temp_extract_dir.rglob("*.dcm")))
    dicom_files.extend(list(temp_extract_dir.rglob("*.DCM")))
    dicom_files.extend(list(temp_extract_dir.rglob("*.dicom")))
    dicom_files.extend(list(temp_extract_dir.rglob("*.DICOM")))
    
    # Also check for files with no extension (common for DICOMs)
    all_files = list(temp_extract_dir.rglob("*"))
    no_ext_files = [f for f in all_files if f.is_file() and not f.suffix]
    
    if no_ext_files:
        print(f"    Found {len(no_ext_files)} files with no extension (likely DICOMs)")
        dicom_files.extend(no_ext_files)
    
    if not dicom_files:
        # List what we actually extracted
        print(f"    ERROR: No DICOM files found. Files extracted:")
        for f in all_files[:10]:
            print(f"      - {f.name} ({f.suffix if f.suffix else 'no extension'})")
        raise Exception("No DICOM files found after extraction")
    
    print(f"    ✓ Found {len(dicom_files)} DICOM files")
    return temp_extract_dir


def process_duke_dicoms_proper(base_path: Path):
    """
    Process Duke dataset using the proper dicom2nifti library:
    1. Find all case directories with DICOM files
    2. Extract DICOMs from nested zip structure
    3. Convert DICOMs to NIfTI using dicom2nifti library
    4. Match with segmentation files
    5. Organize into /volumes and /labels folders
    """
    base_path = Path(base_path)
    imaging_dir = base_path / "DukeCSpineSeg_imaging_files" / "case_image"
    seg_dir = base_path / "DukeCSpineSeg_segmentation"
    
    # Create output directories
    volumes_dir = base_path / "volumes"
    labels_dir = base_path / "labels"
    
    volumes_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("DUKE DATASET: DICOM TO NIFTI CONVERSION (using dicom2nifti library)")
    print("="*80)
    print(f"Base directory: {base_path}")
    print(f"Imaging source: {imaging_dir}")
    print(f"Segmentation source: {seg_dir}")
    print(f"Output volumes: {volumes_dir}")
    print(f"Output labels: {labels_dir}")
    print()
    
    # Check directories exist
    if not imaging_dir.exists():
        print(f"ERROR: Imaging directory not found: {imaging_dir}")
        return
    
    if not seg_dir.exists():
        print(f"ERROR: Segmentation directory not found: {seg_dir}")
        return
    
    # Configure dicom2nifti settings
    # Allow single slices (Duke might have some)
    settings.disable_validate_slicecount()
    
    # Get all case directories
    case_dirs = sorted([d for d in imaging_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(case_dirs)} case directories")
    print()
    
    # Get all segmentation files
    seg_files = sorted(seg_dir.glob("*.nii.gz"))
    seg_dict = {}
    for seg_file in seg_files:
        # Extract case ID (e.g., "593973-000001")
        case_id = seg_file.name.split('_')[0]
        seg_dict[case_id] = seg_file
    
    print(f"Found {len(seg_files)} segmentation files")
    print()
    
    # Check how many are already processed
    already_done = 0
    for case_dir in case_dirs:
        case_id = case_dir.name
        if case_id not in seg_dict:
            continue
        vol_output = volumes_dir / f"{case_id}_0000.nii.gz"
        label_output = labels_dir / f"{case_id}.nii.gz"
        if vol_output.exists() and label_output.exists():
            already_done += 1
    
    remaining = len([d for d in case_dirs if d.name in seg_dict]) - already_done
    
    print("="*80)
    print("RESUME STATUS:")
    print(f"  Total cases: {len([d for d in case_dirs if d.name in seg_dict])}")
    print(f"  Already processed: {already_done}")
    print(f"  Remaining: {remaining}")
    print("="*80)
    print()
    
    if remaining == 0:
        print("All cases already processed! Nothing to do.")
        return
    
    print(f"Resuming processing for {remaining} remaining cases...")
    print()
    
    # Process each case
    success_count = 0
    error_count = 0
    skipped_count = 0  # Already processed
    case_number = 0  # Track which case we're on
    
    for case_dir in case_dirs:
        case_id = case_dir.name
        
        # Check if segmentation exists
        if case_id not in seg_dict:
            continue
        
        case_number += 1
        
        print(f"Processing case {case_number}/{len([d for d in case_dirs if d.name in seg_dict])}: {case_id}")
        
        seg_file = seg_dict[case_id]
        
        # Output paths - ADD _0000 to volume to match universal convention
        vol_output = volumes_dir / f"{case_id}_0000.nii.gz"  # <-- Added _0000
        label_output = labels_dir / f"{case_id}.nii.gz"
        
        # Skip if already processed
        if vol_output.exists() and label_output.exists():
            print(f"  ✓ Already processed, skipping")
            success_count += 1
            print()
            continue
        
        # Also skip if only volume exists (partial processing)
        if vol_output.exists() and not label_output.exists():
            print(f"  ⚠ Partial processing detected - volume exists, copying label only")
            try:
                shutil.copy2(seg_file, label_output)
                print(f"  ✓ Segmentation copied: {label_output.name}")
                success_count += 1
                print()
                continue
            except Exception as e:
                print(f"  ✗ ERROR copying label: {e}")
                error_count += 1
                print()
                continue
        
        # Create temporary directory for this case
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Step 1: Extract DICOMs from nested zip structure
            print(f"  Extracting DICOMs from nested structure...")
            dicom_dir = extract_dicoms_from_nested_structure(case_dir, Path(temp_dir))
            
            # Step 2: Convert using dicom2nifti library
            print(f"  Converting DICOMs to NIfTI...")
            
            # dicom2nifti.dicom_series_to_nifti returns a dictionary with:
            # - 'NII_FILE': path to created nifti file
            # - 'NIFTI': the nibabel image object
            # - 'BVAL_FILE': path to bval file (for DTI, if present)
            # - 'BVEC_FILE': path to bvec file (for DTI, if present)
            
            result = dicom2nifti.dicom_series_to_nifti(
                str(dicom_dir),
                str(vol_output),
                reorient_nifti=True  # Standardize to LAS orientation
            )
            
            if result and 'NII_FILE' in result:
                print(f"  ✓ Volume converted: {vol_output.name}")
                
                # Step 3: Copy segmentation file
                shutil.copy2(seg_file, label_output)
                print(f"  ✓ Segmentation copied: {label_output.name}")
                
                success_count += 1
            else:
                print(f"  ✗ ERROR: Conversion returned no output")
                error_count += 1
            
            print()
            
        except Exception as e:
            print(f"  ✗ ERROR processing {case_id}: {e}")
            error_count += 1
            print()
        
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
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
    print("  1. Run duke_2_check_schema.py to verify label conventions")
    print("  2. Run duke_3_cervical_clean.py to filter and rename files")
    print()


if __name__ == "__main__":
    base_path = Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\DukeCSS")
    process_duke_dicoms_proper(base_path)