from pathlib import Path
from typing import Optional
import numpy as np
import SimpleITK as sitk
import json
import pandas as pd
import pydicom
from rt_utils import RTStructBuilder


def retrieve_dataset(dataset_dir: str, metadata_path: str) -> Optional[list[Path]]:
    """
    Retrieves STUDY paths from metadata.csv, filtering for cervical-relevant studies.
    Returns list of study directory paths.
    """
    try:
        metadata_df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error reading metadata.csv: {e}")
        return None
    
    required_cols = ['Study Description', 'File Location']
    for c in required_cols:
        if c not in metadata_df.columns:
            print(f"metadata.csv missing required column: {c}")
            return None
    
    # Filter for cervical-relevant studies
    cervical_keywords = ['clavicle', 'neck', 'cranio']
    exclude_keywords = ['lumbar', 'pelvis', 'abdomen', 'rib', 'chest', 'thoracic', 'brain', 'head']
    sd = metadata_df['Study Description'].fillna('').str.lower()
    relevant_mask = sd.str.contains('|'.join(cervical_keywords), na=False) & \
                    ~sd.str.contains('|'.join(exclude_keywords), na=False)
    
    relevant_df = metadata_df[relevant_mask]
    print(f"Found {len(relevant_df)} cervical-relevant series across multiple studies.")
    
    if relevant_df.empty:
        print("No cervical-relevant studies found.")
        return None
    
    # Get unique STUDY paths
    data_root = Path(dataset_dir)
    study_paths = set()
    
    for _, row in relevant_df.iterrows():
        loc = Path(str(row['File Location']))
        if loc.is_absolute():
            study_path = loc.parent
        else:
            full_series_path = (data_root / loc).resolve()
            study_path = full_series_path.parent if full_series_path.exists() else None
        
        if study_path and study_path.is_dir():
            study_paths.add(study_path)
    
    study_paths = sorted(study_paths)
    
    if not study_paths:
        print("No valid study directories found.")
        return None
    
    print("\nAvailable cervical-relevant studies:")
    for i, path in enumerate(study_paths):
        print(f"  [{i}] {path.name}")
    
    choice = input("\nWhich study to process? (enter number, or 'all' for batch): ").strip()
    
    try:
        if choice.lower() == 'all':
            return list(study_paths)
        idx = int(choice)
        return [study_paths[idx]]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return None


def dicom_directory(study_dir: Path) -> dict[str, dict]:
    """
    Returns { series_uid: {'modality': str, 'series_dir': Path, 'files': [paths...]} }
    Scans entire study folder and categorizes all series by modality.
    """
    index = {}
    
    for path in study_dir.rglob("*.dcm"):
        if not path.is_file():
            continue
        
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            series_uid = getattr(ds, "SeriesInstanceUID", None)
            modality = getattr(ds, "Modality", None)
            
            if not (series_uid and modality):
                continue
            
            if series_uid not in index:
                index[series_uid] = {
                    'modality': modality,
                    'series_dir': path.parent,
                    'files': []
                }
            
            index[series_uid]['files'].append(str(path))
            
        except Exception:
            continue
    
    return index


def convert_ct_to_nifti(series_dir: Path) -> tuple[sitk.Image, dict]:
    """
    Reads CT DICOM series with proper slice ordering and returns SimpleITK image + metadata.
    """
    reader = sitk.ImageSeriesReader()
    
    # Use GDCM to get properly ordered file list
    series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
    
    if not series_ids:
        raise ValueError(f"No DICOM series found in {series_dir}")
    
    # Use the first series (should only be one per folder)
    series_id = series_ids[0]
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir), series_id)
    
    if not dicom_names:
        raise ValueError(f"No DICOM files found for series {series_id}")
    
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    try:
        img = reader.Execute()
    except Exception as e:
        raise ValueError(f"Failed to read CT series: {e}")
    
    # Extract metadata
    meta = {
        "StudyInstanceUID": reader.GetMetaData(0, "0020|000d").strip() if "0020|000d" in reader.GetMetaDataKeys(0) else "NA",
        "SeriesInstanceUID": reader.GetMetaData(0, "0020|000e").strip() if "0020|000e" in reader.GetMetaDataKeys(0) else "NA",
        "SeriesDescription": reader.GetMetaData(0, "0008|103e").strip() if "0008|103e" in reader.GetMetaDataKeys(0) else "NA",
        "FrameOfReferenceUID": reader.GetMetaData(0, "0020|0052").strip() if "0020|0052" in reader.GetMetaDataKeys(0) else "NA",
        "Modality": "CT",
        "NumSlices": img.GetSize()[2],
        "Spacing": list(img.GetSpacing()),
        "Origin": list(img.GetOrigin()),
        "Direction": list(img.GetDirection()),
    }
    
    return img, meta


def convert_seg_to_nifti(seg_files: list[str], reference_ct: sitk.Image) -> Optional[sitk.Image]:
    """
    Reads DICOM SEG file and resamples to match CT geometry.
    Handles single multi-frame SEG files.
    """
    if len(seg_files) != 1:
        print(f"  Warning: SEG should be single file, found {len(seg_files)}. Using first.")
    
    seg_path = seg_files[0]
    
    try:
        # Read SEG as single file
        seg_img = sitk.ReadImage(seg_path)
        
        # Resample to match CT geometry
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_ct)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        
        seg_resampled = resampler.Execute(seg_img)
        
        return seg_resampled
        
    except Exception as e:
        print(f"  Failed to read SEG: {e}")
        return None


def convert_rtstruct_to_nifti(rtstruct_files: list[str], series_dir: Path, reference_ct: sitk.Image) -> Optional[sitk.Image]:
    """
    Reads RTSTRUCT and rasterizes contours to match CT geometry using rt-utils.
    """
    if len(rtstruct_files) != 1:
        print(f"  Warning: RTSTRUCT should be single file, found {len(rtstruct_files)}. Using first.")
    
    rtstruct_path = rtstruct_files[0]
    
    try:
        # Load RTSTRUCT with rt-utils
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(series_dir),
            rt_struct_path=rtstruct_path
        )
        
        # Get all ROI names
        roi_names = rtstruct.get_roi_names()
        
        if not roi_names:
            print("  No ROIs found in RTSTRUCT")
            return None
        
        print(f"  Found ROIs: {roi_names}")
        
        # Combine all ROIs into single mask
        # Get mask for first ROI as template
        mask_np = rtstruct.get_roi_mask_by_name(roi_names[0])
        
        # Add other ROIs
        for roi_name in roi_names[1:]:
            mask_np = np.logical_or(mask_np, rtstruct.get_roi_mask_by_name(roi_name))
        
        # Convert to SimpleITK image with CT geometry
        mask_img = sitk.GetImageFromArray(mask_np.astype(np.uint8))
        mask_img.CopyInformation(reference_ct)
        
        return mask_img
        
    except Exception as e:
        print(f"  Failed to convert RTSTRUCT: {e}")
        return None


def resample_to_reference(moving_img: sitk.Image, reference_img: sitk.Image) -> sitk.Image:
    """
    Resamples moving image to match reference image geometry.
    Uses nearest neighbor for label images.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(moving_img.GetPixelID())
    
    return resampler.Execute(moving_img)


def create_nnunet_dataset_json(out_dir: Path, num_cases: int):
    """
    Creates dataset.json for nnUNetv2 with proper format.
    """
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "0": "background",
            "1": "spine"
        },
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
    }
    
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)


def main():
    dataset_dir = "C:\\Users\\anoma\\Downloads\\surgipath-datasets\\SpineMETSCTSEG"
    metadata_path = "C:\\Users\\anoma\\Downloads\\surgipath-datasets\\SpineMETSCTSEG\\metadata.csv"
    
    study_paths = retrieve_dataset(dataset_dir, metadata_path)
    
    if study_paths is None:
        print("No studies selected.")
        return
    
    # Create output structure
    out_dir = Path("out")
    images_dir = out_dir / "imagesTr"
    labels_dir = out_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for study_dir in study_paths:
        print(f"\n{'='*60}")
        print(f"Processing study: {study_dir.name}")
        print(f"{'='*60}")
        
        # Extract patient ID from path structure
        patient_id = study_dir.parent.name
        
        # Get all series in this study
        series_index = dicom_directory(study_dir)
        
        if not series_index:
            print(f"  No DICOM series found in {study_dir}")
            continue
        
        # Separate CT and segmentation series
        ct_series = {uid: data for uid, data in series_index.items() if data['modality'] == 'CT'}
        seg_series = {uid: data for uid, data in series_index.items() if data['modality'] == 'SEG'}
        rtstruct_series = {uid: data for uid, data in series_index.items() if data['modality'] == 'RTSTRUCT'}
        
        print(f"  Found {len(ct_series)} CT, {len(seg_series)} SEG, {len(rtstruct_series)} RTSTRUCT series")
        
        if not ct_series:
            print("  No CT series found. Skipping study.")
            continue
        
        if len(ct_series) > 1:
            print(f"  WARNING: Found {len(ct_series)} CT series, expected 1. Using first.")
        
        # Get the single CT series
        ct_uid, ct_data = next(iter(ct_series.items()))
        print(f"  CT series: {len(ct_data['files'])} slices")
        
        try:
            ct_img, ct_meta = convert_ct_to_nifti(ct_data['series_dir'])
        except Exception as e:
            print(f"  Failed to convert CT series: {e}")
            continue
        
        if ct_meta["NumSlices"] < 10:
            print(f"  Too few slices ({ct_meta['NumSlices']}). Skipping.")
            continue
        
        # Save CT volume with patient ID
        ct_filename = f"{patient_id}_0000.nii.gz"
        sitk.WriteImage(ct_img, str(images_dir / ct_filename))
        print(f"  Saved CT: {ct_filename}")
        
        # Find matching segmentation
        mask_img = None
        
        # Try SEG first
        if seg_series:
            if len(seg_series) > 1:
                print(f"  WARNING: Found {len(seg_series)} SEG series, expected 1. Using first.")
            
            _, seg_data = next(iter(seg_series.items()))
            print(f"  Found SEG series with {len(seg_data['files'])} file(s)")
            mask_img = convert_seg_to_nifti(seg_data['files'], ct_img)
            
            if mask_img is not None and mask_img.GetSize() != ct_img.GetSize():
                print(f"  Resampling SEG to match CT geometry")
                mask_img = resample_to_reference(mask_img, ct_img)
        
        # If no SEG, try RTSTRUCT
        if mask_img is None and rtstruct_series:
            if len(rtstruct_series) > 1:
                print(f"  WARNING: Found {len(rtstruct_series)} RTSTRUCT series, expected 1. Using first.")
            
            _, rt_data = next(iter(rtstruct_series.items()))
            print(f"  Found RTSTRUCT series with {len(rt_data['files'])} file(s)")
            mask_img = convert_rtstruct_to_nifti(
                rt_data['files'],
                ct_data['series_dir'],
                ct_img
            )
        
        # Save mask
        if mask_img is not None:
            seg_filename = f"{patient_id}.nii.gz"
            sitk.WriteImage(mask_img, str(labels_dir / seg_filename))
            print(f"  Saved segmentation: {seg_filename}")
            processed_count += 1
            print(f"  SUCCESS: Case {patient_id} complete")
        else:
            print(f"  WARNING: No segmentation found for {patient_id}")
    
    # Create dataset.json
    if processed_count > 0:
        create_nnunet_dataset_json(out_dir, processed_count)
        print(f"\n{'='*60}")
        print(f"Dataset complete: {processed_count} cases processed")
        print(f"Output directory: {out_dir.absolute()}")
        print(f"{'='*60}")
    else:
        print("\nNo cases successfully processed.")


if __name__ == "__main__":
    main()