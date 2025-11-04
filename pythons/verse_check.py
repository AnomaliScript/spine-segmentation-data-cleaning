# save as check_labels.py
from pathlib import Path
import numpy as np
import nibabel as nib
import sys

# Hardcode your file here or pass it as the first CLI arg
DEFAULT_PATH = r"C:\\Users\\anoma\\Downloads\\surgipath-datasets\\VerSe\\segmentations\\sub-verse835\\verse835_CT-iso_seg.nii"

def load_labels(seg_path: Path):
    img = nib.load(str(seg_path))        # works for .nii or .nii.gz
    arr = np.asanyarray(img.dataobj)     # lazy-ish read; will materialize on unique()
    # Coerce floats to integers safely (common in some toolchains)
    if np.issubdtype(arr.dtype, np.floating):
        # Round to nearest int; if values aren't near-integers, we'll warn below
        arr_int = np.rint(arr).astype(np.int32, copy=False)
        if not np.allclose(arr, arr_int, atol=1e-3):
            print("WARNING: segmentation contains non-integer values; rounded to nearest integers.")
        arr = arr_int
    elif not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32, copy=False)

    vals, counts = np.unique(arr, return_counts=True)
    vals = vals.astype(int)
    return vals, counts

def classify_labels(vals: np.ndarray):
    vals_set = set(map(int, vals.tolist()))
    if vals_set.issubset({0, 1}):
        return "BINARY (background=0, vertebrae=1)"
    # Many VerSe masks use 0 for background and 1..N for vertebrae
    if 0 in vals_set and max(vals_set) > 1:
        return f"MULTI-CLASS vertebrae (IDs {sorted(v for v in vals_set if v != 0)})"
    # Fallback for unusual encodings
    return f"NON-STANDARD labels: {sorted(vals_set)}"

def main():
    seg_path_str = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    seg_path = Path(seg_path_str)
    if not seg_path.exists():
        print(f"ERROR: file not found: {seg_path}")
        sys.exit(1)

    try:
        vals, counts = load_labels(seg_path)
    except Exception as e:
        print(f"ERROR reading NIfTI: {e}")
        sys.exit(1)

    verdict = classify_labels(vals)

    total_vox = int(counts.sum())
    print(f"\nFile: {seg_path}")
    print(f"Total voxels: {total_vox}")
    print(f"Unique label IDs ({len(vals)}): {vals.tolist()}")
    print(f"Verdict: {verdict}\n")

    # Nice formatted table
    print("LabelID\tVoxels\tPercent")
    for v, c in zip(vals, counts):
        pct = (c / total_vox) * 100 if total_vox else 0.0
        print(f"{int(v):>6}\t{int(c):>6}\t{pct:6.3f}%")

if __name__ == "__main__":
    main()