from huggingface_hub import snapshot_download

from huggingface_hub import login
login("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")

# Download entire repository 
local_dir = snapshot_download(
    repo_id="alexanderdann/CTSpine1K",
    repo_type="dataset",
    cache_dir="C:\\Users\\anoma\\Downloads\\surgipath-datasets\\CTSpine1K"  # optional
)

# Now you can access files directly:
# local_dir/rawdata/volumes/[dataset]/*.nii.gz
# local_dir/rawdata/labels/[dataset]/*_seg.nii.gz