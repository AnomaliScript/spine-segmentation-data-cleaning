from datasets import load_dataset

# Load 3D volumetric data
dataset_3d = load_dataset(
    'alexanderdann/CTSpine1K',
    name="3d",
    trust_remote_code=True,
    writer_batch_size=1, # see the warning above
)