# Just check a few files
from pathlib import Path

def check_file(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(2)
        if magic == b'\x1f\x8b':
            print(f"✓ {filepath.name} - PROPERLY GZIPPED")
        else:
            print(f"✗ {filepath.name} - NOT GZIPPED")

base = Path(r"C:\\Users\\anoma\\Downloads\\surgipath-datasets\\v2\\cleaned-backup\\imagesTr")
for f in sorted(base.glob("*.gz"))[:5]:
    check_file(f)