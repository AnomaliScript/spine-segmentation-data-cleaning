import os
import gzip
import shutil

def compress_nii_files(folder):
    compressed_count = 0
    
    for root, dirs, files in os.walk(folder):
        print(f"\nSearching in: {root}")
        
        for file in files:
            if file.endswith('.nii') and not file.endswith('.nii.gz'):
                input_path = os.path.join(root, file)
                output_path = input_path + '.gz'
                
                print(f"  Compressing: {file}")
                
                with open(input_path, 'rb') as f_in:
                    with gzip.open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                os.remove(input_path)
                compressed_count += 1
                print(f"  ✓ Created: {file}.gz")
            else:
                print(f"  Skipping: {file}")
    
    print(f"\n{'='*60}")
    print(f"Compression completed: {compressed_count} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    folder_path = "C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\VerSe_clean_v3\\labels"
    
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
    else:
        compress_nii_files(folder_path)