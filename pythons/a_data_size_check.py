import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_file_size_distribution(folder_path: Path, file_pattern: str = "*", remove_outliers: bool = True):
    """
    Create visualizations of file size distribution in a folder.
    
    Args:
        folder_path: Path to the folder to analyze
        file_pattern: Glob pattern for files (e.g., "*.nii.gz", "*.dcm", "*")
        remove_outliers: If True, remove outliers beyond 2 std deviations for visualization
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"ERROR: Folder not found: {folder_path}")
        return
    
    # Get all files matching pattern
    files = list(folder_path.glob(file_pattern))
    
    if not files:
        print(f"ERROR: No files found matching pattern '{file_pattern}' in {folder_path}")
        return
    
    print(f"Analyzing {len(files)} files in: {folder_path}")
    print(f"Pattern: {file_pattern}")
    print()
    
    # Get file sizes in KB and MB
    file_sizes_bytes = [f.stat().st_size for f in files]
    file_sizes_kb = [size / 1024 for size in file_sizes_bytes]
    file_sizes_mb = [size / (1024 * 1024) for size in file_sizes_bytes]
    
    # Calculate statistics on ALL data
    mean_kb = np.mean(file_sizes_kb)
    median_kb = np.median(file_sizes_kb)
    min_kb = np.min(file_sizes_kb)
    max_kb = np.max(file_sizes_kb)
    std_kb = np.std(file_sizes_kb)
    
    print("FILE SIZE STATISTICS (ALL FILES):")
    print("-" * 60)
    print(f"Total files: {len(files)}")
    print(f"Mean: {mean_kb:.2f} KB ({mean_kb/1024:.2f} MB)")
    print(f"Median: {median_kb:.2f} KB ({median_kb/1024:.2f} MB)")
    print(f"Min: {min_kb:.2f} KB ({min_kb/1024:.2f} MB)")
    print(f"Max: {max_kb:.2f} KB ({max_kb/1024:.2f} MB)")
    print(f"Std Dev: {std_kb:.2f} KB")
    print()
    
    # Identify outliers
    lower_bound = mean_kb - 2 * std_kb
    upper_bound = mean_kb + 2 * std_kb
    
    outliers_high = [(f, size_kb) for f, size_kb in zip(files, file_sizes_kb) if size_kb > upper_bound]
    outliers_low = [(f, size_kb) for f, size_kb in zip(files, file_sizes_kb) if size_kb < lower_bound]
    
    # Print outliers
    if outliers_high:
        print(f"HIGH OUTLIERS (> 2 std dev, > {upper_bound:.2f} KB): {len(outliers_high)} files")
        for file, size_kb in sorted(outliers_high, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {file.name}: {size_kb:.2f} KB ({size_kb/1024:.2f} MB)")
        if len(outliers_high) > 5:
            print(f"  ... and {len(outliers_high) - 5} more")
        print()
    
    if outliers_low:
        print(f"LOW OUTLIERS (< 2 std dev, < {lower_bound:.2f} KB): {len(outliers_low)} files")
        for file, size_kb in sorted(outliers_low, key=lambda x: x[1])[:5]:
            print(f"  {file.name}: {size_kb:.2f} KB ({size_kb/1024:.2f} MB)")
        if len(outliers_low) > 5:
            print(f"  ... and {len(outliers_low) - 5} more")
        print()
    
    # Filter data for plotting if remove_outliers is True
    if remove_outliers and (outliers_high or outliers_low):
        filtered_sizes_kb = [size for size in file_sizes_kb if lower_bound <= size <= upper_bound]
        filtered_sizes_mb = [size / 1024 for size in filtered_sizes_kb]
        
        print(f"PLOTTING WITH OUTLIERS REMOVED:")
        print(f"  Using {len(filtered_sizes_kb)}/{len(files)} files for visualization")
        print(f"  Excluded {len(files) - len(filtered_sizes_kb)} outliers")
        print()
        
        plot_sizes_kb = filtered_sizes_kb
        plot_sizes_mb = filtered_sizes_mb
        plot_title_suffix = " (Outliers Removed)"
    else:
        plot_sizes_kb = file_sizes_kb
        plot_sizes_mb = file_sizes_mb
        plot_title_suffix = ""
    
    # Determine best unit (KB or MB) based on file sizes
    use_mb = mean_kb > 1024
    sizes = plot_sizes_mb if use_mb else plot_sizes_kb
    unit = "MB" if use_mb else "KB"
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'File Size Distribution: {folder_path.name}{plot_title_suffix}', fontsize=16, fontweight='bold')
    
    # ========== HISTOGRAM ==========
    ax1.hist(sizes, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.2f} {unit}')
    ax1.axvline(np.median(sizes), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(sizes):.2f} {unit}')
    
    ax1.set_xlabel(f'File Size ({unit})', fontsize=12)
    ax1.set_ylabel('Frequency (Number of Files)', fontsize=12)
    ax1.set_title('Histogram: File Size Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== SORTED LINE PLOT ==========
    sorted_sizes = sorted(sizes)
    ax2.plot(range(len(sorted_sizes)), sorted_sizes, linewidth=2, color='coral')
    ax2.axhline(np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.2f} {unit}')
    ax2.axhline(np.median(sizes), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(sizes):.2f} {unit}')
    
    ax2.set_xlabel('File Index (sorted by size)', fontsize=12)
    ax2.set_ylabel(f'File Size ({unit})', fontsize=12)
    ax2.set_title('Line Plot: Files Sorted by Size', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    suffix = "_no_outliers" if remove_outliers and (outliers_high or outliers_low) else ""
    output_file = folder_path.parent / f"{folder_path.name}_file_size_distribution{suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # Show the plot
    plt.show()


def main():
    """
    Main function with example usage.
    """
    import sys
    
    # Configuration
    if len(sys.argv) >= 2:
        folder_path = Path(sys.argv[1])
        file_pattern = sys.argv[2] if len(sys.argv) >= 3 else "*"
        remove_outliers = sys.argv[3].lower() != "false" if len(sys.argv) >= 4 else True
    else:
        # Default configuration - change these
        folder_path = Path(r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\v3\\labelsTr")
        file_pattern = "*.nii.gz"  # Can be "*.nii.gz", "*.dcm", "*.nii", "*", etc.
        remove_outliers = True  # Set to False to include outliers in plots
    
    plot_file_size_distribution(folder_path, file_pattern, remove_outliers)


if __name__ == "__main__":
    main()