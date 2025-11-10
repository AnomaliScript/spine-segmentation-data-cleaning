#!/usr/bin/env python3
"""
Cervical Spine Surgical Path Planning System
Uses Fast Marching Method (FMM) for optimal path finding

This is a reference implementation for educational purposes.
Modify and adapt as needed for your project.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import nibabel as nib
from scipy.ndimage import distance_transform_edt
try:
    import skfmm
except ImportError:
    print("ERROR: skfmm not installed. Run: pip install scikit-fmm")
    exit(1)

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    print("WARNING: PyVista not installed. 3D visualization disabled.")
    print("Install with: pip install pyvista")
    PYVISTA_AVAILABLE = False
    
# ============================================================================
# CONFIGURATION
# ============================================================================

SEGMENTATION_FILES = [
    r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\CTSpine1K\\clean_labels\\CTS1K_007.nii.gz"
]

# Safety margin in voxels (conservative estimate for unseen vessels/nerves)
SAFETY_MARGIN_MM = 5.0

# Use dummy data for testing (set to False when you have real data)
USE_DUMMY_DATA = False

# Enable 3D visualization with PyVista (opens after path is computed)
ENABLE_3D_VISUALIZATION = True


# ============================================================================
# CORE PATH PLANNING FUNCTIONS
# ============================================================================

def create_dummy_cervical_segmentation(shape=(256, 256, 100)):
    """
    Generate synthetic cervical vertebrae segmentation for testing
    Creates realistic-looking vertebral bodies C1-C7
    """
    print("Generating dummy cervical vertebrae segmentation...")
    seg = np.zeros(shape, dtype=np.uint8)
    
    # Create 7 vertebrae (C1-C7) along z-axis
    center_x, center_y = shape[0] // 2, shape[1] // 2
    
    for i in range(7):
        z_center = 15 + i * 12  # Space vertebrae along z-axis
        
        # Create vertebral body (roughly rectangular with canal)
        for z in range(z_center - 4, z_center + 4):
            if z < 0 or z >= shape[2]:
                continue
                
            # Vertebral body (outer rectangle)
            seg[center_x-20:center_x+20, center_y-25:center_y+25, z] = i + 1
            
            # Spinal canal (hollow center)
            seg[center_x-8:center_x+8, center_y-8:center_y+8, z] = 0
            
            # Add some lateral processes
            seg[center_x-28:center_x-20, center_y-10:center_y+10, z] = i + 1
            seg[center_x+20:center_x+28, center_y-10:center_y+10, z] = i + 1
    
    print(f"Created dummy segmentation: {shape}, 7 vertebrae")
    return seg


def load_segmentation(filepath=None):
    """
    Load vertebrae segmentation from NIfTI file or generate dummy data
    
    Args:
        filepath: Path to .nii or .nii.gz file
        
    Returns:
        segmentation: 3D numpy array
        affine: Affine transformation matrix (for real data)
    """
    if USE_DUMMY_DATA or filepath is None:
        seg = create_dummy_cervical_segmentation()
        affine = np.eye(4)  # Identity matrix for dummy data
        return seg, affine
    
    try:
        print(f"Loading segmentation from: {filepath}")
        nifti_img = nib.load(filepath)
        segmentation = nifti_img.get_fdata()
        affine = nifti_img.affine
        print(f"Loaded: shape={segmentation.shape}, dtype={segmentation.dtype}")
        return segmentation, affine
    except FileNotFoundError:
        print(f"WARNING: File not found: {filepath}")
        print("Using dummy data instead...")
        seg = create_dummy_cervical_segmentation()
        affine = np.eye(4)
        return seg, affine


def compute_distance_transform(segmentation):
    # Calculate distance from each voxel to nearest bone
    # Outputs distance_map: 3D array with distances in voxels
 
    print("Computing distance transform from bone surfaces...")
    
    # Create binary mask (any non-zero value = bone)
    bone_mask = (segmentation > 0).astype(np.uint8)
    
    # Calculate distance to nearest bone voxel
    distance_map = distance_transform_edt(bone_mask == 0)
    
    print(f"Distance range: {distance_map.min():.2f} to {distance_map.max():.2f} voxels")
    return distance_map


def create_speed_map(distance_map, safety_margin=5.0):
    # Convert distance map to speed map for FMM
    # Speed = how fast the wavefront can travel (high speed = safe)
    # distance_map: Distance to nearest bone in voxels (3D pixels)
    # safety_margin: Minimum safe distance in voxels
    # Output: speed_map: Values > 0 (higher = safer/faster travel)
    
    print(f"Creating speed map with {safety_margin} voxel safety margin...")
    
    # Speed increases with distance from bone
    # Add safety margin to avoid division by zero and ensure safe paths
    speed_map = distance_map + safety_margin
    
    # Normalize to reasonable range (0.1 to 1.0)
    speed_map = speed_map / speed_map.max()
    speed_map = np.clip(speed_map, 0.1, 1.0)  # Minimum speed to avoid zero
    
    return speed_map


def plan_path_fmm(speed_map, start_point, end_point):
    """
    Use Fast Marching Method to find optimal path
    
    Args:
        speed_map: 3D array of travel speeds (higher = better)
        start_point: (x, y, z) tuple for entry point
        end_point: (x, y, z) tuple for target point
        
    Returns:
        path: Nx3 array of (x,y,z) coordinates
        travel_time: Total "travel time" (lower = better path)
    """
    print(f"Planning path from {start_point} to {end_point}...")
    
    # Create phi: distance field (negative at start, positive elsewhere)
    phi = np.ones_like(speed_map)
    phi[start_point] = -1  # Start point marked as negative
    
    # Run Fast Marching to compute travel time from start to all points
    try:
        travel_time = skfmm.travel_time(phi, speed_map)
    except Exception as e:
        print(f"ERROR in FMM: {e}")
        return None, None
    
    # Trace path from end back to start using gradient descent
    path = [end_point]
    current = np.array(end_point, dtype=float)
    
    max_iterations = 10000
    step_size = 0.5
    
    for iteration in range(max_iterations):
        # Calculate gradient of travel time (points toward start)
        grad = np.array(np.gradient(travel_time))
        
        # Get gradient at current position (with bounds checking)
        pos = tuple(np.clip(current.astype(int), 0, 
                           [s-1 for s in speed_map.shape]))
        gradient_vec = np.array([grad[i][pos] for i in range(3)])
        
        # Move opposite to gradient (downhill toward start)
        if np.linalg.norm(gradient_vec) < 1e-6:
            break
            
        current = current - step_size * gradient_vec / np.linalg.norm(gradient_vec)
        
        # Check bounds
        if np.any(current < 0) or np.any(current >= speed_map.shape):
            break
            
        path.append(tuple(current.astype(int)))
        
        # Check if we reached start
        if np.linalg.norm(current - np.array(start_point)) < 2.0:
            break
    
    path = np.array(path)
    total_time = travel_time[end_point]
    
    print(f"Path found: {len(path)} points, travel time: {total_time:.2f}")
    return path, total_time


def calculate_safety_metrics(path, distance_map):
    """
    Calculate safety metrics for the planned path
    
    Args:
        path: Nx3 array of path coordinates
        distance_map: Distance to nearest bone
        
    Returns:
        metrics: Dictionary with safety statistics
    """
    # Get distance values along path
    distances = []
    for point in path:
        p = tuple(point.astype(int))
        if all(0 <= p[i] < distance_map.shape[i] for i in range(3)):
            distances.append(distance_map[p])
    
    distances = np.array(distances)
    
    metrics = {
        'min_clearance': float(np.min(distances)),
        'max_clearance': float(np.max(distances)),
        'avg_clearance': float(np.mean(distances)),
        'path_length': len(path),
        'safe': np.min(distances) >= 3.0  # 3 voxel minimum
    }
    
    return metrics


# ============================================================================
# 3D VISUALIZATION WITH PYVISTA
# ============================================================================

def create_vertebrae_mesh(segmentation, threshold=0.5):
    """
    Convert segmentation volume to 3D surface mesh
    
    Args:
        segmentation: 3D numpy array
        threshold: Value to use for surface extraction
        
    Returns:
        PyVista mesh object
    """
    if not PYVISTA_AVAILABLE:
        return None
        
    # Create PyVista uniform grid from numpy array
    grid = pv.wrap(segmentation)
    
    # Extract surface mesh using marching cubes
    mesh = grid.contour([threshold], scalars=segmentation.ravel(order='F'))
    
    return mesh


def visualize_path_3d(segmentation, path, start_point, end_point, metrics):
    """
    Create interactive 3D visualization of vertebrae and surgical path
    
    Args:
        segmentation: 3D vertebrae segmentation
        path: Nx3 array of path coordinates
        start_point: (x,y,z) start coordinates
        end_point: (x,y,z) end coordinates
        metrics: Dictionary of safety metrics
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Skipping 3D visualization.")
        return
    
    if not ENABLE_3D_VISUALIZATION:
        return
        
    print("\nOpening 3D visualization...")
    
    # Create plotter
    plotter = pv.Plotter()
    plotter.set_background('black')
    
    # Add vertebrae mesh
    print("Creating vertebrae mesh...")
    vertebrae_mesh = create_vertebrae_mesh(segmentation, threshold=0.5)
    
    if vertebrae_mesh is not None and vertebrae_mesh.n_points > 0:
        plotter.add_mesh(
            vertebrae_mesh,
            color='lightgray',
            opacity=0.4,
            smooth_shading=True,
            label='Vertebrae'
        )
    
    # Add path as tube
    if path is not None and len(path) > 1:
        print("Adding path...")
        path_polydata = pv.PolyData(path)
        
        # Create line connecting path points
        lines = np.full((len(path)-1, 3), 2, dtype=np.int_)
        lines[:, 1] = np.arange(len(path)-1)
        lines[:, 2] = np.arange(1, len(path))
        path_polydata.lines = lines
        
        # Add as tube for better visibility
        tube = path_polydata.tube(radius=0.8)
        plotter.add_mesh(
            tube,
            color='blue',
            label='Surgical Path'
        )
    
    # Add start point
    if start_point is not None:
        start_sphere = pv.Sphere(radius=2.0, center=start_point)
        plotter.add_mesh(
            start_sphere,
            color='red',
            label='Start Point'
        )
        plotter.add_point_labels(
            [start_point],
            ['START'],
            font_size=20,
            text_color='red',
            point_color='red',
            point_size=10
        )
    
    # Add end point
    if end_point is not None:
        end_sphere = pv.Sphere(radius=2.0, center=end_point)
        plotter.add_mesh(
            end_sphere,
            color='green',
            label='Target Point'
        )
        plotter.add_point_labels(
            [end_point],
            ['TARGET'],
            font_size=20,
            text_color='green',
            point_color='green',
            point_size=10
        )
    
    # Add text with metrics
    if metrics:
        status = "SAFE" if metrics['safe'] else "WARNING"
        metrics_text = (
            f"Path Planning Results\n"
            f"Status: {status}\n"
            f"Min Clearance: {metrics['min_clearance']:.2f} voxels\n"
            f"Avg Clearance: {metrics['avg_clearance']:.2f} voxels\n"
            f"Path Length: {metrics['path_length']} points"
        )
        plotter.add_text(
            metrics_text,
            position='upper_right',
            font_size=10,
            color='white'
        )
    
    # Add title
    plotter.add_text(
        "Cervical Spine Surgical Path Planning - 3D View",
        position='upper_left',
        font_size=14,
        color='white',
        font='arial'
    )
    
    # Add legend
    plotter.add_legend(size=(0.2, 0.2), loc='lower_left')
    
    # Add axes
    plotter.add_axes()
    
    # Show orientation widget
    plotter.add_camera_orientation_widget()
    
    # Set camera position for good initial view
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.2)
    
    # Show the plot
    print("3D visualization window opened. Rotate with mouse, zoom with scroll.")
    print("Close the window to continue...")
    plotter.show()


# ============================================================================
# INTERACTIVE VISUALIZATION
# ============================================================================

class InteractivePathPlanner:
    """
    Interactive matplotlib interface for path planning
    Click to select start and end points, see path drawn in real-time
    """
    
    def __init__(self, segmentation, distance_map, speed_map):
        self.segmentation = segmentation
        self.distance_map = distance_map
        self.speed_map = speed_map
        
        # Selected points
        self.start_point = None
        self.end_point = None
        
        # Computed path and metrics (for 3D visualization)
        self.computed_path = None
        self.computed_metrics = None
        
        # Current slice to display
        self.current_slice = segmentation.shape[2] // 2
        
        # Create figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5))
        self.fig.suptitle('Cervical Spine Surgical Path Planning', 
                         fontsize=14, fontweight='bold')
        
        self.setup_display()
        
    def setup_display(self):
        """Initialize the visualization"""
        
        # Axial view (main interaction)
        self.ax_axial = self.axes[0]
        self.ax_axial.set_title(f'Axial View (Slice {self.current_slice})\n'
                                'Click: RED=start, GREEN=end')
        
        # Sagittal view
        self.ax_sagittal = self.axes[1]
        self.ax_sagittal.set_title('Sagittal View')
        
        # Metrics display
        self.ax_metrics = self.axes[2]
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Safety Metrics')
        
        self.update_display()
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def update_display(self):
        """Redraw all views"""
        
        # Axial view
        self.ax_axial.clear()
        axial_slice = self.segmentation[:, :, self.current_slice]
        self.ax_axial.imshow(axial_slice.T, cmap='gray', origin='lower')
        self.ax_axial.set_title(f'Axial View (Slice {self.current_slice})\n'
                                'Click: RED=start, GREEN=end | Arrow Keys: change slice')
        
        # Draw points if selected
        if self.start_point:
            circle = Circle((self.start_point[0], self.start_point[1]), 
                          3, color='red', linewidth=2, fill=False)
            self.ax_axial.add_patch(circle)
            self.ax_axial.plot(self.start_point[0], self.start_point[1], 
                             'r*', markersize=15)
            
        if self.end_point:
            circle = Circle((self.end_point[0], self.end_point[1]), 
                          3, color='green', linewidth=2, fill=False)
            self.ax_axial.add_patch(circle)
            self.ax_axial.plot(self.end_point[0], self.end_point[1], 
                             'g*', markersize=15)
        
        # Sagittal view
        self.ax_sagittal.clear()
        mid_x = self.segmentation.shape[0] // 2
        sagittal_slice = self.segmentation[mid_x, :, :]
        self.ax_sagittal.imshow(sagittal_slice.T, cmap='gray', origin='lower')
        self.ax_sagittal.set_title('Sagittal View')
        
        plt.draw()
        
    def on_key(self, event):
        """Handle keyboard input for slice navigation"""
        if event.key == 'up':
            self.current_slice = min(self.current_slice + 1, 
                                    self.segmentation.shape[2] - 1)
            self.update_display()
        elif event.key == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
            self.update_display()
            
    def on_click(self, event):
        """Handle mouse clicks to select start/end points"""
        
        # Only process clicks on axial view
        if event.inaxes != self.ax_axial:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        z = self.current_slice
        
        # Check bounds
        if not (0 <= x < self.segmentation.shape[0] and 
                0 <= y < self.segmentation.shape[1]):
            return
        
        # Set start point (first click)
        if self.start_point is None:
            self.start_point = (x, y, z)
            print(f"Start point set: {self.start_point}")
            self.update_display()
            
        # Set end point and compute path (second click)
        elif self.end_point is None:
            self.end_point = (x, y, z)
            print(f"End point set: {self.end_point}")
            self.compute_and_display_path()
            
        # Reset (third click)
        else:
            print("Resetting points...")
            self.start_point = None
            self.end_point = None
            self.update_display()
            
    def compute_and_display_path(self):
        """Run FMM and visualize the result"""
        
        print("\n" + "="*60)
        print("COMPUTING PATH...")
        print("="*60)
        
        # Compute path
        path, travel_time = plan_path_fmm(
            self.speed_map, 
            self.start_point, 
            self.end_point
        )
        
        if path is None:
            print("ERROR: Path planning failed!")
            self.ax_metrics.clear()
            self.ax_metrics.axis('off')
            self.ax_metrics.text(0.1, 0.5, "PATH PLANNING FAILED", 
                               fontsize=14, color='red', weight='bold')
            plt.draw()
            return
        
        # Calculate metrics
        metrics = calculate_safety_metrics(path, self.distance_map)
        
        # Display path on axial view
        path_2d = path[path[:, 2] == self.current_slice]
        if len(path_2d) > 0:
            self.ax_axial.plot(path_2d[:, 0], path_2d[:, 1], 
                             'b-', linewidth=2, label='Path')
        
        # Display path on sagittal view
        mid_x = self.segmentation.shape[0] // 2
        path_sag = path[np.abs(path[:, 0] - mid_x) < 5]  # Within 5 voxels
        if len(path_sag) > 0:
            self.ax_sagittal.plot(path_sag[:, 1], path_sag[:, 2], 
                                'b-', linewidth=2, label='Path')
        
        # Display metrics
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        
        status = "SAFE ✓" if metrics['safe'] else "WARNING ⚠"
        status_color = 'green' if metrics['safe'] else 'orange'
        
        metrics_text = f"""
PATH FOUND!

Status: {status}

Safety Metrics:
━━━━━━━━━━━━━━━━━━━━
Min Clearance:  {metrics['min_clearance']:.2f} voxels
Max Clearance:  {metrics['max_clearance']:.2f} voxels
Avg Clearance:  {metrics['avg_clearance']:.2f} voxels

Path Length:    {metrics['path_length']} points
Travel Time:    {travel_time:.2f}

Recommendation:
{self._get_recommendation(metrics)}

[3D View Available]
Close matplotlib to see 3D
        """
        
        self.ax_metrics.text(0.05, 0.95, metrics_text, 
                           fontsize=10, family='monospace',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add status indicator
        self.ax_metrics.text(0.5, 0.05, status,
                           fontsize=16, weight='bold', color=status_color,
                           ha='center', va='bottom')
        
        plt.draw()
        
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        for key, value in metrics.items():
            print(f"{key:20s}: {value}")
        print("="*60 + "\n")
        
        # Store path and metrics for 3D visualization
        self.computed_path = path
        self.computed_metrics = metrics
        
    def _get_recommendation(self, metrics):
        """Generate clinical recommendation based on metrics"""
        if metrics['min_clearance'] < 3.0:
            return "⚠ Path too close to bone\nConsider alternative approach"
        elif metrics['min_clearance'] < 5.0:
            return "⚠ Acceptable but use caution\nConsider imaging guidance"
        else:
            return "✓ Path is safe\nGood surgical corridor"
    
    def show(self):
        """Display the interactive interface"""
        print("\n" + "="*60)
        print("INTERACTIVE PATH PLANNER")
        print("="*60)
        print("Instructions:")
        print("  1. Click on image to set START point (red)")
        print("  2. Click again to set END point (green)")
        print("  3. Path will be computed automatically")
        print("  4. Click a third time to reset")
        print("  5. Use UP/DOWN arrow keys to change slice")
        if PYVISTA_AVAILABLE and ENABLE_3D_VISUALIZATION:
            print("  6. Close matplotlib window to see 3D visualization")
        print("="*60 + "\n")
        
        plt.tight_layout()
        plt.show()
        
        # After matplotlib closes, show 3D visualization if path was computed
        if (PYVISTA_AVAILABLE and ENABLE_3D_VISUALIZATION and 
            self.computed_path is not None):
            print("\nMatplotlib closed. Opening 3D visualization...")
            visualize_path_3d(
                self.segmentation,
                self.computed_path,
                self.start_point,
                self.end_point,
                self.computed_metrics
            )


# ============================================================================
# BATCH PROCESSING (for multiple files)
# ============================================================================

def process_single_file(filepath):
    """
    Process a single segmentation file
    Can be called in a loop for batch processing
    """
    print(f"\nProcessing: {filepath}")
    
    # Load data
    segmentation, affine = load_segmentation(filepath)
    
    # Compute distance and speed maps
    distance_map = compute_distance_transform(segmentation)
    speed_map = create_speed_map(distance_map, safety_margin=SAFETY_MARGIN_MM)
    
    # Launch interactive planner
    planner = InteractivePathPlanner(segmentation, distance_map, speed_map)
    planner.show()


def process_multiple_files(filepaths):
    """
    Process multiple segmentation files sequentially
    """
    print(f"\nBatch processing {len(filepaths)} files...")
    
    for i, filepath in enumerate(filepaths, 1):
        print(f"\n{'='*60}")
        print(f"File {i}/{len(filepaths)}")
        print(f"{'='*60}")
        process_single_file(filepath)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    
    print("\n" + "="*60)
    print("CERVICAL SPINE SURGICAL PATH PLANNING SYSTEM")
    print("Fast Marching Method (FMM)")
    print("="*60 + "\n")
    
    # Process files based on configuration
    if len(SEGMENTATION_FILES) == 1:
        process_single_file(SEGMENTATION_FILES[0])
    else:
        process_multiple_files(SEGMENTATION_FILES)


if __name__ == "__main__":
    main()