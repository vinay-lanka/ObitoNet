import pyzed.sl as sl
import numpy as np
import os
from datetime import datetime
import time

def save_pointcloud(format='ply'):
    """
    Capture and save a pointcloud from ZED camera
    Args:
        format (str): Output format - 'ply' or 'pcd'
    """
    # Initialize ZED camera
    zed = sl.Camera()
    
    # Create configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening camera: {err}")
        return

    # Create runtime parameters
    runtime_parameters = sl.RuntimeParameters()
    
    # Create point cloud object
    point_cloud = sl.Mat()
    
    print("Initializing camera...")
    # Wait for camera to initialize and auto-adjust
    time.sleep(2)
    
    try:
        # Grab a new frame
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            # Get timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == 'ply':
                filename = f"pointcloud_{timestamp}.ply"
                save_as_ply(point_cloud, filename)
                print(f"Pointcloud saved as {filename}")
            elif format.lower() == 'pcd':
                filename = f"pointcloud_{timestamp}.pcd"
                save_as_pcd(point_cloud, filename)
                print(f"Pointcloud saved as {filename}")
            else:
                print(f"Unsupported format: {format}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
        
    finally:
        zed.close()

def save_as_ply(point_cloud, filename):
    """Save point cloud as PLY file"""
    # Get data as numpy array
    pc_data = point_cloud.get_data()
    print(f"Point cloud shape: {pc_data.shape}")
    
    # Get dimensions
    height, width, channels = pc_data.shape
    
    # Extract XYZ and RGB (first 3 channels for XYZ, next 4 channels for RGBA)
    xyz = pc_data[:, :, :3].reshape(height * width, 3)
    # Convert RGBA to RGB by taking first 3 channels
    rgb = pc_data[:, :, 3:6].reshape(height * width, 3).astype(np.uint8)
    
    # Remove invalid points (infinite or NaN values)
    valid_mask = np.logical_and(
        np.all(np.isfinite(xyz), axis=1),
        ~np.all(xyz == 0, axis=1)  # Remove zero points
    )
    xyz = xyz[valid_mask]
    rgb = rgb[valid_mask]
    
    # Number of valid points
    num_points = len(xyz)
    print(f"Saving {num_points} valid points")
    
    # Write PLY file
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write point data
        for i in range(num_points):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

def save_as_pcd(point_cloud, filename):
    """Save point cloud as PCD file"""
    # Get data as numpy array
    pc_data = point_cloud.get_data()
    print(f"Point cloud shape: {pc_data.shape}")
    
    # Get dimensions
    height, width, channels = pc_data.shape
    
    # Extract XYZ and RGB
    xyz = pc_data[:, :, :3].reshape(height * width, 3)
    # Convert RGBA to RGB by taking first 3 channels
    rgb = pc_data[:, :, 3:6].reshape(height * width, 3).astype(np.uint8)
    
    # Remove invalid points
    valid_mask = np.logical_and(
        np.all(np.isfinite(xyz), axis=1),
        ~np.all(xyz == 0, axis=1)  # Remove zero points
    )
    xyz = xyz[valid_mask]
    rgb = rgb[valid_mask]
    
    # Number of valid points
    num_points = len(xyz)
    print(f"Saving {num_points} valid points")
    
    # Write PCD file
    with open(filename, 'w') as f:
        # Write header
        f.write("# .PCD v0.7 - Point Cloud Data\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {num_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {num_points}\n")
        f.write("DATA ascii\n")
        
        # Write point data
        for i in range(num_points):
            x, y, z = xyz[i]
            # Pack RGB into a single float
            r, g, b = rgb[i]
            rgb_packed = r << 16 | g << 8 | b
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb_packed}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Save ZED camera pointcloud')
    parser.add_argument('--format', type=str, default='ply', choices=['ply', 'pcd'],
                        help='Output format (ply or pcd)')
    args = parser.parse_args()
    
    save_pointcloud(format=args.format)