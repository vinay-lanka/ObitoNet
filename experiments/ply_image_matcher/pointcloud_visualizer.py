import open3d as o3d
import numpy as np

# Provide the location of the .ply file
ply_file_path = "Dataset/Barn_data/Barn_ply/Barn01.ply"

# Load the .ply file
point_cloud = o3d.io.read_point_cloud(ply_file_path)

print(type(point_cloud))

print("Sample points:", np.asarray(point_cloud.points)[:5])

print(point_cloud)
print("Loaded pointcloud")

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud],
                                  window_name="Point Cloud Viewer",
                                  width=1920,
                                  height=1800,
                                  left=50,
                                  top=50,
                                  point_show_normal=True)
