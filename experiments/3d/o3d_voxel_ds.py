import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("Load a ply point cloud, print it, and render it")
    path = "../dataset/Dataset/Barn_is/Barn/Barn01.ply"
    pcd = o3d.io.read_point_cloud(path)
    print(pcd)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.5)
    voxels = voxel_grid.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    colors = np.stack(list(vx.color for vx in voxels))
    print(indices[0:10])
    print(voxel_grid)
    o3d.visualization.draw_geometries([voxel_grid])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                        voxel_size=0.2)
    voxels = voxel_grid.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    colors = np.stack(list(vx.color for vx in voxels))
    print(indices[0:10])
    print(voxel_grid)
    o3d.visualization.draw_geometries([voxel_grid])

