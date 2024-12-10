import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using fps to make Nx3 to npointx3(usually 8192x3) 
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def main():
    k_value = 50
    path = "./dataset/OldDataset/Truck_is/Truck/"
    save_path = "./dataset/Dataset/pcd/"
    print("Processing directory", path)
    print("Output directory", save_path)
    print("Every ", k_value,"th points are selected, could use voxel or uniform downsampling")

    pcd_files = [pcd_file for pcd_file in os.listdir(path) if '.ply' in pcd_file]
    for pcd_file in pcd_files:
        pcd = o3d.io.read_point_cloud(path + pcd_file)
        print("PCD Details Original Size:")
        print(pcd)

        down_pcd = pcd.uniform_down_sample(every_k_points=k_value)
        # np_pcd = np.asarray(down_pcd.points)
        # print("Downsampled via uniform downsampling", np_pcd.shape)
        
        
        # fps_pcd = farthest_point_sample(np_pcd , 16384)
        fps_pcd = down_pcd.farthest_point_down_sample(16384)
        np_fps_pcd = np.asarray(fps_pcd.points)
        print("Downsampled via fps downsampling", np_fps_pcd.shape)
        np.save(save_path + pcd_file[:-3] + "npy", np_fps_pcd)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np_fps_pcd)
        # o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()