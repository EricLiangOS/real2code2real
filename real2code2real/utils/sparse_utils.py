import open3d as o3d
import numpy as np
import copy
import os
from . import generate_utils

def get_voxels(ply_path, grid_size=64, padding=0):
    pcd = o3d.io.read_point_cloud(ply_path)

    obb = pcd.get_oriented_bounding_box()

    # Compute the rotation matrix to align the OBB with the axes
    R = obb.R.T  # Transpose of the rotation matrix to align with the axes

    # Rotate the mesh to align with the axes
    pcd.rotate(R, center=np.zeros(3))

    points = np.asarray(pcd.points)
    
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    point_cloud_size = max_bound - min_bound
    max_dimension = np.max(point_cloud_size)
    
    scale_factor = (grid_size - 2*padding - 1) / max_dimension
    scaled_points = (points - min_bound) * scale_factor
    
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    padding_offset = padding
    
    # Voxelize the point cloud
    for point in scaled_points:
        x = int(np.round(point[0] + padding_offset))
        y = int(np.round(point[1] + padding_offset))
        z = int(np.round(point[2] + padding_offset))
        
        if (0 <= x < grid_size and 
            0 <= y < grid_size and 
            0 <= z < grid_size):
            voxel_grid[x, y, z] = 1.0

    transformation_info = {
        'min_bound': min_bound,
        'max_bound': max_bound,
        'scale_factor': scale_factor,
        'padding_offset': padding_offset,
        'rotation_matrix': R
    }

    return voxel_grid, transformation_info

def convert_voxels_to_pcd(grid):

    voxel_pcd = o3d.geometry.PointCloud()
    points = []

    # Iterate over numpy grid
    for z in range(grid.shape[2]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                if grid[x, y, z]:
                    points.append([x, y, z])
                    continue
    
    voxel_pcd.points = o3d.utility.Vector3dVector(np.array(points))

    return voxel_pcd