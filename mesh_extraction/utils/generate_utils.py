import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from argparse import ArgumentParser
import imageio
from PIL import Image
import open3d as o3d
import trimesh
import numpy as np
import torch
import scipy.ndimage

def convert_obj_to_voxel(obj_path, grid_size=64):
    mesh = trimesh.load(obj_path)
    
    # Compute the bounding box of the mesh
    bounds = mesh.bounding_box.bounds
    
    # Calculate the scale and translation to fit the object in the grid
    mesh_min = bounds[0]
    mesh_max = bounds[1]
    mesh_size = mesh_max - mesh_min
    
    # Determine the maximum dimension to ensure proper scaling
    max_dimension = np.max(mesh_size)
    
    # Scale factor to fit the object in the grid
    scale_factor = (grid_size - 1) / max_dimension
    
    # Create a transformation matrix
    transform = np.eye(4)
    transform[:3, :3] *= scale_factor
    transform[:3, 3] = -mesh_min * scale_factor
    
    # Apply the transformation to center and scale the mesh
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(transform)
    
    # Create a voxel grid
    voxel_grid = transformed_mesh.voxelized(pitch=1.0).matrix
    
    # Pad or crop the voxel grid to exactly match the grid size
    if voxel_grid.shape[0] > grid_size or voxel_grid.shape[1] > grid_size or voxel_grid.shape[2] > grid_size:
        voxel_grid = voxel_grid[:grid_size, :grid_size, :grid_size]
    else:
        pad_needed = grid_size - np.array(voxel_grid.shape)
        voxel_grid = np.pad(voxel_grid, 
                            ((0, pad_needed[0]), 
                            (0, pad_needed[1]), 
                            (0, pad_needed[2])), 
                            mode='constant', 
                            constant_values=0)
    
    # Convert to binary tensor (bool or float)
    voxel_tensor = torch.tensor(voxel_grid > 0, dtype=torch.float32)
    
    return voxel_tensor


def convert_ply_to_voxel(ply_path, grid_size=64, padding=10):
    pcd = o3d.io.read_point_cloud(ply_path)
    
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
    
    # Convert to PyTorch tensor
    voxel_tensor = torch.tensor(voxel_grid, dtype=torch.float32)
    
    return voxel_tensor

def save_tensor_to_voxel(voxels):
    grid = voxels[0][0].detach().cpu().numpy()

    # Create new voxel grid object and set voxel_size to some value
    # --> otherwise it will default to 0 and the grid will be invisible
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = 0.1
    # Iterate over numpy grid
    for z in range(grid.shape[2]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                if grid[x, y, z] == 0:
                    continue
                # Create a voxel object
                voxel = o3d.geometry.Voxel()
                voxel.color = np.array([0.0, 0.0, 0.0]) * (1.0 - grid[x, y, z])
                voxel.grid_index = np.array([x, y, z])
                # Add voxel object to grid
                voxel_grid.add_voxel(voxel)
    o3d.io.write_voxel_grid("voxels.ply", voxel_grid)

def obb_from_axis(points: np.ndarray, axis_idx: int):
    """get the oriented bounding box from a set of points and a pre-defined axis"""
    # Compute the centroid, points shape: (N, 3)
    centroid = np.mean(points, axis=0)
    # Align points with the fixed axis idx ([1, 0, 0]), so ignore x-coordinates
    if axis_idx == 0:
        points_aligned = points[:, 1:]
        axis_1 = np.array([1, 0, 0])
    elif axis_idx == 1:
        points_aligned = points[:, [0, 2]]
        axis_1 = np.array([0, 1, 0])
    elif axis_idx == 2:
        points_aligned = points[:, :2]
        axis_1 = np.array([0, 0, 1])
    else:  
        raise ValueError(f"axis_idx {axis_idx} not supported!") 

    # Compute PCA on the aligned points
    points_centered = points_aligned - np.mean(points_aligned, axis=0)  
    cov = np.cov(points_centered.T)
    _, vh = np.linalg.eig(cov)
    axis_2, axis_3 = vh[:, 0], vh[:, 1] # 2D!!
    # axis_2, axis_3 = vh[0], vh[1] # 2D!! 
    axis_2, axis_3 = np.round(axis_2, 1), np.round(axis_3, 1)  
    x2, y2 = axis_2
    x3, y3 = axis_3 
    
    if sum(axis_2 < 0) == 2 or (sum(axis_2 < 0) == 1 and sum(axis_2 == 0) == 1):
        axis_2 = -axis_2
    if sum(axis_3 < 0) == 2 or (sum(axis_3 < 0) == 1 and sum(axis_3 == 0) == 1):
        axis_3 = -axis_3

    # remove -0
    axis_2 = np.array([0. if x == -0. else x for x in axis_2])
    axis_3 = np.array([0. if x == -0. else x for x in axis_3]) 
    if axis_idx == 0:
        evec = np.array([
            axis_1,
            [0, axis_2[0], axis_2[1]],
            [0, axis_3[0], axis_3[1]]
            ]).T
    elif axis_idx == 1:
        evec = np.array([
            [axis_2[0], 0, axis_2[1]],
            axis_1,
            [axis_3[0], 0, axis_3[1]]
            ]).T 
    elif axis_idx == 2:
        evec = np.array([
            [axis_2[0], axis_2[1], 0],
            [axis_3[0], axis_3[1], 0],
            axis_1,
            ]).T 
    # Use these axes to find the extents of the OBB
    # # Project points onto these axes 
    all_centered = points - centroid # (N, 3)
    projection = all_centered @ evec # (N, 3) @ (3, 3) -> (N, 3)

    # Find min and max projections to get the extents
    _min = np.min(projection, axis=0)
    _max = np.max(projection, axis=0)
    extent = (_max - _min) # / 2 -> o3d takes full length
    # Construct the OBB using the centroid, axes, and extents 
 
    return dict(center=centroid, R=evec, extent=extent)

def get_handcraft_obb(mesh, z_weight=1.5):
    all_obbs = []
    if isinstance(mesh, np.ndarray):
        vertices = mesh    
    else:
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_faces() 
        vertices = np.array(mesh.vertices) 
    if len(vertices) == 0:
        return dict(center=np.zeros(3), R=np.eye(3), extent=np.ones(3))
    for axis_idx in range(3):
        obb_dict = obb_from_axis(vertices, axis_idx)
        all_obbs.append(obb_dict)

    # select obb with smallest volume, but prioritize axis z 
    bbox_sizes = [np.prod(x['extent']) for x in all_obbs] 
    bbox_sizes[2] /= z_weight # prioritize z axis 
    min_size_idx  = np.argmin(bbox_sizes)
    obb_dict = all_obbs[min_size_idx]
    return obb_dict

def convert_ply_to_voxel_detailed(ply_path):
    
    # Read the point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Convert to numpy array
    scaled_points = np.asarray(pcd.points)

    obb_dict = get_handcraft_obb(scaled_points)

    center = torch.tensor(obb_dict['center']).cuda()
    extent = torch.tensor(obb_dict['extent']).cuda()
    R = torch.tensor(obb_dict['R']).cuda()
    scaled_points = (torch.from_numpy(scaled_points).cuda() - center) @ R

    return scaled_points
