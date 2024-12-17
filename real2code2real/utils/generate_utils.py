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
import copy

from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils

def get_voxels(ply_path, grid_size=64, padding=5):
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

    voxel_grid = fill_voxel_holes(voxel_grid)

    transformation_info = {
        'min_bound': min_bound,
        'max_bound': max_bound,
        'scale_factor': scale_factor,
        'padding_offset': padding_offset
    }

    return voxel_grid, transformation_info

def convert_voxels_to_pc(grid):

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

def fill_voxel_holes(voxel_grid, threshold = 15):
    
    filled_voxel_grid = copy.deepcopy(voxel_grid)

    for x in range(1, voxel_grid.shape[2]):
        for y in range(1, voxel_grid.shape[1]):
            for z in range(1, voxel_grid.shape[0]):

                counter = 0

                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            if (x + i >= 0 and x + i < 64 and y + j >= 0 and y + j < 64 and z + k >= 0 and z + k < 64):
                                counter += voxel_grid[x + i, y + j, z + k]
                
                if counter > threshold:
                    filled_voxel_grid[x, y, z] = 1
    
    return filled_voxel_grid

# Transforms aligning_obj to align with fixed_obj
def align_bounding_boxes(aligning_obj, fixed_obj):

    aligned_obj = copy.deepcopy(aligning_obj)

    aligned_obj_bbox = aligned_obj.get_axis_aligned_bounding_box()
    fixed_obj_bbox = fixed_obj.get_axis_aligned_bounding_box()
    
    aligned_obj_extent = aligned_obj_bbox.get_extent()
    fixed_obj_extent = fixed_obj_bbox.get_extent()
 
    scale_factors = fixed_obj_extent / aligned_obj_extent
    fixed_obj_center = fixed_obj_bbox.get_center()
    obj_center = aligned_obj_bbox.get_center()
    translation = fixed_obj_center - obj_center
    
    aligned_obj.scale(sum(scale_factors)/len(scale_factors), obj_center)
    
    aligned_obj.translate(translation)
    
    return aligned_obj

def reverse_mesh(mesh, transformation_info):
    
    translation_vector = np.array([-transformation_info['padding_offset']] * 3)
    mesh.translate(translation_vector)

    min_bound = transformation_info['min_bound']
    scale_factor = transformation_info['scale_factor']
    
    mesh.scale(1 / scale_factor, center=np.zeros(3))
    
    translation_vector = min_bound
    mesh.translate(translation_vector)
        
    return mesh


def save_voxel(voxels, voxels_path):
    grid = voxels[0][0].detach().cpu().numpy()

    # Create new voxel grid object and set voxel_size to some value
    # --> otherwise it will default to 0 and the grid will be invisible
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = 1
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
    o3d.io.write_voxel_grid(voxels_path, voxel_grid)

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

def get_object(output, output_path, object_name="", mapping = {}):
    if object_name:
        object_name += "_"
        
    texture_path = os.path.join(output_path, f"{object_name}texture.png")
    mesh_path = os.path.join(output_path, f"{object_name}mesh.obj")

    obj = postprocessing_utils.to_glb(
        output['gaussian'][0],
        output['mesh'][0],
        # Optional parameters
        simplify=0.1,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )

    obj.export(mesh_path)
    os.rename(os.path.join(output_path, "material_0.png"), texture_path)

    print("Transforming object back to original position")
    if "voxels" in mapping and "transform" in mapping:
        transform_obj = o3d.io.read_triangle_mesh(mesh_path)
        R = transform_obj.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        transform_obj.rotate(R, center=(0, 0, 0))

        voxel_pcd = mapping["voxels"]

        transform_obj = align_bounding_boxes(aligning_obj=transform_obj, fixed_obj=voxel_pcd)
        transform_obj = reverse_mesh(transform_obj, mapping["transform"])

        o3d.io.write_triangle_mesh(mesh_path, transform_obj)



