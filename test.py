import open3d as o3d
import numpy as np
import torch
import copy
from PIL import Image
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
    
    transformation_info = {
        'min_bound': min_bound,
        'max_bound': max_bound,
        'scale_factor': scale_factor,
        'padding_offset': padding_offset
    }

    return voxel_grid, transformation_info


def align_bounding_boxes(point_cloud, object_geometry):

    R = object_geometry.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    object_geometry.rotate(R, center=(0, 0, 0))

    pc_bbox = point_cloud.get_axis_aligned_bounding_box()
    obj_bbox = object_geometry.get_axis_aligned_bounding_box()
    
    pc_extent = pc_bbox.get_extent()
    obj_extent = obj_bbox.get_extent()
    
    scale_factors = pc_extent / obj_extent
    print(scale_factors)
    pc_center = pc_bbox.get_center()
    obj_center = obj_bbox.get_center()
    translation = pc_center - obj_center
    
    object_geometry.scale(sum(scale_factors)/len(scale_factors), obj_center)
    
    object_geometry.translate(translation)
    
    return point_cloud, object_geometry

def reverse_mesh(mesh, transformation_info, output_path):
    
    translation_vector = np.array([-transformation_info['padding_offset']] * 3)
    mesh.translate(translation_vector)

    min_bound = transformation_info['min_bound']
    scale_factor = transformation_info['scale_factor']
    
    mesh.scale(1 / scale_factor, center=np.zeros(3))
    
    translation_vector = min_bound
    mesh.translate(translation_vector)
    
    o3d.io.write_triangle_mesh(output_path, mesh)
    
    return mesh

def save_voxel(voxels, voxels_path):
    grid = voxels.detach().cpu().numpy()

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

def extract_uv_map(mesh):
    """
    Extract UV coordinates from a mesh.
    
    Args:
        mesh (open3d.geometry.TriangleMesh): Input mesh
    
    Returns:
        tuple: (UV coordinates, UV triangle indices)
    """
    # Check if UV coordinates exist
    if not mesh.has_triangle_uvs():
        print("Warning: Mesh does not have UV coordinates.")
        return None, None
    
    # Get UV coordinates and triangle indices
    uv_coords = mesh.triangle_uvs
    triangle_uv_indices = mesh.triangle_material_ids
    
    return uv_coords, triangle_uv_indices
def convert_voxel_grid_to_pointcloud(grid):

    voxel_pointcloud = o3d.geometry.PointCloud()
    points = []

    # Iterate over numpy grid
    for z in range(grid.shape[2]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                if grid[x, y, z]:
                    points.append([x, y, z])

    voxel_pointcloud.points = o3d.utility.Vector3dVector(np.array(points))
    return voxel_pointcloud
voxel_grid, transform = get_voxels("/store/real/ehliang/r2c2r_blender_data_2/r2c2r_data/test/StorageFurniture/44781/loop_0/link_0.ply")

voxel_pointcloud = convert_voxel_grid_to_pointcloud(voxel_grid)

# voxel_np = np.asarray([voxel_grid.origin + pt.grid_index*voxel_grid.voxel_size for pt in voxel_grid.get_voxels()])

# voxel_pointcloud = o3d.geometry.PointCloud()
# voxel_pointcloud.points = o3d.utility.Vector3dVector(voxel_np)
o3d.io.write_point_cloud("test1.ply", voxel_pointcloud)