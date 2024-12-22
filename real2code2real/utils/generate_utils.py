import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from argparse import ArgumentParser
from PIL import Image
import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation

from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

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


def transform_voxel_to_pcd(mesh, transformation_info):
    
    translation_vector = np.array([-transformation_info['padding_offset']] * 3)
    mesh.translate(translation_vector)

    min_bound = transformation_info['min_bound']
    scale_factor = transformation_info['scale_factor']
    
    mesh.scale(1 / scale_factor, center=np.zeros(3))
    
    mesh.translate(min_bound)

    R = transformation_info['rotation_matrix']
    mesh.rotate(R.T, center=np.zeros(3))

    return mesh

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

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def get_alignment_transformation(target, source):

    voxel_size = 0.05

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return result.transformation

def align_pcd(point_clouds):

    aligned_point_clouds = []
    target = point_clouds[0]

    for i, source in enumerate(point_clouds):
        if i != 0:
            transformation = get_alignment_transformation(target, source)
            source.transform(transformation)

        aligned_point_clouds.append(source) 
    
    return aligned_point_clouds

def align_scale_pcd(source_pcd, target_pcd):

    target_obb = target_pcd.get_oriented_bounding_box()
    target_extents = np.sort(target_obb.extent)

    source_obb = source_pcd.get_oriented_bounding_box()
    source_extents = np.sort(source_obb.extent)

    scale_factor = np.mean(target_extents / source_extents)
    source_pcd.scale(scale_factor, center=source_obb.center)

    scaled_source_obb = source_pcd.get_oriented_bounding_box()

    translation = target_obb.center - scaled_source_obb.center
    source_pcd.translate(translation)

    source_extents = scaled_source_obb.extent

    # Compile the transformation
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    return scale_factor, transformation

# Transforms aligning_obj to align with fixed_obj
def align_bounding_boxes(aligning_obj, fixed_obj):

    aligned_obj = copy.deepcopy(aligning_obj)

    aligned_obj_bbox = aligned_obj.get_axis_aligned_bounding_box()
    fixed_obj_bbox = fixed_obj.get_axis_aligned_bounding_box()
    
    aligned_obj_extent = aligned_obj_bbox.get_extent()
    fixed_obj_extent = fixed_obj_bbox.get_extent()
 
    scale_factors = fixed_obj_extent / aligned_obj_extent
    scale_factors = sum(scale_factors)/len(scale_factors)

    fixed_obj_center = fixed_obj_bbox.get_center()
    obj_center = aligned_obj_bbox.get_center()
    translation = fixed_obj_center - obj_center
    
    
    return scale_factors, translation

def get_pcd_alignment_rotation(source, target):

    rotations = [0, np.pi/2, np.pi, 3*np.pi/2]

    best_distance = 1E9
    best_rotation = None

    for x in rotations:
        for y in rotations:
            for z in rotations:
                source_copy = copy.deepcopy(source)
                R = source_copy.get_rotation_matrix_from_xyz((x, y, z))
                source_copy.rotate(R, center=source_copy.get_center())

                v = target.compute_point_cloud_distance(source_copy)
                v = np.square(np.asarray(v))
                average_distance = np.mean(v)

                if average_distance < best_distance:
                    best_distance = average_distance
                    best_rotation = R
    
    return best_rotation

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
 
def reverse_transformation(source_object, reference_object, mapping):

    R = source_object.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    source_object.rotate(R, center=(0, 0, 0))

    voxel_pcd = mapping["voxels"]
    scale_factor, translation = align_bounding_boxes(aligning_obj=reference_object, fixed_obj=voxel_pcd)

    source_object.scale(scale_factor, center=source_object.get_center())
    source_object.translate(translation)

    source_object = transform_voxel_to_pcd(source_object, mapping["transform"])

    return source_object

def get_object(object_output, output_path, object_name=""):
    if object_name:
        object_name += "_"
        
    texture_path = os.path.join(output_path, f"{object_name}texture.png")
    mesh_path = os.path.join(output_path, f"{object_name}mesh.obj")
    
    obj = postprocessing_utils.to_glb(
        object_output['gaussian'][0],
        object_output['mesh'][0],
        # Optional parameters
        simplify=0.1,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )

    obj.export(mesh_path)
    os.rename(os.path.join(output_path, "material_0.png"), texture_path)

