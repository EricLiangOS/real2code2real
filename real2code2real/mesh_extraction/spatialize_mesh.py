import os

import open3d as o3d
import cv2
import OpenEXR
import Imath
import numpy as np
import json
from scipy.spatial.transform import Rotation

from real2code2real.utils.generate_utils import *

def read_exr_depth(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    
    data_window = exr_file.header()['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    
    depth_str = exr_file.channels(['R'], Imath.PixelType(Imath.PixelType.FLOAT))[0]
    
    # Convert depth data to numpy array
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = depth.reshape((height, width))
    
    return depth

def reshape_depth(depth, width, height):

    new_depth = np.zeros((height, width), dtype=np.float32)

    i_frequency = height / depth.shape[0]
    j_frequency = width / depth.shape[1]

    for i in range(height):
        for j in range(width):
            new_depth[i][j] = depth[int(i / i_frequency)][int(j / j_frequency)]

    return new_depth


def prepare_data(images_dir, depth_dir, metadata_path) -> int:

    frames = [get_number(os.path.splitext(p)[0]) for p in os.listdir(images_dir)]
    frames.sort()

    with open(metadata_path, 'r') as file:
        metadata_dict = json.load(file)

    poses_data = np.array(metadata_dict["poses"])
    
    camera_to_worlds = np.concatenate(
        [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses_data[:, 4:, None]],
        axis=-1,
    ).astype(np.float32)

    homogeneous_coord = np.zeros_like(camera_to_worlds[..., :1, :])
    homogeneous_coord[..., :, 3] = 1
    camera_to_worlds = np.concatenate([camera_to_worlds, homogeneous_coord], -2)

    W, H = metadata_dict["w"], metadata_dict["h"]
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    focal_length = K[0, 0]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=W, 
        height=H,
        fx=focal_length,  # focal length x
        fy=focal_length, 
        cx=W/2, 
        cy=H/2 
    )
    
    output = {
        "h": H,
        "w": W,
        "intrinsic": intrinsic,
        "frames": {}
    }

    for frame in frames:
        img_file = os.path.join(images_dir, f"{frame}.png")
        exr_file = os.path.join(depth_dir, f"{frame}.exr")

        mask_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2RGBA)
        
        depth_img = read_exr_depth(exr_file)
        depth_img = reshape_depth(depth_img, mask_img.shape[1], mask_img.shape[0])

        output["frames"][frame] = [mask_img, depth_img, camera_to_worlds[frame]]


    return output

def process_rgbd_directory( images_dir, depth_dir, metadata_path, sample_size_per_image=1000, output_path=None):

    data = prepare_data(images_dir, depth_dir, metadata_path)

    frames = data["frames"]
    intrinsic = data["intrinsic"]

    point_clouds = []

    for frame in frames:
        print(f"Processing frame {frame}...")

        mask_img, depth_img = frames[frame][:2]
        extrinsic = frames[frame][2]

        alpha_mask = mask_img[:, :, 3] > 0

        if sample_size_per_image is not None:
            alpha_indices = np.argwhere(alpha_mask)
            sampled_indices = alpha_indices[np.random.choice(alpha_indices.shape[0], sample_size_per_image, replace=False)]
            alpha_mask = np.zeros(mask_img.shape[:2], dtype=bool)
            alpha_mask[sampled_indices[:, 0], sampled_indices[:, 1]] = True
        else:
            alpha_mask = np.ones(mask_img.shape[:2], dtype=bool)

        for i in range(depth_img.shape[0]):
            for j in range(depth_img.shape[1]):
                if not alpha_mask[i][j]:
                    depth_img[i][j] = 0

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.array(mask_img[:, :, :3]).astype(np.uint8)),
            o3d.geometry.Image(depth_img),
            depth_scale=1000.0,
            depth_trunc=1000.0
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic, extrinsic
        )

        pcd = pcd.remove_duplicated_points()

        pcd.scale(1000, center=pcd.get_center())

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.5)
        pcd = pcd.select_by_index(ind)
        
        point_clouds.append(pcd)

    combined_pcd = o3d.geometry.PointCloud()

    print("Aligning point clouds...")
    point_clouds = align_pcd(point_clouds)

    for pcd in point_clouds:
        combined_pcd += pcd

    cl, ind = combined_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.3)

    combined_pcd = combined_pcd.select_by_index(ind)

    if output_path:
        o3d.io.write_point_cloud(output_path, combined_pcd)
    
    return combined_pcd

def align_mesh(source_mesh, target_mesh):
    source_pcd = source_mesh.sample_points_uniformly(20000)
    target_pcd = target_mesh.sample_points_uniformly(40000)

    scale_factor, transformation = align_scale_pcd(source_pcd, target_pcd)

    source_mesh.scale(0.9 * scale_factor, center=np.array([0, 0, 0]))
    source_mesh.transform(transformation)
    source_pcd.scale(0.9 * scale_factor, center=np.array([0, 0, 0]))
    source_pcd.transform(transformation)
    
    rotation = get_pcd_alignment_rotation(source_pcd, target_pcd)
    source_mesh.rotate(rotation, center=source_mesh.get_center())

    return source_mesh


