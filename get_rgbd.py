import os

import numpy as np
import open3d as o3d
import cv2
import OpenEXR
import Imath
import numpy as np
from PIL import Image
import quaternion  # For quaternion operations
import json

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

def read_exr_depth(exr_path, new_height, new_width):

    # Open the EXR file
    exr_file = OpenEXR.InputFile(exr_path)
    
    # Get the data window (image dimensions)
    data_window = exr_file.header()['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    
    # Read the depth channel
    depth_channel = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_str = exr_file.channels(['R'], Imath.PixelType(Imath.PixelType.FLOAT))[0]
    
    # Convert depth data to numpy array
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = depth.reshape((height, width))
    
    new_depth = np.zeros((new_height, new_width), dtype=np.float32)

    i_freq = new_height/height
    j_freq = new_width/width

    for i in range(new_height):
        for j in range(new_width):
            new_depth[i, j] = depth[int(i / i_freq), int(j / j_freq)]

    return new_depth

def create_rgbd_point_cloud(png_mask_path, 
                             exr_depth_path, 
                             camera_pose,
                             intrinsic,
                             sample_size=None):

    # Read mask image
    mask_img = cv2.imread(png_mask_path, cv2.IMREAD_UNCHANGED)
    
    # Read depth image
    depth_img = read_exr_depth(exr_depth_path, mask_img.shape[0], mask_img.shape[1])
    
    # Extract color image (RGB channels from mask)
    color_img = mask_img[:, :, :3]
    alpha_mask = mask_img[:, :, 3] > 0
    
    # Extract camera position and rotation
    camera_pos = np.array(camera_pose[:3])
    camera_rot = quaternion.from_float_array(camera_pose[3:])

    # Find valid pixels (within mask and with valid depth)
    valid_pixels = np.where((alpha_mask) & (depth_img > 0))
    
    # Randomly sample points if sample_size is specified
    if sample_size is not None and sample_size < len(valid_pixels[0]):
        indices = np.random.choice(len(valid_pixels[0]), sample_size, replace=False)
        valid_pixels = (valid_pixels[0][indices], valid_pixels[1][indices])
    
    # Prepare point cloud data
    points = []
    colors = []
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=data["w"], 
        height=data["h"],
        fx=intrinsic[0],  # focal length x
        fy=intrinsic[1],  # focal length y
        cx=intrinsic[2],  # principal point x
        cy=intrinsic[3]   # principal point y
    )

    # Convert pixel coordinates to 3D points
    for y, x in zip(*valid_pixels):
        # Get depth value
        depth = depth_img[y, x]
        
        # Convert pixel to 3D point in camera coordinate system
        point_camera = np.array([
            (x - intrinsic.intrinsic_matrix[0, 2]) * depth / intrinsic.intrinsic_matrix[0, 0],
            (y - intrinsic.intrinsic_matrix[1, 2]) * depth / intrinsic.intrinsic_matrix[1, 1],
            depth
        ])
        
        # Rotate point using camera rotation
        point_rotated = quaternion.as_rotation_matrix(camera_rot) @ point_camera
        
        # Translate point with camera position
        point_world = point_rotated + camera_pos
        
        # Add point and corresponding color
        points.append(point_world)
        colors.append(color_img[y, x] / 255.0)  # Normalize colors
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def process_rgbd_directory(images_dir,
                           depth_dir,
                            camera_poses,
                            intrinsics,
                            sample_size_per_image=1000,
                            output_path=None):

    # Initialize combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    
    frames = [get_number(os.path.splitext(p)[0]) for p in os.listdir(images_dir)]
    frames.sort()

    for frame in frames:
        mask_path = os.path.join(images_dir, f"{frame}.png")
        depth_path = os.path.join(depth_dir, f"{frame}.exr")

        # Create point cloud for this image
        current_pcd = create_rgbd_point_cloud(
            mask_path, 
            depth_path, 
            camera_poses[frame],
            intrinsics[frame], 
            sample_size=sample_size_per_image
        )
        
        # Add to combined point cloud
        combined_pcd += current_pcd
    
    # Optional: save point cloud
    if output_path:
        o3d.io.write_point_cloud(output_path, combined_pcd)
    
    return combined_pcd

def rewrite_json(json_dir, output_dir, frame_correspondance, rewrite_categories):
    with open(json_dir, 'r') as file:
        data = json.load(file)
    
    new_json = {}

    for key in data:
        if key not in rewrite_categories:
            new_json[key] = data[key]
            continue

        new_values = []
        for new_frame in frame_correspondance:
            new_values.append(data[key][frame_correspondance[new_frame]])
        
        new_json[key] = new_values
    
    with open(output_dir, 'w') as file:
        json.dump(new_json, file)

json_path = '/store/real/ehliang/data/home_kitchen_2/new_metadata.json'
with open(json_path, 'r') as file:
    data = json.load(file)

image_path = "/store/real/ehliang/cabinet_input"
exr_path = "/store/real/ehliang/data/home_kitchen_2/input_depth"

poses = data["poses"]

# Create point cloud (sampling 1000 random points)data["perFrameIntrinsicCoeffs"][3]
pcd = process_rgbd_directory(
    image_path, 
    exr_path, 
    poses,
    data["perFrameIntrinsicCoeffs"],
    sample_size_per_image=20000
)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=0.5)

# Extract the inlier cloud
inlier_cloud = pcd.select_by_index(ind)

# Visualize point cloud
o3d.io.write_point_cloud("test.ply", inlier_cloud)
