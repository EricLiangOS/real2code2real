import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from argparse import ArgumentParser
import open3d as o3d
import numpy as np
import copy
import torch
from PIL import Image
import OpenEXR
import Imath
import json
import cv2
from scipy.spatial.transform import Rotation
import math

from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils
from submodules.TRELLIS.trellis.renderers import MeshRenderer, GaussianRenderer

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

def convert_to_rgba(image):
    rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=image.dtype)

    rgba_image[:, :, :3] = image[:, :, :3]

    rgba_image[:, :, 3] = 255

    black_pixels = np.all(image[:, :, :3] == [0, 0, 0], axis=-1)

    return rgba_image 

def remove_zero_rows(arr1, arr2):
    mask1 = ~np.all(arr1 == 0, axis=1)
    mask2 = ~np.all(arr2 == 0, axis=1)
    mask = mask1 & mask2
    return arr1[mask], arr2[mask]

def combine_transformations(transform_list):
    result = np.eye(4)
    for transform in transform_list:
        result = result @ transform
    return result

def save_object(object_output, output_path, object_name="", is_glb=False):
    if object_name:
        object_name += "_"
        
    import imageio
    video = render_utils.render_video(object_output['mesh'][0])['normal']
    imageio.mimsave(os.path.join(output_path, f"{object_name}sample_mesh.mp4"), video, fps=30)
    video = render_utils.render_video(object_output['gaussian'][0])['color']
    imageio.mimsave(os.path.join(output_path, f"{object_name}sample_gs.mp4"), video, fps=30)

    obj = postprocessing_utils.to_glb(
        object_output['gaussian'][0],
        object_output['mesh'][0],
        # Optional parameters
        simplify=0.85,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
        verbose=False
    )

    if not is_glb:
        mesh_path = os.path.join(output_path, f"{object_name}mesh.obj")
    else:
        mesh_path = os.path.join(output_path, f"{object_name}mesh.glb")

    obj.export(mesh_path)


def get_extrinsics_intrinsics(num_frames=200, r=2.7, fov=40):
    yaws = []
    pitches = []

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(num_frames):
        y = 1.0 - (2.0 * i + 1.0) / num_frames
        radius = math.sqrt(1.0 - y * y)
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        yaw = math.atan2(x, z)
        pitch = math.asin(y)

        yaws.append(yaw)
        pitches.append(pitch)

    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, r, fov)
    return extrinsics, intrinsics

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

def get_rgb_frames(output, num_frames, resolution=512, bg_color=(0, 0, 0), colors_overwrite=None):
    rgb_frames = []

    options = {'resolution': resolution, 'bg_color': bg_color}
    extrinsics, intrinsics = get_extrinsics_intrinsics(num_frames = num_frames)

    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = options.get('resolution', 512)
    renderer.rendering_options.near = options.get('near', 0.8)
    renderer.rendering_options.far = options.get('far', 1.6)
    renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
    renderer.rendering_options.ssaa = options.get('ssaa', 1)
    renderer.pipe.kernel_size = 0.1
    renderer.pipe.use_mip_gaussian = True

    for j, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
        res = renderer.render(output['gaussian'][0], extr, intr, colors_overwrite=colors_overwrite)
        rgb_frames.append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))

    return rgb_frames
        
def get_depth_frames(output, num_frames, resolution=512, bg_color=(0, 0, 0)):
    depth_frames = []
    
    options = {'resolution': resolution, 'bg_color': bg_color}
    extrinsics, intrinsics = get_extrinsics_intrinsics(num_frames = num_frames)

    renderer = MeshRenderer()
    renderer.rendering_options.resolution = options.get('resolution', 512)
    renderer.rendering_options.near = options.get('near', 1)
    renderer.rendering_options.far = options.get('far', 100)
    renderer.rendering_options.ssaa = options.get('ssaa', 4)

    for j, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
        res = renderer.render(output['mesh'][0], extr, intr, ["depth"])
        depth_frames.append(res['depth'].detach().cpu().numpy())

    return depth_frames


def resize_rgb_frames(frames, height, width, output_dir=None):
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    curr_height, curr_width = frames[0].shape[:2]


    vertical_pad = 0
    horizontal_pad = 0

    if width < height:
        new_height = round(height * curr_width / width)
        vertical_pad = (new_height - curr_height) // 2
        curr_height = new_height
    else:
        new_width = round(width * curr_height / height)
        horizontal_pad = (new_width - curr_width) // 2
        curr_width = new_width

    padded_rgb = []

    for i, frame in enumerate(frames):
        image_np = np.asarray(frame)
        image_np = np.pad(image_np, ((vertical_pad, vertical_pad), (horizontal_pad, horizontal_pad), (0, 0)), mode='constant', constant_values=0)
        image_np = convert_to_rgba(image_np)
        padded_rgb.append(image_np)

        if output_dir is not None:
            image = Image.fromarray(image_np)
            image.save(f"{output_dir}/rgb_{i:05}.png")
    
    return padded_rgb

# Pads the depth to the desired aspect ratio
def resize_depth_frames(frames, height, width, output_dir=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    curr_height, curr_width = frames[0].shape[:2]

    vertical_pad = 0
    horizontal_pad = 0

    if width < height:
        new_height = round(height * curr_width / width)
        vertical_pad = (new_height - curr_height) // 2
    else:
        new_width = round(width * curr_height / height)
        horizontal_pad = (new_width - curr_width) // 2

    padded_depths = []

    for i, frame in enumerate(frames):
        depth_np = np.asarray(frame)
        depth_np = np.pad(depth_np, ((vertical_pad, vertical_pad), (horizontal_pad, horizontal_pad)), mode='constant', constant_values=0)
        padded_depths.append(depth_np)

        if output_dir is not None:

            header = OpenEXR.Header(depth_np.shape[1], depth_np.shape[0])
            header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT), 1, 1)}
            
            output = OpenEXR.OutputFile(f"{output_dir}/depth_{i:05}.exr", header)
            depth = depth_np.astype(np.float32).tobytes()
            output.writePixels({'R': depth})
            output.close()

    return padded_depths

def prepare_existing_mesh_data(output_path, object_name, generated_images):
    rgb_path = os.path.join(output_path, f"{object_name}_rgb")
    depth_path = os.path.join(output_path, f"{object_name}_depth")

    extrinsics_tensor, intrinsics_tensor = get_extrinsics_intrinsics(generated_images)
    intrinsics_matrix = intrinsics_tensor[0].cpu().numpy()
    
    assert os.path.isdir(rgb_path) and os.path.isdir(depth_path), "RGB and depth images must be present in output directory"

    frames = {}
    
    for frame in range(generated_images):
        img_file = os.path.join(rgb_path, f"rgb_{frame:05}.png")
        exr_file = os.path.join(depth_path, f"depth_{frame:05}.exr")

        mask_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2RGBA)

        mask_img = convert_to_rgba(mask_img)

        depth_img = read_exr_depth(exr_file)
        depth_img = cv2.resize(depth_img, dsize=(mask_img.shape[1], mask_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_img[depth_img < 1.5] = 0

        frames[frame] = [mask_img, depth_img, extrinsics_tensor[frame].detach().cpu().numpy()]

    H, W = frames[0][0].shape[:2]
    intrinsics_matrix[:3, :] *= min(H, W)

    intrinsics  = o3d.camera.PinholeCameraIntrinsic(
        width=W, 
        height=H,
        fx=intrinsics_matrix[0, 0],
        fy=intrinsics_matrix[1, 1], 
        cx= W/2, 
        cy= H/2 
    )

    data = {
        "h": H,
        "w": W,
        "intrinsics": intrinsics,
        "frames": frames
    }

    return data

def prepare_mesh_data(output, H, W, generated_images=300, output_path = None, object_name=None):
    rgb_frames = get_rgb_frames(output, generated_images)
    depth_frames = get_depth_frames(output, generated_images)

    if output_path is not None:
        rgb_path = os.path.join(output_path, f"{object_name}_rgb")
        depth_path = os.path.join(output_path, f"{object_name}_depth")
    else:
        rgb_path = None
        depth_path = None

    rgb_frames = resize_rgb_frames(rgb_frames, H, W, rgb_path)
    depth_frames = resize_depth_frames(depth_frames, H, W, depth_path)

    extrinsics_tensor, intrinsics_tensor = get_extrinsics_intrinsics(generated_images)
    intrinsics_matrix = intrinsics_tensor[0].cpu().numpy()

    new_H, new_W = rgb_frames[0].shape[:2]
    intrinsics_matrix[:3, :] *= min(new_H, new_W)

    intrinsics  = o3d.camera.PinholeCameraIntrinsic(
        width=new_W, 
        height=new_H,
        fx=intrinsics_matrix[0, 0],
        fy=intrinsics_matrix[1, 1], 
        cx=new_W/2, 
        cy=new_H/2 
    )

    data = {
        "h": new_H,
        "w": new_W,
        "intrinsics": intrinsics,
        "frames": {}
    }

    for i in range(generated_images):
        depth_frames[i][depth_frames[i] < 1.5] = 0
        data["frames"][i] = [rgb_frames[i], depth_frames[i], extrinsics_tensor[i].detach().cpu().numpy()]

    return data

def prepare_3d_scanner_data(images_dir,  depth_dir, json_dir):
    frame_names = [
        get_number(os.path.basename(p)) for p in os.listdir(images_dir)
        if ("frame_" in os.path.basename(p) and os.path.splitext(p)[-1] in [".jpg"])
    ]
    frame_names.sort()

    output = {
        "frames": {}
    }

    for i, frame in enumerate(frame_names):
        json_path = os.path.join(json_dir, f"frame_{frame:05d}.json")
        image_path = os.path.join(images_dir, f"frame_{frame:05d}.jpg") 
        depth_path = os.path.join(depth_dir, f"depth_{frame:05d}.png")

        if os.path.exists(json_path) and os.path.exists(image_path) and os.path.exists(depth_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

        image_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGRA2RGBA)

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        depth_raw = cv2.resize(depth_raw, dsize=(image_raw.shape[1], image_raw.shape[0]), interpolation=cv2.INTER_CUBIC)

        extrinsics_data = np.reshape(np.array(data["cameraPoseARFrame"]), (4, 4))
        extrinsics_data[:, 3] /= 1000
        extrinsics_data[:3, :3] = extrinsics_data[:3, :3].T
        extrinsics_data = np.linalg.inv(extrinsics_data)

        output["frames"][frame] = [image_raw, depth_raw, extrinsics_data]
        
        if i == 0:
            intrinsics_data = np.array(data["intrinsics"])
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=image_raw.shape[0],
                height=image_raw.shape[1],
                fx=intrinsics_data[0],
                fy=intrinsics_data[4], 
                cx=intrinsics_data[2], 
                cy=intrinsics_data[5] 
            )

            output.update({
                "w": image_raw.shape[0],
                "h": image_raw.shape[1],
                "intrinsics": intrinsics
            })
        
    return output

def prepare_record3d_data(images_dir, depth_dir, metadata_path):

    frames = [get_number(os.path.splitext(p)[0]) for p in os.listdir(images_dir)]
    frames.sort()

    with open(metadata_path, 'r') as file:
        metadata_dict = json.load(file)

    poses_data = np.array(metadata_dict["poses"])

    W, H = metadata_dict["w"], metadata_dict["h"]
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    focal_length = K[0, 0]

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=W, 
        height=H,
        fx=focal_length,
        fy=focal_length, 
        cx=W/2, 
        cy=H/2 
    )
    
    output = {
        "h": H,
        "w": W,
        "intrinsics": intrinsics,
        "frames": {}
    }

    for frame in frames:
        img_file = os.path.join(images_dir, f"{frame}.png")

        if not os.path.isfile(img_file):
            img_file = os.path.join(images_dir, f"{frame}.jpg")

        exr_file = os.path.join(depth_dir, f"{frame}.exr")

        mask_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2RGBA)

        depth_img = read_exr_depth(exr_file)
        depth_img = cv2.resize(depth_img, dsize=(mask_img.shape[1], mask_img.shape[0]), interpolation=cv2.INTER_CUBIC)

        extrinsics = np.eye(4)
        rotation = Rotation.from_quat(poses_data[frame][:4]).as_matrix()
        translation = poses_data[frame][4:]
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = translation

        flip_mat = np.eye(4)
        flip_mat[1, 1] = -1
        flip_mat[2, 2] = -1
        extrinsics = flip_mat @ np.linalg.inv(extrinsics)
        
        output["frames"][frame] = [mask_img, depth_img, extrinsics]

    return output

def create_pcd_from_frame(data, frame_index, samples=5000, remove_outliers=True):

    intrinsics = data["intrinsics"]
    frame = data["frames"][frame_index]
    
    mask_img = frame[0]
    depth_img = frame[1].copy()
    extrinsics = frame[2]

    if mask_img.shape[2] == 4:
        alpha_mask = mask_img[:, :, 3] > 0
    else:
        alpha_mask = np.ones(mask_img.shape[:2], dtype=bool)        

    alpha_indices = np.argwhere(alpha_mask)

    sampled_indices = alpha_indices[np.random.choice(alpha_indices.shape[0], samples, replace=False)]
    alpha_mask[sampled_indices[:, 0], sampled_indices[:, 1]] = True

    depth_img[~alpha_mask] = 0

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(mask_img[:, :, :3].astype(np.uint8)),
        o3d.geometry.Image(depth_img),
        depth_scale=1.0,
        depth_trunc=1000.0,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics, extrinsics
    )

    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd = pcd.remove_duplicated_points()
    
    if remove_outliers:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.5)
        pcd = pcd.select_by_index(ind)
    
    return pcd


def create_points_from_coordinates(data, frame_index, points):
    intrinsics = data["intrinsics"]
    frame = data["frames"][frame_index]
    
    mask_img = frame[0]
    extrinsics = frame[2]

    projected_points = []

    for x, y in points:
        alpha_mask = np.zeros(mask_img.shape[:2], dtype=bool)
        alpha_mask[y, x] = True
        
        depth_img = frame[1].copy()
        depth_img[~alpha_mask] = 0

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(mask_img[:, :, :3].astype(np.uint8)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=1000.0,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsics, extrinsics
        )

        pcd = pcd.remove_duplicated_points()

        if (len(pcd.points) > 0):
            projected_points.append(pcd.points[0])
        else:
            projected_points.append([0, 0, 0])

    projected_points = np.asarray(projected_points)

    return projected_points

def find_p2p_transformation(source_points, target_points):

    scales = []
    for i in range(len(source_points)):
        for j in range(i + 1, len(source_points)):
            dist1 = np.linalg.norm(source_points[i] - source_points[j])
            dist2 = np.linalg.norm(target_points[i] - target_points[j])
            if dist1 > 1e-10 and dist2 > 1e-10:
                scales.append(dist2 / dist1)
    avg_scale = np.mean(scales) if scales else 1.0

    source_points = source_points * avg_scale
    N = source_points.shape[0]

    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    denominator = np.linalg.norm(source_centered)
    if denominator < 1e-10:
        scale = 1.0
    else:
        scale = np.linalg.norm(target_centered) / denominator

    H = np.dot(source_centered.T, target_centered)
    U, _, Vt = np.linalg.svd(H)
    rotation_matrix = np.dot(Vt.T, U.T)

    if np.linalg.det(rotation_matrix) < 0:
        Vt[2, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)

    translation = target_mean - scale * np.dot(rotation_matrix, source_mean)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = scale * rotation_matrix
    transformation_matrix[:3, 3] = translation

    return avg_scale, transformation_matrix

def compute_scaling_factor(source, target):
    source_centroid = np.mean(np.asarray(source.points), axis=0)
    target_centroid = np.mean(np.asarray(target.points), axis=0)
    source_dists = np.linalg.norm(np.asarray(source.points) - source_centroid, axis=1)
    target_dists = np.linalg.norm(np.asarray(target.points) - target_centroid, axis=1)
    scale = np.mean(target_dists) / np.mean(source_dists)
    return scale

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, fpfh

def find_ransac_transformation(source_pcd, target_pcd, voxel_size=0.05, distance_threshold=0.1):
    scale = compute_scaling_factor(source_pcd, target_pcd)
    
    scaled_source = copy.deepcopy(source_pcd)
    scaled_points = np.asarray(scaled_source.points) * scale
    scaled_source.points = o3d.utility.Vector3dVector(scaled_points)
    
    source_down, source_fpfh = preprocess_point_cloud(scaled_source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1.0)
    )
    S = np.eye(4)
    S[:3, :3] *= scale
    final_transformation = result.transformation @ S
    return final_transformation

def find_icp_transformation(source_pcd, target_pcd, threshold=0.05, init_transformation=np.eye(4)):
    source_down = source_pcd.voxel_down_sample(voxel_size=threshold)
    target_down = target_pcd.voxel_down_sample(voxel_size=threshold)

    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
    source_down.orient_normals_consistent_tangent_plane(30)
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
    target_down.orient_normals_consistent_tangent_plane(30)

    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, 
        target_down, 
        threshold, 
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, 
            relative_rmse=1e-7, 
            max_iteration=2000
        )
    )
    return result_icp.transformation