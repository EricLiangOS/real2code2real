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
from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils
from submodules.TRELLIS.trellis.renderers import MeshRenderer, GaussianRenderer
import math


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

def get_extrinsics_intrinsics(num_frames=200, r=1.5, fov=40):
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
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.HALF)
    depth_str = exr_file.channel("R", pt)
    depth = np.frombuffer(depth_str, dtype=np.float16).astype(np.float32)
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

def crop_image(img):
    mask_img = img[:, :, 3]
    raw_img = img[:, :, :3]       

    bbox = np.argwhere(mask_img > 0.8 * 255)
    bbox = (
        np.min(bbox[:, 1]),
        np.min(bbox[:, 0]),
        np.max(bbox[:, 1]),
        np.max(bbox[:, 0]),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = (
        int(center[0] - size // 2),
        int(center[1] - size // 2),
        int(center[0] + size // 2),
        int(center[1] + size // 2),
    )
    # Make sure the bounding box is within the image
    bbox = (
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(raw_img.shape[1], bbox[2]),
        min(raw_img.shape[0], bbox[3]),
    )
    # Get the masked cropped image used for superglue
    crop_img = raw_img.copy()
    mask_bool = mask_img > 0
    crop_img[~mask_bool] = 0
    crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    return crop_img, bbox


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

def combine_images_horizontally(image_paths, shift_color):
    # Read all images
    images = [cv2.imread(img_path) for img_path in image_paths]
    images = [img for img in images if img is not None]

    if shift_color:
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

    if not images:
        raise ValueError("No valid images found in the provided paths")
    
    heights = [img.shape[0] for img in images]
    width_sum = sum(img.shape[1] for img in images)
    max_height = max(heights)
    
    result = np.zeros((max_height, width_sum, 3), dtype=np.uint8)
    
    current_x = 0
    for img in images:
        h, w = img.shape[:2]

        result[0:h, current_x:current_x+w] = img
        current_x += w
    
    return result

def plot_points_on_image(image, coordinates, output_path, radius=3, color=(0, 0, 255), thickness=2):
    marked_image = image.copy()
    marked_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)

    if isinstance(coordinates, tuple) and len(coordinates) == 2:
        xs, ys = coordinates
        coordinates = list(zip(xs, ys))
    
    # Draw circles at each point
    for x, y in coordinates:
        x, y = int(x), int(y)
        cv2.circle(marked_image, (x, y), radius, color, thickness)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, marked_image)
    
    return marked_image

def create_image_grid_from_paths(image_paths_array):

    if not isinstance(image_paths_array, np.ndarray) or image_paths_array.ndim != 2:
        raise ValueError("Input must be a 2D numpy array of image paths")
    
    rows, cols = image_paths_array.shape
    
    images = np.empty(image_paths_array.shape, dtype=object)
    row_heights = np.zeros(rows, dtype=int)
    col_widths = np.zeros(cols, dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            if image_paths_array[i, j]:
                img = cv2.imread(image_paths_array[i, j])
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images[i, j] = img
                    h, w = img.shape[:2]
                    row_heights[i] = max(row_heights[i], h)
                    col_widths[j] = max(col_widths[j], w)
    
    grid_height = np.sum(row_heights)
    grid_width = np.sum(col_widths)
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background
    
    y_offset = 0
    for i in range(rows):
        x_offset = 0
        for j in range(cols):
            if images[i, j] is not None:
                img = images[i, j]
                h, w = img.shape[:2]
                
                # Center the image in its cell
                y_pad_top = (row_heights[i] - h) // 2
                x_pad_left = (col_widths[j] - w) // 2
                
                grid[y_offset + y_pad_top:y_offset + y_pad_top + h, 
                     x_offset + x_pad_left:x_offset + x_pad_left + w] = img
            
            x_offset += col_widths[j]
        y_offset += row_heights[i]
    
    return grid

# pcd is an open3d pointcloud, highlight_pcd is a numpy array of points to highlight
def plot_points_3d(pcd, output_path=None, point_size=0.05, fig_size=(10, 10), highlight_pcd=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3)) * 0.5
    
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Swap x and y coordinates to show y-axis in front
    ax.scatter(
        points[:, 1], points[:, 0], points[:, 2],  # Swapped x and y
        c=colors,
        s=point_size,
        marker='.'
    )
    
    if highlight_pcd is not None:
        highlight_points = np.asarray(highlight_pcd)
        # Also swap x and y for highlighted points
        ax.scatter(
            highlight_points[:, 1], highlight_points[:, 0], highlight_points[:, 2],  # Swapped x and y
            c='magenta',
            s=10,
            marker='.',
            alpha=1.0
        )
    
    # Update axis labels to reflect the swap
    ax.set_xlabel('Y (front)')  # This was x
    ax.set_ylabel('X')          # This was y
    ax.set_zlabel('Z')
    
    # Adjust view to better show the now-front y-axis
    ax.view_init(elev=20, azim=0)  # Different view angle to highlight front y-axis
    
    # Keep the rest of the function the same
    ax.set_box_aspect([1.0, 1.0, 1.0])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()