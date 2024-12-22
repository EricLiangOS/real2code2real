import os
import json
import numpy as np

os.environ["EGL_PLATFORM"] = "surfaceless"
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R

def convert_to_rgba(image):
    rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=image.dtype)

    rgba_image[:, :, :3] = image

    rgba_image[:, :, 3] = 255

    # Set the alpha channel to 0 (fully transparent) for black pixels
    black_pixels = np.all(image == [0, 0, 0], axis=-1)
    rgba_image[black_pixels, 3] = 0

    return rgba_image 

def create_camera_frustum(intrinsic_matrix, extrinsic_matrix, width, height, scale=0.1):
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # Define the frustum corners in the camera coordinate system
    corners = np.array([
        [0, 0, 0],
        [cx / fx, cy / fy, 1],
        [-cx / fx, cy / fy, 1],
        [-cx / fx, -cy / fy, 1],
        [cx / fx, -cy / fy, 1]
    ]) * scale

    # Transform the corners to the world coordinate system
    corners = np.dot(extrinsic_matrix[:3, :3], corners.T).T + extrinsic_matrix[:3, 3]

    # Create lines connecting the corners
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def position_and_scale_object(frames, obj):
    poses = np.array([frames[frame][1][:3] for frame in frames])
    center = np.mean(poses, axis=0)

    obj.translate(center)

    distances = np.linalg.norm(poses - center, axis=1)
    radius = np.max(distances)

    scale_factor = 0.5 * radius / np.max(obj.get_max_bound() - obj.get_min_bound())
    obj.scale(scale_factor, center)

    return obj

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

image_dir = "/User/ehliang/Downloads/drawer_input_2"
object_path = "/User/ehliang/Downloads/multiview_drawer_2_mesh.obj"
# object_path = "/store/real/ehliang/data/home_kitchen_3/textured_output.obj"
texture_path = "/User/ehliang/Downloads/multiview_drawer_2_texture.png"
json_path = "/User/ehliang/Downloads/new_metadata.json"

frame_names = [
        p for p in os.listdir(image_dir)
    ]
    # .split("_")[1])
frame_names.sort(key=lambda p: get_number(os.path.splitext(p)[0]))

frames = {}

for frame in frame_names:
    image = Image.open(os.path.join(image_dir, frame))
    frames[int(frame.split(".")[0])] = [image]
    
with open(json_path, 'r') as file:
    data = json.load(file)

for frame in frames:
    frames[frame].append(data["poses"][frame])
    frames[frame].append(data["perFrameIntrinsicCoeffs"][frame])

obj = o3d.io.read_triangle_mesh(object_path, True)

obj = position_and_scale_object(frames, obj)

texture = o3d.io.read_image(texture_path)
# obj.textures = [texture]
obj.compute_vertex_normals()

width, height = image.size

mat_mesh =  o3d.visualization.rendering.MaterialRecord()
mat_mesh.albedo_img = texture
mat_mesh.shader = "defaultUnlit"

geometries = [obj]

for frame in frames:
    intrinsic_matrix = np.array([[frames[frame][2][0], 0, frames[frame][2][3]], 
                                 [0, frames[frame][2][1], frames[frame][2][2]], 
                                 [0, 0, 1]])
    pose = frames[frame][1]
    translation = pose[:3]
    quaternion = pose[3:]
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation

    frustum = create_camera_frustum(intrinsic_matrix, extrinsic_matrix, width, height)
    geometries.append(frustum)

o3d.visualization.draw_geometries(geometries)

print("Simulating capture...")
# simulate_capture(frames, renderer)

