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

def position_and_scale_object(frames, obj):
    centers = np.array([frames[frame][1][:3, 3] for frame in frames])

    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(centers))

    center = obb.center

    # Step 4: Translate the object to this center
    obj.translate(center)

    # Step 5: Scale the object to be a quarter of the size of the OBB
    scale_factor = 0.75 * max(obb.extent) / np.max(obj.get_max_bound() - obj.get_min_bound())
    obj.scale(scale_factor, center)

    return obj

def simulate_capture(frames, renderer):

    simulated_images = []

    for frame in frames:
        width, height = frames[frame][0].size
        extrinsic_matrix = frames[frame][1]
        intrinsic_matrix = frames[frame][2]

        renderer.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height) 

        # Capture the image
        image_np = np.asarray(renderer.render_to_image())
        image_np = convert_to_rgba(image_np)
        image = Image.fromarray(image_np)
        image.save(f"outputs/rendered/example_{frame}.png")

        simulated_images.append(image_np)

    return simulated_images

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

def get_metadata(images_dir, json_dir):
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    frame_range = [get_number(f.split('.')[0]) for f in image_files]
    start_frame, end_frame = min(frame_range), max(frame_range)

    frames = {}

    for frame in range(start_frame, end_frame + 1):

        json_path = os.path.join(json_dir, f"{frame:05d}.json")
        image_path = os.path.join(images_dir, f"{frame}.png")

        if os.path.exists(json_path) and os.path.exists(image_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            frames[frame] = [Image.open(os.path.join(images_dir, image_path)) , np.reshape(data["cameraPoseARFrame"], (4, 4)), np.reshape(data["intrinsics"], (3, 3))]
            frames[frame][1][:3, :3] = -frames[frame][1][:3, :3]
    
    return frames


image_dir = "/store/real/ehliang/data/home_kitchen_3/object_1/images"
object_path = "/home/ehliang/real2code2real/outputs/drawer_3/multiview_drawer_2_mesh.obj"
# object_path = "/store/real/ehliang/data/home_kitchen_3/textured_output.obj"
texture_path = "/home/ehliang/real2code2real/outputs/drawer_3/multiview_drawer_2_texture.png"
json_path = "/store/real/ehliang/data/home_kitchen_3/optimized_poses"


frames = get_metadata(image_dir, json_path)

obj = o3d.io.read_triangle_mesh(object_path, True)

obj = position_and_scale_object(frames, obj)

texture = o3d.io.read_image(texture_path)
obj.textures = [texture]
obj.compute_vertex_normals()

width, height = frames[0][0].size
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
renderer.scene.view.set_post_processing(False)

mat_mesh =  o3d.visualization.rendering.MaterialRecord()
mat_mesh.albedo_img = texture
mat_mesh.shader = "defaultUnlit"
renderer.scene.add_geometry("object", obj, mat_mesh, True)



print("Simulating capture...")
simulate_capture(frames, renderer)

