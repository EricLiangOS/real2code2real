import os
import json
import numpy as np
import open3d as o3d
from PIL import Image
import random

def load_intrinsics(intrinsics_list):
    return np.array(intrinsics_list).reshape(3, 3)

def load_pose(pose_list):
    return np.array(pose_list).reshape(4, 4)

def generate_rays(image_shape, intrinsics, extrinsics):
    height, width = image_shape[:2]
    rays = np.zeros((height, width, 6))

    for y in range(height):
        for x in range(width):
            if random.random() > 0.05:
                continue
            ray_origin = extrinsics[:3, 3]
            ray_direction = np.linalg.inv(intrinsics) @ np.array([x, y, 1])
            ray_direction = extrinsics[:3, :3] @ ray_direction
            ray_direction = -10*ray_direction / np.linalg.norm(ray_direction)

            rays[y, x, :3] = ray_origin
            rays[y, x, 3:] = ray_direction

    return rays

def ray_intersect(ray_origin, ray_direction, scene):
    
    ray = []
    for i in ray_origin:
        ray.append(i)
    for i in ray_direction:
        ray.append(i)

    rays = o3d.core.Tensor([ray], dtype=o3d.core.Dtype.Float32)
    intersections = scene.cast_rays(rays)

    return intersections

def project_points_to_mesh(image, intrinsics, pose, scene):
    height, width = image.shape[:2]
    all_intersections = []

    points = []
    colors = []

    rays = []
    for y in range(height):
        for x in range(width):
            if image[y, x, 3] > 0:  # Check alpha channel
                ray_origin = pose[:3, 3]
                ray_direction = np.linalg.inv(intrinsics) @ np.array([x, y, 1])
                ray_direction = pose[:3, :3] @ ray_direction
                ray_direction = ray_direction / np.linalg.norm(ray_direction)

                ray = []
                for i in ray_origin:
                    ray.append(i)
                for i in ray_direction:
                    ray.append(i)

                rays.append(ray)

    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    intersections = scene.cast_rays(rays)

    hit = intersections['t_hit'].isfinite()
    points = rays[hit][:,:3] + rays[hit][:,3:]*intersections['t_hit'][hit].reshape((-1,1))
    pcd = o3d.t.geometry.PointCloud(points)

                # # Perform ray casting
                # intersections = ray_intersect(ray_origin, ray_direction, scene)
                # if intersections:
                #     all_intersections.append(intersections)
                #     print(intersections)

    return pcd

def visualize_rays(rays, output_file):
    lines = []
    points = []

    height, width, _ = rays.shape
    for y in range(height):
        for x in range(width):
            ray_origin = rays[y, x, :3]
            ray_direction = rays[y, x, 3:]
            ray_end = ray_origin + ray_direction * 0.1  # Scale the direction for visualization

            points.append(ray_origin)
            points.append(ray_end)
            lines.append([len(points) - 2, len(points) - 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Save the line set to a file
    o3d.io.write_line_set(output_file, line_set)

    # Optionally, visualize the line set
    o3d.visualization.draw_geometries([line_set])

def create_point_cloud(points, colors=None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def mask_rays(rays, image):

    new_rays = []
    counter = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x, 3] != 0 and image[y, x] != [0, 0, 0, 0, 0, 0] and x > 200 and x < 440 and y > 100 and y < 260:
                counter += 1
                new_rays.append(rays[y, x])

    print(counter)    
    return rays

def process_object(object_dir, mesh, optimized_poses_dir):
    images_dir = os.path.join(object_dir, "images")
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    frame_range = [int(f.split('.')[0]) for f in image_files]
    start_frame, end_frame = min(frame_range), max(frame_range)

    points = []
    colors = []

    all_points = o3d.geometry.PointCloud()

    for frame in range(start_frame, end_frame + 1):
        if random.random() > 0.05:
            continue

        json_path = os.path.join(optimized_poses_dir, f"{frame:05d}.json")
        image_path = os.path.join(images_dir, f"{frame}.png")

        if os.path.exists(json_path) and os.path.exists(image_path):
            print(json_path, image_path)
            with open(json_path, 'r') as f:
                pose_data = json.load(f)
            
            intrinsics = load_intrinsics(pose_data["intrinsics"])
            pose = load_pose(pose_data["cameraPoseARFrame"])
            image = np.array(Image.open(image_path))
            print(image.shape)
            
            rays = generate_rays(image.shape, intrinsics, pose)
            rays = mask_rays(rays, image)

            rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

            intersections = mesh.cast_rays(rays)

            hit = intersections['t_hit'].isfinite()
            points = rays[hit][:,:3] + rays[hit][:,3:]*intersections['t_hit'][hit].reshape((-1,1))
            pcd = o3d.t.geometry.PointCloud(points)

            all_points += pcd.to_legacy()

    o3d.io.write_point_cloud(os.path.join(object_dir, "object_point_cloud.ply"), all_points)

def main(scene_dir):
    mesh_path = os.path.join(scene_dir, "textured_output.obj")
    optimized_poses_dir = os.path.join(scene_dir, "optimized_poses")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    # Create a scene and add the mesh
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    object_dirs = [os.path.join(scene_dir, d) for d in os.listdir(scene_dir) if d.startswith("object_")]
    for object_dir in object_dirs:
        process_object(object_dir, scene, optimized_poses_dir)


if __name__ == "__main__":
    scene_dir = "/store/real/ehliang/data/home_kitchen_3"
    main(scene_dir)