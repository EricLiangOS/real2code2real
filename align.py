import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import pickle
import trimesh
import cv2
import json
import torch
import os
from real2code2real.mesh_extraction.object_matching import target_matching
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from align_util import (
    render_multi_images,
    render_image,
    as_mesh,
    project_2d_to_3d,
    plot_mesh_with_points,
    plot_image_with_points,
    select_point,
)

VIS = True


import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    RasterizationSettings,
    AmbientLights,
    BlendParams,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from scipy.spatial import cKDTree

def sample_camera_poses(radius, num_samples, num_up_samples=8, device="cpu"):
    """
    Generate camera poses around a sphere with a given radius.
    camera_poses: A list of 4x4 transformation matrices representing the camera poses.
    camera_view_coord = word_coord @ camera_pose
    """
    camera_poses = []
    phi = np.linspace(0, np.pi, num_samples)  # Elevation angle
    phi = phi[1:-1]  # Exclude poles
    theta = np.linspace(0, 2 * np.pi, num_samples)  # Azimuthal angle

    # Generate different up vectors
    up_vectors = [np.array([0, 0, 1])]  # z-axis is up
    for i in range(1, num_up_samples):
        angle = (i / num_up_samples) * np.pi * 2
        up = np.array([np.sin(angle), 0, np.cos(angle)])  # Rotate around y-axis
        up_vectors.append(up)

    for p in phi:
        for t in theta:
            for up in up_vectors:
                x = radius * np.sin(p) * np.cos(t)
                y = radius * np.sin(p) * np.sin(t)
                z = radius * np.cos(p)
                position = np.array([x, y, z])[None, :]
                lookat = np.array([0, 0, 0])[None, :]
                up = up[None, :]
                R, T = look_at_view_transform(radius, t, p, False, position, lookat, up)
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = R
                camera_pose[3, :3] = T
                camera_poses.append(camera_pose)

    print("total poses", len(camera_poses))
    return torch.tensor(np.array(camera_poses), device=device)


def pose_selection_render_superglue(
    raw_img, fov, mesh_path, mesh, crop_img, output_dir
):
    # Calculate suitable rendering radius
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 1.2 * (max_dimension / 2) / np.tan(fov / 2)

    # Render multimle images and feature matching
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(
        mesh_path,
        raw_img.shape[1],
        raw_img.shape[0],
        fov,
        radius=radius,
        num_samples=10,
        num_ups=5,
        device="cuda",
    )
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]
    # Use superglue to match the features
    # best_idx, match_result = image_pair_matching(
    #     grays, crop_img, "temp_output/", viz=True
    # )
    match_result, best_idx, out = target_matching(
        crop_img, grays, -1, f"{output_path}/matches.png",  prepare=False
    )

    best_color = colors[best_idx]
    best_depth = depths[best_idx]
    best_pose = camera_poses[best_idx].cpu().numpy()
    return best_color, best_depth, best_pose, match_result, camera_intrinsics


if __name__ == "__main__":

    json_path = "/store/real/ehliang/data/basement_kitchen/kitchen_interaction_3/new_metadata.json"

    with open(json_path, "r") as f:
        metadata = json.load(f)

    # img_path = "/store/real/ehliang/multiview_data/kitchen_static_3/10_img/object_3/state_1/588.png"
    # mesh_path = "/home/ehliang/real2code2real/outputs/kitchen_interaction_3/object_3/object_3_mesh.glb"
    # output_path = "quick_test/drawer_higher_threshold"
    # input_pose = np.array(metadata["poses"][588])

    img_path = "/store/real/ehliang/multiview_data/kitchen_static_3/10_img/object_1/generation_state/142.png"
    mesh_path = "/home/ehliang/real2code2real/outputs/kitchen_interaction_3/object_1/object_1_mesh.glb"
    output_path = "quick_test/cabinet_1_higher_threshold"
    input_pose = np.array(metadata["poses"][142])

    # img_path = "/store/real/ehliang/multiview_data/kitchen_static_3/10_img/object_5/state_1/810.png"
    # mesh_path = "/home/ehliang/real2code2real/outputs/kitchen_interaction_3/object_5/object_5_mesh.glb"
    # output_path = "quick_test/cabinet_5"
    # input_pose = np.array(metadata["poses"][260])

    
    os.makedirs(output_path, exist_ok=True)


    extrinsics = np.eye(4)
    rotation = Rotation.from_quat(input_pose[:4]).as_matrix()
    translation = input_pose[4:]
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation

    flip_mat = np.eye(4)
    flip_mat[1, 1] = -1
    flip_mat[2, 2] = -1
    extrinsics = flip_mat @ np.linalg.inv(extrinsics)


    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    mesh = as_mesh(mesh)

    raw_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    mask_img = raw_data[:, :, 3]
    raw_img = raw_data[:, :, :3]
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (480, 640))
    mask_img = cv2.resize(mask_img, (480, 640))

    intrinsic = np.array(
        [
            [1333.8367919921875 / 3, 0, 240],
            [0, 1333.8367919921875 / 3, 320],
            [0, 0, 1],
        ]
    )

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))

    if not os.path.exists(f"{output_path}/best_match.pkl"):
        # 2D feature Matching to get the best pose of the object
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
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

        # Render the object and match the features
        best_color, best_depth, best_pose, match_result, camera_intrinsics = (
            pose_selection_render_superglue(
                raw_img,
                fov,
                mesh_path,
                mesh,
                crop_img,
                output_dir=f"{output_path}",
            )
        )
        with open(f"{output_path}/best_match.pkl", "wb") as f:
            pickle.dump(
                [
                    best_color,
                    best_depth,
                    best_pose,
                    match_result,
                    camera_intrinsics,
                    bbox,
                ],
                f,
            )
    else:
        with open(f"{output_path}/best_match.pkl", "rb") as f:
            best_color, best_depth, best_pose, match_result, camera_intrinsics, bbox = (
                pickle.load(f)
            )
   
    # Get the projected 3D matching points on the mesh
    render_matching_points = np.array(match_result[1])
    mesh_matching_points, valid_mask = project_2d_to_3d(
        render_matching_points, best_depth, camera_intrinsics, best_pose
    )
    render_matching_points = render_matching_points[valid_mask]

    raw_matching_points_box = np.array(match_result[0])
    raw_matching_points_box = raw_matching_points_box[valid_mask]
    raw_matching_points = raw_matching_points_box + np.array([bbox[0], bbox[1]])

    if VIS:
        # Do visualization for the matching
        plot_mesh_with_points(
            mesh,
            mesh_matching_points,
            f"{output_path}/mesh_matching.png",
        )
        plot_image_with_points(
            best_depth,
            render_matching_points,
            f"{output_path}/render_matching.png",
        )
        plot_image_with_points(
            raw_img,
            raw_matching_points,
            f"{output_path}/raw_matching.png",
        )

    # Do PnP optimization
    success, rvec, tvec = cv2.solvePnP(
        np.float32(mesh_matching_points),
        np.float32(raw_matching_points),
        np.float32(intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
    )
    projected_points, _ = cv2.projectPoints(
        np.float32(mesh_matching_points),
        rvec,
        tvec,
        intrinsic,
        np.zeros(4, dtype=np.float32),
    )
    error = np.linalg.norm(
        np.float32(raw_matching_points) - projected_points.reshape(-1, 2), axis=1
    ).mean()
    print(f"Reprojection Error: {error}")
    if error > 50:
        print(f"solvePnP failed for this case .$$$$$$$$$$$$$$$$$$$$$$$$$$")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    mesh2raw_camera = np.eye(4, dtype=np.float32)
    mesh2raw_camera[:3, :3] = rotation_matrix
    mesh2raw_camera[:3, 3] = tvec.squeeze()
    

    # test_mesh = o3d.io.read_triangle_mesh(mesh_path)
    # test_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([test_mesh])
    # import pdb
    # pdb.set_trace()
    if VIS:
        pnp_camera_pose = np.eye(4, dtype=np.float32)
        pnp_camera_pose[:3, :3] = np.linalg.inv(rotation_matrix)
        pnp_camera_pose[3, :3] = tvec.squeeze()  # change due to pytorch3D setting
        pnp_camera_pose[:, :2] = -pnp_camera_pose[
            :, :2
        ]  # change due to pytorch3D setting
        color, depth = render_image(
            mesh_path, pnp_camera_pose, raw_img.shape[1], raw_img.shape[0], fov, "cuda"
        )
        vis_mask = depth > 0
        color[0][~vis_mask] = raw_img[~vis_mask]
        plt.imsave(f"{output_path}/pnp_results.png", color[0])

    # Transform the mesh into the real world coordinate
    mesh_points_cam = np.dot(
        mesh2raw_camera,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_points_cam = mesh_points_cam[:, :3]

    final_transform = extrinsics @ mesh2raw_camera

    # Apply the combined transform to mesh vertices
    homogeneous_vertices = np.hstack(
        (np.asarray(mesh.vertices), np.ones((len(mesh.vertices), 1), dtype=np.float32))
    )
    transformed_vertices = (final_transform @ homogeneous_vertices.T).T[:, :3]
    mesh.vertices = transformed_vertices

    # Save result
    mesh.export(f"{output_path}/aligned_mesh.obj")