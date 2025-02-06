import os

import open3d as o3d
import cv2
import OpenEXR
import Imath
import numpy as np
import json
from scipy.spatial.transform import Rotation
from real2code2real.utils import generate_utils
from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils
import torch
import math

def combine_linesets(linesets):
    combined = o3d.geometry.LineSet()
    points = []
    lines = []
    colors = []
    point_index_offset = 0

    for ls in linesets:
        ls_points = np.asarray(ls.points)
        ls_lines = np.asarray(ls.lines)
        ls_colors = np.asarray(ls.colors)

        points.append(ls_points)
        lines.append(ls_lines + point_index_offset)
        colors.append(ls_colors)

        point_index_offset += ls_points.shape[0]

    combined.points = o3d.utility.Vector3dVector(np.vstack(points))
    combined.lines = o3d.utility.Vector2iVector(np.vstack(lines))
    combined.colors = o3d.utility.Vector3dVector(np.vstack(colors))

    return combined

def create_camera_frustums_ply(extrinsics_list, intrinsics_list, width, height, output_path):
    """
    Creates and saves a single LineSet representing all camera frustums.
    """
    linesets = []
    for extrinsic, intrinsic in zip(extrinsics_list, intrinsics_list):
        extrinsic = extrinsic.detach().cpu().numpy()
        intrinsic = intrinsic.detach().cpu().numpy()
        intrinsic[:3, :] *= 512
        # Intrinsic can be a 3x3 or 4x4; if using Open3D PinholeCameraIntrinsic, pass that here
        lineset = o3d.geometry.LineSet.create_camera_visualization(width, height, intrinsic, extrinsic)
        linesets.append(lineset)

    all_frustums = combine_linesets(linesets)
    o3d.io.write_line_set(output_path, all_frustums)
    return all_frustums

def get_extrinsics_intrinsics(num_frames=100, r=2, fov=40):
    yaws1 = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch1 = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))

    yaws2 = [1.5708] * num_frames
    pitch2 = torch.linspace(0, 2 * 3.1415, num_frames)

    extrinsics1, intrinsics1 = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws1.tolist(), pitch1.tolist(), r, fov
    )
    extrinsics2, intrinsics2 = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws2, pitch2.tolist(), r, fov
    )

    extrinsics = extrinsics1 + extrinsics2
    intrinsics = intrinsics1 + intrinsics2

    return extrinsics, intrinsics

def get_new_extrinsics_intrinsics(num_frames=100, r=2, fov=40):
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


extrinsics, intrinsics = get_new_extrinsics_intrinsics()
create_camera_frustums_ply(extrinsics, intrinsics, 682, 512, "camera_frustums.ply")





# json_path = "/store/real/ehliang/data/home_kitchen_7/new_metadata.json"
# exr_path = "/store/real/ehliang/data/home_kitchen_7/input_depth"
# jpg_path = "/store/real/ehliang/data/home_kitchen_7/input"
# old_json_path = "/store/real/ehliang/data/home_kitchen_7/metadata.json"

# with open(json_path, "r") as f:
#     data = json.load(f)

# # gt_data = generate_utils.prepare_record3d_data(jpg_path, exr_path, json_path)
# data = generate_utils.prepare_existing_mesh_data("outputs/home_kitchen_multiview/object_2", "object_2", 100)
# alignment_frames = [frame for frame in data["frames"] if frame % (len(data["frames"]) // 20) == 0]

# mesh = o3d.io.read_triangle_mesh("outputs/home_kitchen_multiview/object_2/object_2_mesh.obj")

# rot_matrix = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
# mesh.rotate(rot_matrix, center=(0, 0, 0))
# o3d.io.write_triangle_mesh("outputs/home_kitchen_multiview/object_2/object_2_mesh.obj", mesh)


# for frame in alignment_frames:
#     pcd = generate_utils.create_pcd_from_frame(data, frame)
#     o3d.io.write_point_cloud(f"pcd/aligned_pcd_{frame}.ply", pcd)
