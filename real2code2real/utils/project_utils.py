
import os
import pickle
import open3d as o3d
import numpy as np
import copy
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import cv2
import json
import OpenEXR
from scipy.spatial.transform import Rotation
import Imath
import test3utils
from collections import deque


import numpy as np

# Manual implementation of open3d's 3d projection function for images, returns a W x H x 3 numpy array
def image_to_3d(data, frame_index):
    intrinsics = data["intrinsics"]
    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()

    frame = data["frames"][frame_index]
    mask_img = frame[0]
    depth_img = frame[1]
    extrinsics = frame[2]

    if mask_img.shape[2] == 4:
        alpha_mask = mask_img[:, :, 3] > 0
    else:
        alpha_mask = np.ones(mask_img.shape[:2], dtype=bool)

    point_array = []

    for y in range(depth_img.shape[0]):
        row = []
        for x in range(depth_img.shape[1]):
            if not alpha_mask[y, x]:
                continue

            Z = depth_img[y, x]

            if Z <= 0 or np.isnan(Z):
                row.append([0, 0, 0])
                continue

            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy

            point_camera = np.array([X, Y, Z, 1.0])
            point_world = np.linalg.inv(extrinsics) @ point_camera

            row.append(point_world[:3])

        point_array.append(row)

    return np.asarray(point_array)

def find_nearest_valid_coordinate(depth_img, x, y, max_radius=25):
    if (x < 0 or y < 0 or x >= depth_img.shape[1] or y >= depth_img.shape[0]):
        return -1, -1
    
    if (depth_img[y, x] > 0 and not np.isnan(depth_img[y, x])):
        return x, y
    
    for radius in range(1, min(max_radius + 1, 100)):
        x_min = max(0, x - radius)
        x_max = min(depth_img.shape[1] - 1, x + radius)
        y_min = max(0, y - radius)
        y_max = min(depth_img.shape[0] - 1, y + radius)
        
        window = depth_img[y_min:y_max+1, x_min:x_max+1]
        
        valid_mask = (window > 0) & (~np.isnan(window))
        valid_indices = np.where(valid_mask)
        
        if len(valid_indices[0]) > 0:
            y_coords = valid_indices[0] + y_min
            x_coords = valid_indices[1] + x_min
            
            sq_distances = (y_coords - y)**2 + (x_coords - x)**2
            
            min_idx = np.argmin(sq_distances)
            
            return int(x_coords[min_idx]), int(y_coords[min_idx])
    
    return -1, -1

# Manual implementation of open3d's 3d projection function for points, returns a N x 3 numpy array
def coordinates_to_3d(data, frame_index, points):

    intrinsics = data["intrinsics"]
    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()
    
    # Get frame data
    frame = data["frames"][frame_index]
    depth_img = frame[1]
    extrinsics = frame[2]
    
    projected_points = []
    
    for x, y in points: 
        x, y = find_nearest_valid_coordinate(depth_img, x, y)
        
        if x < 0 or y < 0 or x >= depth_img.shape[1] or y >= depth_img.shape[0]:
            projected_points.append([0, 0, 0])
            continue
        
        Z = depth_img[y, x]
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        
        point_camera = np.array([X, Y, Z, 1.0])
        point_world = np.linalg.inv(extrinsics) @ point_camera
        projected_points.append(point_world[:3])
    
    return np.asarray(projected_points)

def count_zero_points(points):
    zero_points = np.all(points == 0, axis=1)
    return np.sum(zero_points)

pickle_path =  "/home/ehliang/real2code2real/outputs/test/object_1/object_1_alignment.pkl"
output_path = "quick_test"
with open(pickle_path, "rb") as f:
    alignment_states = pickle.load(f)

state = 'state_1'

t_data = test3utils.prepare_record3d_data(
    "/store/real/ehliang/data/basement_kitchen/kitchen_interaction_3/object_1/images",
    "/store/real/ehliang/data/basement_kitchen/kitchen_interaction_3/input_depth",
    "/store/real/ehliang/data/basement_kitchen/kitchen_interaction_3/new_metadata.json"
)

matched_info = {
    "object_name": "object_1",
    "state": state,
    # "mesh_data": mesh_data,
    # "multiview_data": state_data,
    # "mesh": copy.deepcopy(mesh),
    "matches": {}
}
print("starting")
for state_frame in alignment_states[state]:
    matched_points = alignment_states[state][state_frame][0]
    matched_frame = alignment_states[state][state_frame][1]

    if len(matched_points[0])  < 3:
        continue

    matched_info["matches"][int(state_frame)] = [matched_points, matched_frame]

s_matched_all = []
t_matched_all = []
for t_frame in matched_info["matches"]:
    t_coords, s_coords = matched_info["matches"][t_frame][0]
    s_frame = matched_info["matches"][t_frame][1]

    t_matched_new = coordinates_to_3d(t_data, t_frame, t_coords)


    t_pcd_frame = test3utils.create_pcd_from_frame(t_data, t_frame, remove_outliers=True)

    t_matched_new = np.asarray(t_matched_new)
    print(t_frame, count_zero_points(t_matched_new))
    t_matched_new = test3utils.find_nearest_neighbor(t_matched_new, np.asarray(t_pcd_frame.points))
    t_matched_points_new = o3d.geometry.PointCloud()
    t_matched_points_new.points = o3d.utility.Vector3dVector(t_matched_new)
    t_matched_points_new.paint_uniform_color([0, 0, 1])

    o3d.io.write_point_cloud(os.path.join(output_path, f"{t_frame}_matched_new.ply"), t_matched_points_new)
    print(t_frame, count_zero_points(t_matched_new))
