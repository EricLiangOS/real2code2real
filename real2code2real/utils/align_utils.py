import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from argparse import ArgumentParser
import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import cv2
import itertools


from submodules.TRELLIS.trellis.renderers import MeshRenderer, GaussianRenderer

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

def remove_outliers_largest_cluster(pcd, eps=0.05, min_points=10):

    if len(np.asarray(pcd.points)) < min_points:
        return pcd
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    
    if len(np.unique(labels)) <= 1 and np.unique(labels)[0] == -1:
        return pcd
    
    labels_count = np.bincount(labels[labels >= 0])
    if len(labels_count) == 0:  # No valid clusters found
        return pcd
    
    largest_cluster_label = np.argmax(labels_count)
    
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    largest_cluster_pcd = pcd.select_by_index(largest_cluster_indices)
    
    if pcd.has_colors():
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[largest_cluster_indices])
    
    return largest_cluster_pcd

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

    pcd = pcd.voxel_down_sample(voxel_size=0.002)
    pcd = pcd.remove_duplicated_points()
    
    if remove_outliers:
        pcd = remove_outliers_largest_cluster(pcd)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=3)
        pcd = pcd.select_by_index(ind)
    
    return pcd

def find_nearest_valid_coordinate(depth_img, x, y, max_radius=100):
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
def coordinates_to_3d(data, frame_index, points, output_path = None):

    intrinsics = data["intrinsics"]
    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()
    
    # Get frame data
    frame = data["frames"][frame_index]
    depth_img = frame[1]
    extrinsics = frame[2]
    
    if output_path is not None:
        rgb_img = frame[0].copy()
        vis_img = rgb_img.copy()

    projected_points = []
    new_coords = []
    for i, (x, y) in enumerate(points): 
        new_x, new_y = find_nearest_valid_coordinate(depth_img, x, y)
        new_coords.append((new_x, new_y))

        if output_path is not None and new_x >= 0 and new_y >= 0:
            cv2.circle(vis_img, (int(new_x), int(new_y)), 3, (0, 255, 0), -1)  # Green circles
            cv2.line(vis_img, (int(x), int(y)), (int(new_x), int(new_y)), (0, 255, 255), 1)  # Yellow lines
            cv2.putText(vis_img, f"{i}", (int(new_x)+5, int(new_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if new_x < 0 or new_y < 0 or new_x >= depth_img.shape[1] or new_y >= depth_img.shape[0]:
            projected_points.append([0, 0, 0])
            continue
        
        Z = depth_img[new_y, new_x]
        X = (new_x - cx) * Z / fx
        Y = (new_y - cy) * Z / fy
        
        point_camera = np.array([X, Y, Z, 1.0])
        point_world = np.linalg.inv(extrinsics) @ point_camera
        projected_points.append(point_world[:3])
    
        if output_path is not None:
            # Create legend with the same number of channels as vis_img
            legend_img = np.ones((100, vis_img.shape[1], vis_img.shape[2]), dtype=np.uint8) * 255
            cv2.circle(legend_img, (30, 70), 3, (0, 255, 0, 255) if vis_img.shape[2] == 4 else (0, 255, 0), -1)
            cv2.putText(legend_img, "Nearest Valid Coordinates", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255) if vis_img.shape[2] == 4 else (0, 0, 0), 1)
            
            final_img = np.vstack([vis_img, legend_img])
            
            cv2.imwrite(output_path, final_img)

    return np.asarray(projected_points)


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

def find_nearest_neighbor(source_points, target_points, distance_threshold=0.2):

    tree = KDTree(target_points)
        
    non_zero_indices = np.where(~np.all(source_points == 0, axis=1))[0]
    non_zero_points = source_points[non_zero_indices]
    
    result = np.copy(source_points)
    
    if len(non_zero_points) > 0:
        distances, indices = tree.query(non_zero_points)
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            # if dist <= distance_threshold:
            result[non_zero_indices[i]] = target_points[idx]
            # else:
            #     result[non_zero_indices[i]] = np.zeros(3)
                
    return result

def average_pairwise_distance(points):

    n = points.shape[0]
    
    if n < 2:
        return 0.0
    
    total_distance = 0.0
    num_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(points[i] - points[j])
            total_distance += distance
            num_pairs += 1
    
    return total_distance / num_pairs if num_pairs > 0 else 0.0

def find_best_scale(source_points, target_points, sample_size=4, threshold = 0.05):

    n = source_points.shape[0]
    candidate_ratios = []


    for indices in itertools.combinations(range(n), min(sample_size, n)):
        indices = np.array(indices)
        sample_source = source_points[indices]
        sample_target = target_points[indices]
        
        num_points = len(indices)
        ratios = []

        for i in range(num_points):
            for j in range(i + 1, num_points):
                source_dist = np.linalg.norm(sample_source[i] - sample_source[j])
                target_dist = np.linalg.norm(sample_target[i] - sample_target[j])
                
                if source_dist > 1e-10:  # Avoid division by very small numbers
                    ratios.append(target_dist / source_dist)
        
        candidate_ratios.append(np.mean(ratios))
    
    best_ratio = 1
    best_ratio_votes = 0
    
    for ratio in candidate_ratios:
        ratio_votes = 0
        for i in range(n):
            for j in range(i + 1, n):
                scaled_source = source_points * ratio
                scaled_dist = np.linalg.norm(scaled_source[i] - scaled_source[j])
                target_dist = np.linalg.norm(target_points[i] - target_points[j])
                
                if scaled_dist < 1e-10 or target_dist < 1e-10:
                    continue
                
                if abs(scaled_dist/target_dist - 1.0) < threshold:
                    ratio_votes += 1
                
        print(ratio, ratio_votes)
        
        if ratio_votes > best_ratio_votes:
            best_ratio_votes = ratio_votes
            best_ratio = ratio
    
    return best_ratio

def find_best_transformation_with_voting(source_points, target_points, scale, num_samples=10000, sample_size=9, threshold=0.05):

    scaled_source = source_points * scale
    n = len(source_points)
    
    max_samples = min(num_samples, n if n < sample_size else 10000)
    
    best_transformation = np.eye(4)
    best_votes = 0
    
    for _ in range(max_samples):
        indices = np.random.choice(n, min(sample_size, n), replace=False)
        sample_source = scaled_source[indices]
        sample_target = target_points[indices]
        
        source_mean = np.mean(sample_source, axis=0)
        target_mean = np.mean(sample_target, axis=0)
        source_centered = sample_source - source_mean
        target_centered = sample_target - target_mean
        
        H = np.dot(source_centered.T, target_centered)
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
            
        t = target_mean - np.dot(R, source_mean)
        
        transformation = np.eye(4)
        transformation[:3, :3] = R
        transformation[:3, 3] = t
        
        source_homogeneous = np.hstack((scaled_source, np.ones((n, 1))))
        transformed = (transformation @ source_homogeneous.T).T[:, :3]
        
        distances = np.linalg.norm(transformed - target_points, axis=1)
        votes = np.sum(distances < threshold)
        
        if votes > best_votes:
            best_votes = votes
            best_transformation = transformation
    
    return best_transformation

def find_p2p_transformation(source_points, target_points):

    best_scale = 1
    best_scale_votes = 0

    for i, (s_match, t_match) in enumerate(zip(source_points, target_points)):
        scale = find_best_scale(s_match, t_match, 0.05)


        

        transformation = find_best_transformation_with_voting(s_match, t_match, scale)

    return best_scale, np.eye(4)


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

def find_ransac_transformation(source_pcd, target_pcd, voxel_size=0.05, distance_threshold=0.1, ransac_n=3):
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
        ransac_n=ransac_n,
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

def estimate_similarity_transformation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
    """
    k, n = source.shape

    mx = source.mean(axis=1)
    my = target.mean(axis=1)
    source_centered = source - np.tile(mx, (n, 1)).T
    target_centered = target - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(source_centered**2, axis=0))
    sy = np.mean(np.sum(target_centered**2, axis=0))

    Sxy = (target_centered @ source_centered.T) / n

    U, D, Vt = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = Vt.T
    rank = np.linalg.matrix_rank(Sxy)
    if rank < k:
        raise ValueError("Failed to estimate similarity transformation")

    S = np.eye(k)
    if np.linalg.det(Sxy) < 0:
        S[k - 1, k - 1] = -1

    R = U @ S @ V.T

    s = np.trace(np.diag(D) @ S) / sx
    t = my - s * (R @ mx)

    return R, s, t

def find_best_transformation_ransac(s_matched, t_matched, num_iterations=100, sample_size=6, distance_threshold=0.05):
    
    best_R = np.eye(3)
    best_s = 1.0
    best_t = np.zeros(3)
    best_inlier_ratio = 0.0
    best_inlier_count = 0
    total_points = len(s_matched)
    
    if total_points < sample_size:

        return best_R, best_s, best_t, best_inlier_ratio
    
    for i in range(num_iterations):

        indices = np.random.choice(total_points, sample_size, replace=False)
        sample_source = s_matched[indices]
        sample_target = t_matched[indices]
        
        try:
            R, s, t = estimate_similarity_transformation(sample_source, sample_target)
            
            transformed_source = np.dot(s_matched, R.T) * s + t
            
            distances = np.linalg.norm(transformed_source - t_matched, axis=1)
            inliers = distances < distance_threshold
            inlier_count = np.sum(inliers)
            inlier_ratio = inlier_count / total_points
            
            if inlier_count > best_inlier_count:
                best_R = R
                best_s = s
                best_t = t
                best_inlier_count = inlier_count
                best_inlier_ratio = inlier_ratio
                print(f"Iteration {i}: New best transformation with {inlier_count}/{total_points} inliers ({inlier_ratio:.2%})")
        
        except Exception as e:
            print(f"Warning: Failed to estimate transformation: {e}")
            continue
    
    print(f"Best transformation found with {best_inlier_count}/{total_points} inliers ({best_inlier_ratio:.2%})")

    
    return best_R, best_s, best_t, best_inlier_ratio