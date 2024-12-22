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
from scipy.spatial.transform import Rotation
from real2code2real.utils.generate_utils import *
from real2code2real.mesh_extraction.spatialize_mesh import prepare_data, process_rgbd_directory

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


target_path = "/home/ehliang/real2code2real/outputs/object_3_duplicate/object_3_duplicate_mesh.obj"
object_path = '/home/ehliang/real2code2real/outputs/object_3/object_3_mesh.obj'
mesh = o3d.io.read_triangle_mesh(object_path)
sampled_mesh_pcd = mesh.sample_points_uniformly(20000)

target_mesh = o3d.io.read_triangle_mesh(target_path)
target_pcd = target_mesh.sample_points_uniformly(40000)

scale_factor, transformation = align_scale_pcd(sampled_mesh_pcd, target_pcd)

mesh.scale(scale_factor, center=np.array([0, 0, 0]))
mesh.transform(transformation)
sampled_mesh_pcd.scale(0.9*scale_factor, center=np.array([0, 0, 0]))
sampled_mesh_pcd.transform(transformation)

rotation = get_pcd_alignment_rotation(sampled_mesh_pcd, target_pcd)

mesh.rotate(rotation, center=mesh.get_center())
o3d.io.write_triangle_mesh("test.obj", mesh)

print("Done!")
