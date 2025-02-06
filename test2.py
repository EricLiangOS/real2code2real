
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from argparse import ArgumentParser
from PIL import Image
from real2code2real.mesh_extraction import PointCloudTo3DPipeline, target_matching, pairwise_matching
from real2code2real.utils import generate_utils
import open3d as o3d
import numpy as np
import copy
import logging

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
logging.getLogger("transformers").setLevel(logging.ERROR)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_pipeline():
    pipeline = PointCloudTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")

    return pipeline

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

# Generates an object from multiview images, saves the mesh and rendered images, and returns the relevant metadata
def generate(multiview_path, output_path, object_name, pipeline, sparse_path=None, generated_images=200):

    pipeline.cuda()

    images = []
    frame_names = [
        p for p in os.listdir(multiview_path)
    ]

    frame_names.sort(key=lambda p: get_number(os.path.splitext(p)[0]))
    for frame_name in frame_names:
        images.append(Image.open(os.path.join(multiview_path, frame_name)))

    if sparse_path:
        # Use mapping if we have accurate pcd to transform the mesh back to
        output, mapping = pipeline.run_sparse_structure(
            images,
            input_path=sparse_path,
            seed=1,
        )
    elif len(images) == 1:
        output = pipeline.run(
            images[0],
            seed=1,
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": 20,
                "cfg_strength": 15,
            },
            slat_sampler_params={
                "steps": 20,
                "cfg_strength": 15,
            }
        )
    else:
        output = pipeline.run_multi_image(
            images,
            seed=1,
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": 40,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 40,
                "cfg_strength": 3,
            }
        )

    generate_utils.save_object(output, output_path, object_name)

    H, W = images[0].size[1], images[0].size[0]
    mesh_data = generate_utils.prepare_mesh_data(output, H, W, generated_images, output_path, object_name)

    return mesh_data

pipeline = get_pipeline()
object_name = "object_4"
pcd_path = "/store/real/ehliang/multiview_outputs/kitchen_static_4/10_img_100_steps_7_sparse_3_latent/object_4/state_1/object_4_state_1_ground_truth_pcd.ply"
# pcd_path = f"/home/ehliang/real2code2real/outputs/kitchen_static_1/{object_name}/state_1/{object_name}_state_1_ground_truth_pcd.ply"
output_path = "sparse"
multiview_path = f"/store/real/ehliang/multiview_data/kitchen_static_4/20_img_pruned/object_4/generation_state"

mesh_data = generate(multiview_path, output_path, object_name, pipeline, generated_images=200)