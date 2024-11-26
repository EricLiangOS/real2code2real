import os
import shutil
from argparse import ArgumentParser
from utils.mask_utils import get_labeled_images, get_object_masks
from utils.dataset_utils import copy_dataset,  convert_files
from render import render_mesh
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training
import time

parser = ArgumentParser("Get masks and mesh extracts of objects within a scene")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--dataset_size", default = 1500, required = True, type = int)

# Mesh arguments
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')


args = parser.parse_args()
# if args.model_path:
#     args.model_path = os.path.join(args.model_path, "%Y%m%d-%H%M%S")

render_mesh(args)



