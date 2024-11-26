import os
import sys
from argparse import ArgumentParser
from render import render_mesh
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training
import time

if __name__ == "__main__":
    parser = ArgumentParser("Get masks and mesh extracts of objects within a scene")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Training arguments
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # Mesh arguments
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = parser.parse_args(sys.argv[1:])

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

    os.makedirs(args.model_path, exist_ok=True)
    base_dir = os.path.basename(os.path.normpath(args.source_path))
    args.model_path = os.path.join(args.model_path, time.strftime("%Y%m%d-%H%M%S"), base_dir)
    os.makedirs(args.model_path, exist_ok=True)

    print(bcolors.OKCYAN + f"Training 2DGS for {base_dir}" + bcolors.ENDC)   
    try:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)
        print(bcolors.OKGREEN + f"2DGS successfully trained for object {base_dir} at {args.model_path}" + bcolors.ENDC)
    except:
        print(bcolors.FAIL + "Error training" + bcolors.ENDC)

        # Get the mesh
    print(bcolors.OKCYAN + f"Training 2DGS for {base_dir}" + bcolors.ENDC)   

    try:
        render_mesh(args, f"object_{base_dir}")
        print(bcolors.OKGREEN + f"Mesh successfully extracted for object {base_dir} at {args.model_path}" + bcolors.ENDC)
    except:
        print(bcolors.FAIL + "Error extracting mesh" + bcolors.ENDC)

