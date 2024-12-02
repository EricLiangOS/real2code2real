import os
import sys
from argparse import ArgumentParser
import numpy as np
import open3d as o3d
import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from utils.mesh_utils import GaussianExtractor, post_process_mesh, remove_black_spots, remove_smaller_clusters
from scene import Scene
from utils.render_utils import generate_path, create_videos

def render_mesh(args, mesh_name):
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iterations, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'output')
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    

    print("export mesh ...")
    os.makedirs(train_dir, exist_ok=True)
    # set the active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(scene.getTrainCameras())
    # extract the mesh and save
    name = f'unprocessed_{mesh_name}.ply'
    depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
    voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
    sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    
    o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
    print("mesh saved at {}".format(os.path.join(train_dir, name)))
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=1)
    mesh_post = remove_black_spots(mesh_post)
    mesh_post = remove_smaller_clusters(mesh_post)

    o3d.io.write_triangle_mesh(os.path.join(train_dir, f"{mesh_name}.ply"), mesh_post)
    print(f"mesh post processed saved at {mesh_name}.ply")


# for i in range(len(model_files)):
#     print(f"Now getting mesh for object {i + 1}")
#     os.system(f"python render.py -m {model_files[i]} -s {dataset_files[i]} --iteration 30000")
#     mesh = o3d.io.read_triangle_mesh(f"{model_files[i]}/train/meshes_30000/fuse_post.ply")
#     mesh = remove_smaller_clusters(mesh)


#     o3d.io.write_triangle_mesh(os.path.join(dataset_files[i], "output"), "object.mesh")