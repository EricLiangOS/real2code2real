
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
import wandb
import time
import cv2
import json

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
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
    
    if numbers == '':
        numbers = 0
    return int(numbers)

# Generates an object from multiview images, saves the mesh and rendered images, and returns the relevant metadata
def generate(images, output_path, object_name, pipeline, **kwargs):

    pipeline.cuda()

    sparse_path = kwargs.get('sparse_path')
    generated_images = kwargs.get('generated_images', 200)
    sparse_params = kwargs.get('sparse_params')
    slat_params = kwargs.get('slat_params')

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
            sparse_structure_sampler_params=sparse_params,
            slat_sampler_params=slat_params
        )
    else:
        output = pipeline.run_multi_image(
            images,
            seed=1,
            # Optional parameters
            sparse_structure_sampler_params=sparse_params,
            slat_sampler_params=slat_params,
        )

    generate_utils.save_object(output, output_path, object_name)
    generate_utils.save_object(output, output_path, object_name, is_glb=True)

    H, W = images[0].size[1], images[0].size[0]
    mesh_data = generate_utils.prepare_mesh_data(output, H, W, generated_images, output_path, object_name)

    return mesh_data

def align_target_matched_points(s_data, t_data, matches):

    s_matched_all = []
    t_matched_all = []

    for t_frame in matches:
        t_coords, s_coords = matches[t_frame][0]
        s_frame = matches[t_frame][1]

        s_matched = generate_utils.create_points_from_coordinates(s_data, s_frame, s_coords)
        t_matched = generate_utils.create_points_from_coordinates(t_data, t_frame, t_coords)

        s_matched, t_matched = generate_utils.remove_zero_rows(s_matched, t_matched)
        s_matched_all.append(s_matched)
        t_matched_all.append(t_matched)

    if len(s_matched_all) == 0 or len(t_matched_all) == 0:
        print(bcolors.WARNING + "No matched points found" + bcolors.ENDC)
        return 1, np.eye(4)

    s_matched = np.concatenate(s_matched_all, axis=0)
    t_matched = np.concatenate(t_matched_all, axis=0)

    scale, T = generate_utils.find_p2p_transformation(s_matched, t_matched)

    return scale, T

def save_aligned_mesh(matched_info, output_path, object_name):

    s_data = matched_info["mesh_data"]
    t_data = matched_info["multiview_data"]
    mesh = matched_info["mesh"]
    matches = matched_info["matches"]

    scale, T = align_target_matched_points(s_data, t_data, matches)

    s_pcd = o3d.geometry.PointCloud()
    t_pcd = o3d.geometry.PointCloud()
    for frame in matches:
        s_pcd_frame = generate_utils.create_pcd_from_frame(s_data, matches[frame][1])
        t_pcd_frame = generate_utils.create_pcd_from_frame(t_data, frame, remove_outliers=False)
        s_pcd += s_pcd_frame
        t_pcd += t_pcd_frame
    
    o3d.io.write_point_cloud(os.path.join(output_path, f"{object_name}_ground_truth_pcd.ply"), t_pcd)
    o3d.io.write_point_cloud(os.path.join(output_path,f"{object_name}_s_pcd.ply"), s_pcd)

    rot_matrix = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    mesh.rotate(rot_matrix, center=(0, 0, 0))
    mesh.scale(scale, center=(0, 0, 0))
    mesh.transform(T)
    s_pcd.scale(scale, center=(0, 0, 0))

    s_pcd.transform(T)

    print("Finding ransac transformation")
    T = generate_utils.find_ransac_transformation(s_pcd, t_pcd)
    mesh.transform(T)
    s_pcd.transform(T)

    # print("Finding ICP transformation")
    # T = generate_utils.find_icp_transformation(s_pcd, t_pcd)
    # mesh.transform(T)
    # s_pcd.transform(T)

    o3d.io.write_triangle_mesh(os.path.join(output_path, f"{object_name}_aligned_mesh.obj"), mesh)
    
    return mesh

def log_wandb(combined_scene_pcd, mesh_paths, combined_mesh_glb_path, uniform_view_table, alignment_table):
    wandb.log({"object_meshes": [wandb.Object3D(glb_path, scene={"bg_color": "#000000"}) for glb_path in mesh_paths.values()]})
    wandb.log({"generated_mesh_information": uniform_view_table})
    wandb.log({f"alignment_information": alignment_table})

    wandb.log({ "combined_scene_mesh": wandb.Object3D( combined_mesh_glb_path, scene={"bg_color": "#000000"})})

    points = np.asarray(combined_scene_pcd.points)
    colors = np.asarray(combined_scene_pcd.colors)
    print(colors.shape[0], points.shape[0])
    data_3d = np.hstack([points, colors])
    wandb.log({"combined_scene_pcd": wandb.Object3D.from_numpy(data_3d)})

if __name__ == "__main__":

    parser = ArgumentParser("Get masks and mesh extracts of objects within a scene")

    parser.add_argument("--source_dir", "-s", required=True, type=str)
    parser.add_argument("--output_path", "-o", required=True, type=str)
    parser.add_argument("--base_directory", "-b", type=str)
    parser.add_argument("--skip_generation", action="store_true", default=False) 
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--select_frames", action="store_true", default=False)
    parser.add_argument("--sparse_path", type=str, default=None)
    parser.add_argument("--resize", type=int, nargs='+', default=[192, 256])
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sparse_params", type=float, nargs='+', default=[12, 7.5])
    parser.add_argument("--latent_params", type=int, nargs='+', default=[12, 3])

    pipeline = get_pipeline()
    args = parser.parse_args() 

    source_dir = args.source_dir
    output_path = args.output_path
    num_images = args.num_images
    
    os.makedirs(output_path, exist_ok=True)

    wandb.init(project="object_generation_logging", name= os.path.basename(output_path) + time.strftime("%Y%m%d-%H%M%S"))
    uniform_view_table = wandb.Table(columns=["object_name", "multiview_images", "mesh_images"])
    alignment_table = wandb.Table(columns=["object_state", "alignment_images"])
    mesh_paths = {}
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_scene_pcd = o3d.geometry.PointCloud()

    for object_name in sorted(os.listdir(source_dir), key=lambda x: get_number(x)):
        generation_path = os.path.join(source_dir, object_name, "generation_state")
        object_output_path = os.path.join(output_path, object_name)
        os.makedirs(object_output_path, exist_ok=True)

        print(bcolors.OKCYAN + f"Generating object {object_name} from multiview images" + bcolors.ENDC)
        images = []
        selected_frames = []

        if args.select_frames:
            selected_frames_input = input("Enter the frames to select for generation separated by spaces: ")
            selected_frames = selected_frames_input.split(" ")
            selected_frames = [int(frame) for frame in selected_frames]
        else:
            selected_frames = [get_number(os.path.splitext(p)[0]) for p in os.listdir(generation_path)]

        for frame_name in selected_frames:
            image = Image.open(os.path.join(generation_path, f"{frame_name}.{'jpg' if 'jpg' in os.listdir(generation_path)[0] else 'png'}"))
            images.append(image)

        alignment_json_path = os.path.join(object_output_path, f"{object_name}_alignment.json")
        alignment_json = {}

        if not args.skip_generation:
            sparse_params = {"steps": int(args.sparse_params[0]), "cfg_strength": args.sparse_params[1]}
            latent_params = {"steps": int(args.latent_params[0]), "cfg_strength": args.latent_params[1]}
            mesh_data = generate(images, object_output_path, object_name, pipeline, generated_images=num_images, sparse_params=sparse_params, slat_params=latent_params)
        else: 
            print(bcolors.OKGREEN + f"Using existing mesh for {object_name}" + bcolors.ENDC)

            mesh_data = generate_utils.prepare_existing_mesh_data(object_output_path, object_name, num_images)

            if os.path.exists(alignment_json_path):
                with open(alignment_json_path, "r") as f:
                    alignment_json = json.load(f)

        mesh_frames = []
        for frame in mesh_data["frames"]:
            if frame%10 == 0:
                mesh_frames.append(wandb.Image(mesh_data["frames"][frame][0]))
        uniform_view_table.add_data(object_name, [wandb.Image(image) for image in images], mesh_frames)
        
        glb_path = os.path.join(object_output_path, f"{object_name}_mesh.glb")
        mesh_paths[object_name] = glb_path

        mesh_path = os.path.join(object_output_path, f"{object_name}_mesh.obj")
        # texture_path = os.path.join(object_output_path, f"{object_name}_texture.png")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        for state in sorted(os.listdir(os.path.join(source_dir, object_name)), key=lambda x: get_number(x)):
            
            if state == "generation_state":
                continue

            state_multiview_path = os.path.join(source_dir, object_name, state)
            state_output_path = os.path.join(object_output_path, state)

            os.makedirs(state_output_path, exist_ok=True)
            state_data = generate_utils.prepare_record3d_data(state_multiview_path, os.path.join(args.base_directory, "input_depth"), os.path.join(args.base_directory, "new_metadata.json"))

            matched_info = {
                "mesh_data": mesh_data,
                "multiview_data": state_data,
                "mesh": copy.deepcopy(mesh),
                "matches": {}
            }
            
            print(bcolors.OKCYAN + f"Finding {object_name} mesh alignment for {state}" + bcolors.ENDC)
            alignment_images = []

            if args.skip_generation and os.path.exists(alignment_json_path):
                print(bcolors.OKGREEN + f"Using existing alignment for {object_name}, {state}" + bcolors.ENDC)

                for state_frame in alignment_json[state]:
                    matched_points = alignment_json[state][state_frame][0]
                    matched_frame = alignment_json[state][state_frame][1]

                    alignment_image_path = os.path.join(state_output_path, f"{state_frame}_alignment.png")
                    if os.path.exists(alignment_image_path):
                        alignment_images.append(wandb.Image(cv2.imread(alignment_image_path)))
                    
                    print(state, state_frame, matched_frame, len(matched_points[0]))
                    if len(matched_points[0])  < 3:
                        continue

                    matched_info["matches"][int(state_frame)] = [matched_points, matched_frame]
            
            else:
                for state_frame in state_data["frames"]:
                    state_target = state_data["frames"][state_frame][0]
                    mesh_images = [mesh_data["frames"][frame][0] for frame in mesh_data["frames"]]

                    target_matched_path = os.path.join(state_output_path, f"{state_frame}_target_matched.png")
                    matched_points, matched_frame, alignment_image = target_matching(state_target, mesh_images, args.resize, target_matched_path, max_matches=9)
                    if alignment_image is not None:
                        alignment_image = cv2.cvtColor(alignment_image, cv2.COLOR_RGB2BGR) 
                        alignment_images.append(wandb.Image(alignment_image))

                    print(state, state_frame, matched_frame, len(matched_points[0]))
                    if len(matched_points[0])  < 3:
                        continue
                    matched_info["matches"][state_frame] = [matched_points, matched_frame]
                alignment_json[state] = matched_info["matches"]


            with open(alignment_json_path, "w") as f:
                json.dump(alignment_json, f)
            alignment_table.add_data(f"{object_name}_{state}", alignment_images)
            aligned_mesh = save_aligned_mesh(matched_info, state_output_path, f"{object_name}_{state}")

            if state == "state_1":
                # texture_img = np.asarray(Image.open(texture_path))
                # aligned_mesh.textures = [o3d.geometry.Image(texture_img)]
                # aligned_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(aligned_mesh.triangles))

                combined_mesh += aligned_mesh
                combined_scene_pcd += o3d.io.read_point_cloud(os.path.join(state_output_path, f"{object_name}_{state}_ground_truth_pcd.ply"))

    combined_scene_pcd = combined_scene_pcd.voxel_down_sample(voxel_size=0.01)
    
    o3d.io.write_point_cloud(os.path.join(output_path, f"{os.path.basename(args.base_directory)}_combined_scene_pcd.ply"), combined_scene_pcd)
    o3d.io.write_triangle_mesh(os.path.join(output_path, f"{os.path.basename(args.base_directory)}_combined_scene_mesh.obj"), combined_mesh)

    log_wandb(combined_scene_pcd, mesh_paths, os.path.join(output_path, f"{os.path.basename(args.base_directory)}_combined_scene_mesh.obj"), uniform_view_table, alignment_table)
    wandb.finish()        
