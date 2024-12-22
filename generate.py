
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from argparse import ArgumentParser
from PIL import Image
from real2code2real.mesh_extraction import PointCloudTo3DPipeline, process_rgbd_directory, align_mesh
from real2code2real.utils.generate_utils import get_object, reverse_transformation
from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils
import imageio
import open3d as o3d

def get_pipeline():
    pipeline = PointCloudTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")

    return pipeline

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

def generate(multiview_path, output_path, object_name, num_images, pipeline, sparse_path=None):

    pipeline.cuda()

    images = []

    frame_names = [
        p for p in os.listdir(multiview_path)
    ]
    frame_names.sort(key=lambda p: get_number(os.path.splitext(p)[0]))
    num_images = min(num_images, len(os.listdir(multiview_path)))
    frequency = len(frame_names)/args.num_images
    for i in range(args.num_images):
        image_name = frame_names[int(i * frequency)]
        print(image_name)
        images.append(Image.open(os.path.join(multiview_path, image_name)))

    mapping = {}

    if sparse_path:
        output, mapping = pipeline.run(
            images,
            input_path=sparse_path,
            seed=1,
        )
    else:
        output = pipeline.run_multi_image(
            images,
            seed=1,
        )

    get_object(output, output_path, object_name)

    return mapping

if __name__ == "__main__":

    parser = ArgumentParser("Get masks and mesh extracts of objects within a scene")

    parser.add_argument("--multiview_path", "-v", required=True, type=str)
    parser.add_argument("--output_path", "-o", required=True, type=str)
    parser.add_argument("--base_directory", "-b", type=str)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=15000)
    parser.add_argument("--sparse_path", "-s", type=str, default=None)

    pipeline = get_pipeline()
    args = parser.parse_args()

    multiview_path = args.multiview_path
    output_path = args.output_path
    num_images = args.num_images
    sample_size = args.sample_size
    
    os.makedirs(output_path, exist_ok=True)
    
    exr_path = os.path.join(args.base_directory, "input_depth")
    json_path = os.path.join(args.base_directory, "new_metadata.json")
    depth_pcd_path = os.path.join(output_path, "depth_pcd.ply")

    print("Generating object from multiview images")
    generate(multiview_path, output_path, "multiview_object", num_images, pipeline)

    print("Generating object from depth point cloud")
    depth_pcd = process_rgbd_directory(multiview_path, exr_path, json_path, sample_size, depth_pcd_path)
    mapping = generate(multiview_path, output_path, "depth_object", num_images, pipeline, depth_pcd_path)
    
    print("Transforming object back to original position")
    depth_mesh_path = os.path.join(output_path, "depth_object_mesh.obj")
    depth_mesh = o3d.io.read_triangle_mesh(depth_mesh_path)
    source_mesh_path = os.path.join(output_path, "multiview_object_mesh.obj")
    source_mesh = o3d.io.read_triangle_mesh(source_mesh_path)
    source_mesh = align_mesh(source_mesh, depth_mesh)

    source_mesh = reverse_transformation(source_mesh, depth_mesh, mapping)
    depth_mesh = reverse_transformation(depth_mesh, depth_mesh, mapping)
    
    o3d.io.write_triangle_mesh(source_mesh_path, source_mesh)
    
