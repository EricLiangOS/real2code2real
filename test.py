import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from argparse import ArgumentParser
import imageio
from PIL import Image
from mesh_extraction.trellis.pipelines import TrellisImageTo3DPipeline
from mesh_extraction.trellis.utils import render_utils, postprocessing_utils

def get_pipeline():
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")

    return pipeline

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

def generate(args, pipeline):
    source_path = args.source_path
    pointcloud_path = args.file_path
    output_path = args.output_path

    pipeline.cuda()

    images = []

    frame_names = [
        p for p in os.listdir(source_path)
    ]
    # .split("_")[1])
    frame_names.sort(key=lambda p: get_number(os.path.splitext(p)[0]))

    # Load an image
    frequency = len(frame_names)/args.num_images

    for i in range(args.num_images):
        image_name = frame_names[int(i * frequency)]
        print(image_name)
        images.append(Image.open(os.path.join(source_path, image_name)))


    # Run the pipeline
    output = pipeline.run(
        images,
        # Optional parameters
        seed=1,
        main_path=pointcloud_path,
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )

    # Render the outputs
    video = render_utils.render_video(output['gaussian'][0])['color']
    imageio.mimsave(output_path.split(".")[0] + "_gs.mp4", video, fps=30)
    video = render_utils.render_video(output['mesh'][0])['normal']
    imageio.mimsave(output_path.split(".")[0] + "_mesh.mp4", video, fps=30)
    # GLB files can be extracted from the output
    glb = postprocessing_utils.to_glb(
        output['gaussian'][0],
        output['mesh'][0],
        # Optional parameters
        simplify=0.1,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(output_path)

    

if __name__ == "__main__":

    parser = ArgumentParser("Get masks and mesh extracts of objects within a scene")

    # parser.add_argument("--source_path", "-s", required=True, type=str)
    # parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--num_images", type=int, default=10)

    pipeline = get_pipeline()
    args = parser.parse_args()

    args.source_path = "/store/real/ehliang/r2c2r_blender_data_2/r2c2r_data/test/StorageFurniture/44781/loop_0/link_2_rgb"
    args.output_path = "outputs/drawer4.glb"
    args.file_path = "/store/real/ehliang/r2c2r_blender_data_2/r2c2r_data/test/StorageFurniture/44781/loop_0/link_2.ply"

    # source_path = "/store/real/ehliang/r2c2r_blender_data/r2c2r_data/test/StorageFurniture/44781/loop_0/link_3_rgb"
    # model_path = "outputs/merged_example.glb"
    # pointcloud_path = "/store/real/ehliang/r2c2r_blender_data/r2c2r_data/test/StorageFurniture/44781/loop_0/link_3.ply"

    generate(args, pipeline)
    
