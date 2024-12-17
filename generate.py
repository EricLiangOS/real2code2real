
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from argparse import ArgumentParser
from PIL import Image
from real2code2real.mesh_extraction import PointCloudTo3DPipeline
from real2code2real.utils.generate_utils import get_object
from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils
import imageio

def get_pipeline():
    pipeline = PointCloudTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")

    return pipeline

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

def generate(args, pipeline):
    structure_path = args.structure_path
    multiview_path = args.multiview_path
    output_path = args.output_path

    object_name = os.path.basename(os.path.normpath(output_path))

    os.makedirs(output_path, exist_ok=True)

    pipeline.cuda()

    images = []

    frame_names = [
        p for p in os.listdir(multiview_path)
    ]
    # .split("_")[1])
    frame_names.sort(key=lambda p: get_number(os.path.splitext(p)[0]))

    # Load an image
    frequency = len(frame_names)/args.num_images

    for i in range(args.num_images):
        image_name = frame_names[int(i * frequency)]
        print(image_name)
        images.append(Image.open(os.path.join(multiview_path, image_name)))


    # Run the pipeline
    output, mapping = pipeline.run(
        images,
        # Optional parameters
        seed=1,
        input_path=structure_path,
    )

    video = render_utils.render_video(output['gaussian'][0])['color']
    imageio.mimsave(os.path.join(output_path, "sample_gs.mp4"), video, fps=30)

    video = render_utils.render_video(output['mesh'][0])['normal']
    imageio.mimsave(os.path.join(output_path, "sample_mesh.mp4"), video, fps=30)
    get_object(output, output_path, object_name, mapping)

    

if __name__ == "__main__":

    parser = ArgumentParser("Get masks and mesh extracts of objects within a scene")

    parser.add_argument("--structure_path", "-s", required=True, type=str)
    parser.add_argument("--multiview_path", "-v", required=True, type=str)
    parser.add_argument("--output_path", "-o", required=True, type=str)
    parser.add_argument("--num_images", type=int, default=10)

    pipeline = get_pipeline()
    args = parser.parse_args()

    args.num_images = max(args.num_images, len(os.listdir(args.multiview_path)))

    # source_path = "/store/real/ehliang/r2c2r_blender_data/r2c2r_data/test/StorageFurniture/44781/loop_0/link_3_rgb"
    # model_path = "outputs/merged_example.glb"
    # point_cloud_path = "/store/real/ehliang/r2c2r_blender_data/r2c2r_data/test/StorageFurniture/44781/loop_0/link_3.ply"

    generate(args, pipeline)
    
