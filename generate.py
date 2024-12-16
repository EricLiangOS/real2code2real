
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

from argparse import ArgumentParser
from PIL import Image
from real2code2real.mesh_extraction import PointcloudTo3DPipeline
from real2code2real.utils.generate_utils import get_object
from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils

def get_pipeline():
    pipeline = PointcloudTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")

    return pipeline

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

def generate(args, pipeline):
    source_path = args.source_path
    file_path = args.file_path
    output_path = args.output_path

    object_name = os.path.basename(os.path.normpath(output_path))

    os.makedirs(output_path, exist_ok=True)

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
    output, mapping = pipeline.run(
        images,
        # Optional parameters
        seed=1,
        input_path=file_path,
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )

    get_object(output, output_path, object_name, mapping)

    

if __name__ == "__main__":

    parser = ArgumentParser("Get masks and mesh extracts of objects within a scene")

    # parser.add_argument("--source_path", "-s", required=True, type=str)
    # parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--num_images", type=int, default=10)

    pipeline = get_pipeline()
    args = parser.parse_args()

    args.source_path = "/store/real/ehliang/r2c2r_blender_data_2/r2c2r_data/test/StorageFurniture/44781/loop_0/link_3_rgb"
    args.output_path = "outputs/object_3_test"
    args.file_path = "/store/real/ehliang/r2c2r_blender_data_2/r2c2r_data/test/StorageFurniture/44781/loop_0/link_3.ply"

    # source_path = "/store/real/ehliang/r2c2r_blender_data/r2c2r_data/test/StorageFurniture/44781/loop_0/link_3_rgb"
    # model_path = "outputs/merged_example.glb"
    # pointcloud_path = "/store/real/ehliang/r2c2r_blender_data/r2c2r_data/test/StorageFurniture/44781/loop_0/link_3.ply"

    generate(args, pipeline)
    
