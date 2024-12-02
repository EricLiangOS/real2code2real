from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import logging

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
from argparse import ArgumentParser
from tqdm import tqdm
import shutil

def convert_files(source_path, resize = False, camera = "OPENCV", colmap_executable = "", magick_executable = "", use_gpu = 1):
    colmap_command = '"{}"'.format(colmap_executable) if len(colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(magick_executable) if len(magick_executable) > 0 else "magick"

    os.makedirs(source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + source_path + "/distorted/database.db \
        --image_path " + source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + source_path + "/distorted/database.db \
        --image_path "  + source_path + "/input \
        --output_path "  + source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + source_path + "/input \
        --input_path " + source_path + "/distorted/sparse/0 \
        --output_path " + source_path + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    print("Done.")


def undistort_image(image_array, device_calib):
    sensor_name = "camera-rgb"
    src_calib = device_calib.get_camera_calib(sensor_name)
    dst_calib = calibration.get_linear_camera_calibration(325, 325, 150, sensor_name)

    undistorted_image_array = calibration.distort_by_calibration(image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR)
    
    return undistorted_image_array

def get_images(vrs_dir, output_dir, dataset_size):
    
    os.makedirs(output_dir, exist_ok=True)

    provider = data_provider.create_vrs_data_provider(vrs_dir)
    device_calib = provider.get_device_calibration()

    rgb_stream_id = StreamId('214-1')
    time_domain = TimeDomain.DEVICE_TIME
    option = TimeQueryOptions.CLOSEST

    start_time = provider.get_first_time_ns(rgb_stream_id, time_domain)
    end_time = provider.get_last_time_ns(rgb_stream_id, time_domain)

    # Collect equal samples of the recorded video
    sample_timestamps = np.linspace(start_time, end_time, dataset_size)
    for i, sample in enumerate(tqdm(sample_timestamps)):
        image_tuple = provider.get_image_data_by_time_ns(rgb_stream_id, int(sample), time_domain, option)
        image_array = image_tuple[0].to_numpy_array()

        undistorted_image_array = undistort_image(image_array, device_calib)
        undistorted_image_array = np.rot90(undistorted_image_array, 3)

        image = Image.fromarray(undistorted_image_array)

        image.save(os.path.join(output_dir, f"{i}.jpg"))


if __name__ == "__main__":
    parser = ArgumentParser("Input directory for egocentric vrs file")
    parser.add_argument("--source_path", "-s", type = str, required=True)
    parser.add_argument("--dataset_size", default=1100, type = int)

    args = parser.parse_args()
    vrs_dir = args.source_path
    output_dir = os.path.join(os.path.dirname(vrs_dir), "input")


    get_images(vrs_dir, output_dir, args.dataset_size)
    convert_files(os.path.dirname(vrs_dir))

