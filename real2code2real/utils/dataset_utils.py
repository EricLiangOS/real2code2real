import os
import logging
from argparse import ArgumentParser
import shutil
from PIL import Image
import json

def convert_files(source_path, resize = False, camera = "OPENCV", colmap_executable = "", magick_executable = "", use_gpu = 1):
    colmap_command = '"{}"'.format(colmap_executable) if len(colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(magick_executable) if len(magick_executable) > 0 else "magick"

    colmap_path = source_path + "/colmap"

    os.makedirs(colmap_path, exist_ok=True)
    os.makedirs(colmap_path + "/distorted/sparse", exist_ok=True)
    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + colmap_path + "/distorted/database.db \
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
        --database_path " + colmap_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + colmap_path + "/distorted/database.db \
        --image_path "  + source_path + "/input \
        --output_path "  + colmap_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + source_path + "/input \
        --input_path " + colmap_path + "/distorted/sparse/0 \
        --output_path " + colmap_path  + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(colmap_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(colmap_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    print("Done.")

def copy_dataset(input_directory, output_directory, dataset_size):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get a sorted list of all image files in the input directory
    # Get all image files in the folder
    frame_names = [
        p for p in os.listdir(input_directory)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".exr"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    frame_correspondance = {}

    # Copy and rename every nth image to the output directory
    save_frequency = len(os.listdir(input_directory)) * 1.0/dataset_size
    for index in range(dataset_size):
        src_path = os.path.join(input_directory, frame_names[int(index * save_frequency)])
        dst_path = os.path.join(output_directory, f"{index}.jpg")
        frame_correspondance[index] = int(index * save_frequency)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    print(f"Made a copy of {dataset_size} files to '{output_directory}' with lossless quality.")

    return frame_correspondance

def rewrite_json(json_dir, output_dir, frame_correspondance, rewrite_categories):
    with open(json_dir, 'r') as file:
        data = json.load(file)
    
    new_json = {}

    for key in data:
        if key not in rewrite_categories:
            new_json[key] = data[key]
            continue

        new_values = []
        for new_frame in frame_correspondance:
            new_values.append(data[key][frame_correspondance[new_frame]])
        
        new_json[key] = new_values
    
    with open(output_dir, 'w') as file:
        json.dump(new_json, file)


def remove_jpegs(input_directory):
    # Get a list of all files in the directory
    all_files = os.listdir(input_directory)

    # Filter only .jpg files
    jpg_files = [file for file in all_files if file.endswith(".jpg")]

    # Delete each .jpg file
    for file in jpg_files:
        file_path = os.path.join(input_directory, file)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    print("All .jpg files have been deleted.")

def rename_images(input_directory):
    # Initial values for renaming
    start_number = 0
    extension = ".jpg"

    # Get all image files in the folder
    frame_names = [
        p for p in os.listdir(input_directory)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Rename each file
    for i, file in enumerate(frame_names):
        new_name = f"{start_number + i}{extension}"
        old_path = os.path.join(input_directory, file)
        new_path = os.path.join(input_directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")

if __name__ == "__main__":
    
    # This Python script is based on the shell converter script provided in the MipNerF 360 repository.
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action='store_true')
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)
    args = parser.parse_args()
    use_gpu = 1 if not args.no_gpu else 0

    convert_files(args.source_path, args.resize, args.camera, args.colmap_executable, args.magick_executable, use_gpu)