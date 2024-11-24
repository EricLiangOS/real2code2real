import os
from PIL import Image
import re
from argparse import ArgumentParser

def copy_dataset(input_directory, output_directory, save_frequency):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get a sorted list of all image files in the input directory
    # Get all image files in the folder
    frame_names = [
        p for p in os.listdir(input_directory)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Copy and rename every nth image to the output directory
    renamed_counter = 0
    for index, file in enumerate(frame_names):
        if index % save_frequency == 0:
            src_path = os.path.join(input_directory, file)
            dst_path = os.path.join(output_directory, f"{renamed_counter}.jpg")
            
            # Open the image and save it without loss
            img = Image.open(src_path)
            img.save(dst_path, quality=100)  # High-quality, lossless save
            renamed_counter += 1

    print(f"Made a copy and saved every {save_frequency}th image to '{output_directory}' with lossless quality.")

if __name__ == "__main__":
    parser = ArgumentParser("Copy every nth image from a dataset into a new directory")
    parser.add_argument("--original_dir", "-o", required=True, type=str)
    parser.add_argument("--new_dir", "-n", required=True, type=str)
    parser.add_argument("--frequency", required = True, type = int)
    args = parser.parse_args()

    # Variables
    input_directory = args.original_dir  # "/home/ehliang/data/multi_object_1/rgb/"
    output_directory = args.new_dir #"/home/ehliang/data/multi_object_1/input"
    save_frequency = args.frequency  # Change this value as needed

    copy_dataset(input_directory, output_directory, save_frequency)
