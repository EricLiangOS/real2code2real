
import os, shutil
from PIL import Image


def copy_dataset(input_directory, output_directory, dataset_size):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get a sorted list of all image files in the input directory
    # Get all image files in the folder
    frame_names = [
        p for p in os.listdir(input_directory)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Copy and rename every nth image to the output directory
    save_frequency = len(os.listdir(input_directory)) * 1.0/dataset_size
    for index in range(dataset_size):
        src_path = os.path.join(input_directory, frame_names[int(index * save_frequency)])
        dst_path = os.path.join(output_directory, f"{index}.jpg")

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    print(f"Made a copy of {dataset_size} files to '{output_directory}' with lossless quality.")




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
        new_name = f"frame_{(start_number + i):05d}{extension}"
        old_path = os.path.join(input_directory, file)
        new_path = os.path.join(input_directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")

# dataset = "/store/real/ehliang/data/test_2/rgb"
# output = "/store/real/ehliang/data/test_2/desk/cam01"
# copy_dataset(dataset, output, 5)
# rename_images(output)

copy_dataset("/store/real/ehliang/data/new_kitchen_3/object_1/images", "/store/real/ehliang/data/sample/object_1", 30)
copy_dataset("/store/real/ehliang/data/new_kitchen_3/object_2/images", "/store/real/ehliang/data/sample/object_2", 30)
copy_dataset("/store/real/ehliang/data/new_kitchen_3/object_3/images", "/store/real/ehliang/data/sample/object_3", 30)