import os

# Specify the directory containing the .jpg files
target_directory = "/home/ehliang/data/bottle_rgb_3/images"  # Change this to the desired directory

# Get a list of all files in the directory
all_files = os.listdir(target_directory)

# Filter only .jpg files
jpg_files = [file for file in all_files if file.endswith(".jpg")]

# Delete each .jpg file
for file in jpg_files:
    file_path = os.path.join(target_directory, file)
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("All .jpg files have been deleted.")