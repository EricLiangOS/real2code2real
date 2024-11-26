import os

# Path to the folder containing images
folder_path = "/home/ehliang/data/multi_object_1/object_2/masks"  # Replace with your folder path

# Initial values for renaming
start_number = 0
extension = ".jpg"

# Get all image files in the folder
frame_names = [
    p for p in os.listdir(folder_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Rename each file
for i, file in enumerate(frame_names):
    new_name = f"{start_number + i}{extension}"
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed: {file} -> {new_name}")
