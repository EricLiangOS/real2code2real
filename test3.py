import open3d as o3d
import numpy as np
from PIL import Image
import trimesh
import os
import shutil

base_dir = "/store/real/ehliang/data/basement_kitchen/kitchen_interaction_5"
generation_num_images = 24
state_1_num_images = 4

output_dir = f"/store/real/ehliang/multiview_data/kitchen_static_5/20_img"
num_objects = 8

os.makedirs(output_dir, exist_ok=True)

for obj in range(1, num_objects + 1):
    object_dir = os.path.join(base_dir, f"object_{obj}", "images")
    object_output_dir = os.path.join(output_dir, f"object_{obj}", "generation_state")
    state_output_dir = os.path.join(output_dir, f"object_{obj}", "state_1")

    os.makedirs(object_output_dir, exist_ok=True)
    os.makedirs(state_output_dir, exist_ok=True)

    all_images = sorted(
        [f for f in os.listdir(object_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    if not all_images:
        print(f"No images found in {object_dir}")
        continue
    
    if generation_num_images == "all":
        generation_indices = range(1, len(all_images))
    else:
        generation_indices = np.linspace(1, len(all_images) - 1, generation_num_images, dtype=int)

    state_indices = np.linspace(1, len(all_images) - 1, state_1_num_images, dtype=int)

    for img_file in [all_images[i] for i in generation_indices]:
        shutil.copy2(os.path.join(object_dir, img_file), object_output_dir)

    for img_file in [all_images[i] for i in state_indices]:
        shutil.copy2(os.path.join(object_dir, img_file), state_output_dir)