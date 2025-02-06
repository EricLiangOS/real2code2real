import os

input_dir = "/store/real/ehliang/multiview_data/kitchen_static_5/20_img"
num_objects = 8
num_states = 1

os.makedirs(input_dir, exist_ok=True)

for obj in range(1, num_objects + 1):
    object_dir = os.path.join(input_dir, f"object_{obj}")
    os.makedirs(object_dir, exist_ok=True)

    generation_dir = os.path.join(object_dir, "generation_state")
    os.makedirs(generation_dir, exist_ok=True)

    for state in range(1, num_states + 1):
        state_dir = os.path.join(object_dir, f"state_{state}")
        os.makedirs(state_dir, exist_ok=True)

