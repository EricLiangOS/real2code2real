import cv2, glob, os

def get_number(word):
    numbers = ""
    for char in word:
        if char.isnumeric():
            numbers += char
    
    return int(numbers)

def images_to_video(input_dir, output_path, fps=48):
    # Get all image files and sort them by number
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files = sorted(image_files, key=lambda x: get_number(os.path.basename(x)))
    
    if not image_files:
        raise ValueError(f"No jpg images found in {input_dir}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width = first_image.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each image to video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        out.write(frame)
    
    # Release resources
    out.release()

directories = [
    "/store/real/ehliang/data/basement_kitchen/kitchen_interaction_3/input", "/store/real/ehliang/data/basement_kitchen/kitchen_interaction_4/input", "/store/real/ehliang/data/basement_kitchen/kitchen_interaction_5/input", "/store/real/ehliang/data/fridge/fridge_static_1/input", "/store/real/ehliang/data/meeting/meeting_interactive_1/input"
]

output_names = [
    "input_kitchen_3.mp4", "input_kitchen_4.mp4", "input_kitchen_5.mp4", "input_fridge_1.mp4", "input_meeting_1.mp4"
]

for i in range(len(directories)):
    output = os.path.join(os.path.dirname(directories[i]), output_names[i])
    images_to_video(directories[i], output)