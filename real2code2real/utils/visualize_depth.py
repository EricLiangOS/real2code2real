# filepath: /home/ehliang/real2code2real/test5.py
import OpenEXR
import Imath
import numpy as np
import array

def read_exr_to_array(file_path):
    """
    Reads an .exr file from the given file_path using OpenEXR and returns a NumPy array.
    Also counts and prints the number of NaN pixels in the file.
    """
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Use FLOAT pixel type
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # Determine the available channels
    channels = header['channels'].keys()
    if "R" in channels and "G" in channels and "B" in channels:
        # Assume it's a color image with channels R, G, B
        r_str = exr_file.channel('R', FLOAT)
        g_str = exr_file.channel('G', FLOAT)
        b_str = exr_file.channel('B', FLOAT)
        # Convert byte strings to numpy arrays
        r = np.frombuffer(r_str, dtype=np.float32).reshape(size[1], size[0])
        g = np.frombuffer(g_str, dtype=np.float32).reshape(size[1], size[0])
        b = np.frombuffer(b_str, dtype=np.float32).reshape(size[1], size[0])
        # Stack channels into an image array
        img = np.stack((r, g, b), axis=-1)
    else:
        # Fallback for single channel or different channel names; using the first available channel
        channel_name = list(channels)[0]
        channel_str = exr_file.channel(channel_name, FLOAT)
        img = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], size[0])
    
    # Count the number of NaN pixels
    nan_count = np.isnan(img).sum()
    print("Number of NaN pixels:", nan_count)

    return img

# Example usage:
if __name__ == "__main__":
    exr_file_path = "/store/real/ehliang/data/basement_kitchen/kitchen_interaction_3/input_depth/110.exr"
    image_array = read_exr_to_array(exr_file_path)
    # ...existing code...