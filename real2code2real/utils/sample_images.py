#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

def numeric_sort_key(filename: str) -> int:
    # strip extension and convert to int; assumes names like "001.png" or "23.png"
    return int(Path(filename).stem)

def sample_indices(n_items: int, n_samples: int) -> list[int]:
    """
    Uniformly sample n_samples indices from 0..n_items-1 inclusive,
    in ascending order.
    """
    if n_items < n_samples:
        raise ValueError(f"Not enough items ({n_items}) to sample {n_samples}.")
    # include both endpoints 0 and n_items-1
    return [int(round(i * (n_items - 1) / (n_samples - 1))) for i in range(n_samples)]

def process_object_folder(input_obj_dir: Path, output_obj_dir: Path, n_samples: int = 20):
    imgs_dir = input_obj_dir / "images"
    if not imgs_dir.is_dir():
        print(f"Skipping {input_obj_dir}: no 'images/' subdir")
        return

    # List and sort all PNGs by numeric basename
    png_files = sorted(
        [f.name for f in imgs_dir.iterdir() if f.suffix.lower() == ".png"],
        key=numeric_sort_key
    )
    total = len(png_files)
    if total < n_samples:
        print(f"Warning: only {total} images in {imgs_dir}, fewer than {n_samples}.")
        indices = list(range(total))
    else:
        indices = sample_indices(total, n_samples)

    # Ensure target subdirs exist
    gen_state_dir = output_obj_dir / "generation_state"
    state1_dir    = output_obj_dir / "state_1"
    gen_state_dir.mkdir(parents=True, exist_ok=True)
    state1_dir   .mkdir(parents=True, exist_ok=True)

    # Copy
    for idx in indices:
        fname = png_files[idx]
        src = imgs_dir / fname
        shutil.copy2(src, gen_state_dir / fname)
        shutil.copy2(src, state1_dir    / fname)
    print(f"Copied {len(indices)} images from {imgs_dir} to {output_obj_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Uniformly sample 20 PNGs from each input object_n/images and copy to output object_n/{generation_state,state_1}"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path containing input object_n folders"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path containing output object_n folders"
    )
    parser.add_argument(
        "--samples", "-k",
        type=int,
        default=20,
        help="Number of images to sample (default: 20)"
    )
    args = parser.parse_args()

    # For each subfolder named object_*
    for input_obj in sorted(args.input_dir.iterdir()):
        if input_obj.is_dir() and input_obj.name.startswith("object_"):
            output_obj = args.output_dir / input_obj.name
            if not output_obj.is_dir():
                print(f"Output folder {output_obj} does not exist. Creating it.")
                output_obj.mkdir(parents=True, exist_ok=True)
            process_object_folder(input_obj, output_obj, n_samples=args.samples)

if __name__ == "__main__":
    main()
