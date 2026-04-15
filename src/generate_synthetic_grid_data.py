import math
import os
import json
import shutil
import sys
import uuid
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
import random
from pathlib import Path
import requests
import argparse
from urllib.parse import urlparse
import subprocess


class ImageAugmenter:
    @staticmethod
    def safe_rotate(img):
        """
        Rotates image by a random angle and crops it to avoid black borders.
        """
        angle = random.uniform(0, 360)
        # Convert angle to radians for calculation
        phi = math.radians(angle % 90)
        if (angle // 90) % 2 != 0:
            phi = math.radians(90 - (angle % 90))

        # Calculate the side length of the largest square that fits
        # inside the rotated square to avoid black borders.
        s = img.width
        new_side = s / (math.sin(phi) + math.cos(phi))

        # Rotate and then crop to the safe center square
        rotated = img.rotate(angle, resample=Image.Resampling.BICUBIC)

        left = (s - new_side) / 2
        top = (s - new_side) / 2
        right = (s + new_side) / 2
        bottom = (s + new_side) / 2

        return rotated.crop((left, top, right, bottom))

    @staticmethod
    def apply(img):
        # 1. Flips
        if random.random() > 0.5:
            img = ImageOps.mirror(img)
        if random.random() > 0.5:
            img = ImageOps.flip(img)

        # 2. Rotations (90, 180, 270)

        img = ImageAugmenter.safe_rotate(img)

        # 3. Color Jittering (Brightness/Contrast)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))

        # 4. Slight Blur (to simulate focus variations)
        if random.random() > 0.8:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        return img

def save_as_jsonl(data_list, output_filename):
    """
    Saves a list of dictionaries into a .jsonl file.
    Each dictionary becomes exactly one line.
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        for entry in data_list:
            # json.dumps converts a dict to a string
            json_record = json.dumps(entry, ensure_ascii=False)
            f.write(json_record + '\n')

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(map(json.loads, f.readlines()))

def generate_grid_from_jsonl(
        jsonl_path,
        image_source_dir,
        output_path,
        big_image_size=1024,
        grid_size=4,
        num_samples_class_0=4,
        num_output_images=5,
        class_0_name="Muffin"  # Adjust this to match your JSONL class name
):
    """
    Reads from JSONL and loads images on-demand to create grid composites.
    """
    # 1. Index the dataset by class without loading images
    class_map = {0: [], 1: []}

    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Map the class_name to binary 0 or 1
            label = 0 if str(entry['class']) == str(class_0_name) else 1
            class_map[label].append(entry['image'])

    if not class_map[0] or not class_map[1]:
        raise ValueError("Could not find both classes in JSONL. Check class_0_name.")

    # Setup output directories
    images_out_dir = os.path.join(output_path, "images")
    os.makedirs(images_out_dir, exist_ok=True)

    total_cells = grid_size * grid_size
    cell_size = big_image_size // grid_size

    font_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arial.ttf"]
    font = ImageFont.load_default()
    for font_path in font_paths:
        # Try to load a larger font; fallback to default if not found
        try:
            # Standard Windows path, might need adjustment for other OS
            font = ImageFont.truetype(font_path, size=cell_size // 8)
            break
        except:
            print(f"Warning: Could not load font {font_path}.")
            continue


    all_metadata = []

    for img_idx in range(num_output_images):
        #print(f"start with image {img_idx}")
        big_img = Image.new('RGB', (big_image_size, big_image_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(big_img)

        # Determine cell labels for this big image
        cell_indices = list(range(total_cells))
        if total_cells == 1:
            if img_idx < num_output_images//2:
                class_0_cells = []
            else:
                class_0_cells = [0]
        else:
            class_0_cells = random.sample(cell_indices, min(num_samples_class_0, total_cells))

        image_labels_map = {}

        for i in range(total_cells):
            #print(f"start with cell {i}")
            row, col = divmod(i, grid_size)
            current_label = 0 if i in class_0_cells else 1

            # Lazy Load: Only pick the filename and open the image now
            img_filename = random.choice(class_map[current_label])
            img_path = os.path.join(image_source_dir, img_filename)

            try:
                with (Image.open(img_path) as sample_img):
                    sample_img = sample_img.convert('RGB')
                    sample_img = ImageAugmenter.apply(sample_img)
                    sample_img = sample_img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                    big_img.paste(sample_img, (col * cell_size, row * cell_size))
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
                # Optional: draw a gray box if image fails to load
                draw.rectangle([col * cell_size, row * cell_size, (col + 1) * cell_size, (row + 1) * cell_size],
                               fill="gray")

            # Draw Index with Background Box
            text_str = str(i)
            text_pos = (col * cell_size + 10, row * cell_size + 10)
            bbox = draw.textbbox(text_pos, text_str, font=font)
            draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill="white")
            draw.text(text_pos, text_str, fill="black", font=font)

            image_labels_map[i] = current_label

        # Draw Grid Lines
        for line in range(0, big_image_size + 1, cell_size):
            draw.line([(line, 0), (line, big_image_size)], fill="black", width=3)
            draw.line([(0, line), (big_image_size, line)], fill="black", width=3)

        # Save
        out_name = f"grid_{img_idx}.png"
        big_img.save(os.path.join(images_out_dir, out_name))
        all_metadata.append({"image": out_name, "labels": image_labels_map})

    with open(os.path.join(output_path, "test.jsonl"), "w") as f:
        json.dump(all_metadata, f, indent=4)


def initial_preprocess(download_dir:str):
    new_image_path = os.path.join(download_dir, "images")
    if os.path.exists(new_image_path):
        shutil.rmtree(new_image_path)
    os.makedirs(new_image_path)
    if os.path.exists(os.path.join(download_dir, "test.jsonl")):
        os.remove(os.path.join(download_dir, "test.jsonl"))
    dataset = []

    classes = ["Chihuahua", "Muffin"]
    for cls in classes:

        print(cls)
        cls_path = os.path.join(download_dir, cls.lower())
        for img_path in os.listdir(cls_path):
            full_image_path = os.path.join(cls_path, img_path)
            print(img_path)
            image_name = uuid.uuid4().hex + ".jpg"
            shutil.copy(full_image_path, os.path.join(new_image_path, image_name))
            dataset.append({"class": cls, "image": image_name})

    save_as_jsonl(dataset, os.path.join(download_dir, "test.jsonl"))

def make_grid_data(configs:list, download_dir: str, save_path_prefix: str, total_images: int, base_seed:int = 42):
    for config in tqdm(configs):
        seed = get_seed(base_seed, config)
        random.seed(seed)
        np.random.seed(seed)
        print(f"generating data for {config}")
        output_path = os.path.join(save_path_prefix, f"grid_pixels_{config['big_image_size']}_gridsize_{config['grid_size']}_samples_class_0_{config['num_samples_class_0']}")
        if os.path.exists(output_path):
            print(f"Skipping {output_path}")
            continue
        os.makedirs(output_path, exist_ok=True)
        generate_grid_from_jsonl(
            jsonl_path=os.path.join(download_dir, "test.jsonl"),
            image_source_dir=os.path.join(download_dir, "images"),
            output_path=os.path.join(save_path_prefix, output_path),
            big_image_size=config['big_image_size'],
            grid_size=config['grid_size'],
            num_samples_class_0=config['num_samples_class_0'],
            num_output_images=total_images,
            class_0_name="Muffin"  # Adjust this to match your JSONL class name
        )

def make_vqa_dataset(configs: list, save_path_prefix: str, variant:str, total_images: int, base_seed:int=42):
    if variant == "both":
        variants = ["single_cell_query", "find_outlier"]
    elif variant in ["single_cell_query", "find_outlier"]:
        variants = [variant]
    else:
        raise ValueError(f"variant {variant} is not supported")
    for variant in variants:
        for config in configs:
            seed = get_seed(base_seed, config)
            random.seed(seed)
            np.random.seed(seed)
            full_path = os.path.join(save_path_prefix, f"grid_pixels_{config['big_image_size']}_gridsize_{config['grid_size']}_samples_class_0_{config['num_samples_class_0']}")
            save_path = os.path.join(full_path, f"{variant}.jsonl")
            if os.path.exists(save_path):
                print(f"already exists {save_path} for {config}, skipped")
                continue
            data = json.load(open(os.path.join(full_path, "test.jsonl"), "r"))
            if variant == "single_cell_query":
                if config["num_samples_class_0"] == 1 and config["grid_size"] != 1:
                    print(f"skipped {config} for {variant} because it is find outlier distribution")
                    continue
                for idx, entry in enumerate(data):
                    if config['grid_size'] == 1:
                        true_label = entry["labels"]["0"]
                        if true_label == 0:
                            correct_answer = "Muffin"
                        elif true_label == 1:
                            correct_answer = "Chihuahua"
                        else:
                            raise ValueError(f"expected label 0 or 1, but found {true_label}")
                        chosen_label = 0
                    else:
                        if idx < int(total_images * 0.5):
                            true_label = 0
                            correct_answer = "Muffin"
                        else:
                            true_label = 1
                            correct_answer = "Chihuahua"

                        labels = entry['labels'].values()
                        true_labels = []
                        for idx, label in enumerate(labels):
                            if label == true_label:
                                true_labels.append(idx)
                        chosen_label = random.choice(true_labels)

                    if random.random() > 0.5:
                        option_A = "Muffin"
                        option_B = "Chihuahua"
                    else:
                        option_A = "Chihuahua"
                        option_B = "Muffin"
                    if correct_answer == option_A:
                        correct_tag = "A"
                    elif correct_answer == option_B:
                        correct_tag = "B"
                    else:
                        raise ValueError(f"correct answer {correct_answer} not in options {option_A} and {option_B}")

                    query = (f"Which object is in cell number {chosen_label}?\n(A) {option_A}"
                                                        f"\n(B) {option_B}"
                                               f"\nAnswer with the option's letter from the given choices directly.")
                    entry["question"] = query
                    entry["answer"] = correct_tag
                    entry["bbox"] = get_bbox(chosen_label, config['grid_size'], config['big_image_size'] // config['grid_size'])
            elif variant == "find_outlier":
                if config["num_samples_class_0"] != 1:
                    print(f"skipped {config} for {variant} because num_samples_class_0 is not 1")
                    continue
                if config["grid_size"] == 1:
                    print(f"skipped {config} for {variant} because grid_size is 1")
                    continue
                for idx, entry in enumerate(data):
                    outliers = []
                    for k,v in entry["labels"].items():
                        if v == 0:
                            outliers.append(int(k))
                    if len(outliers) != 1:
                        raise ValueError(f"expected exactly one outlier, but found {len(outliers)}")



                    query = (f"In all cells except one you see a Chihuahua. Which cell does not contain a Chihuahua, but a Muffin?"
                             f"\nAnswer only with the cell number.")

                    entry["question"] = query
                    entry["answer"] = outliers[0]
                    entry["bbox"] = get_bbox(outliers[0], config['grid_size'], config['big_image_size'] // config['grid_size'])
            else:
                raise ValueError(f"variant {variant} not supported")


            save_as_jsonl(data, os.path.join(full_path, f"{variant}.jsonl"))

def get_bbox(correct_label:int, cells_per_row:int, cell_size: int) -> list[int]:

    col_position = correct_label % cells_per_row
    row_position = correct_label // cells_per_row

    bbox = [col_position * cell_size, row_position * cell_size,
            (col_position + 1) * cell_size, (row_position + 1) * cell_size]

    return bbox

def get_seed(base_seed:int, config:dict, offset=1000000):
    config_str = f"{config['big_image_size']}_{config['grid_size']}_{config['num_samples_class_0']}"
    config_seed = base_seed + hash(config_str) % offset
    return config_seed

def download_images(images: list[dict], directory: str = ".") -> None:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        target_dir_for_class = target_dir / image["target"]
        target_dir_for_class.mkdir(parents=True, exist_ok=True)
        url = image["source"]

        filename = Path(urlparse(url).path).name or "image.png"
        output_path = target_dir_for_class / filename

        # Use curl since you mentioned it works
        try:
            result = subprocess.run(
                ['curl', '-L', '-o', str(output_path), url],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ Downloaded: {url} -> {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {url}: {e}")
            print(f"  stderr: {e.stderr}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic grid data for M&C dataset')
    parser.add_argument('--download_dir', type=str, default=None,
                        help='Path to the raw dataset directory')
    parser.add_argument('--save_path_prefix', type=str, default=None,
                        help='Path prefix where the generated splits will be saved')
    args = parser.parse_args()

    # Set default paths if not provided via CLI
    download_dir = args.download_dir if args.download_dir else "/path/to/downloaded/mc_images"
    save_path_prefix = args.save_path_prefix if args.save_path_prefix else "/save/path"

    image_base_url = "https://www.topbots.com/downloads/code/vision/chihuahua_vs_muffin/"
    muffin_idxs = [1, 10, 11, 13, 16, 3, 5, 8]
    image_list = []
    for i in range(1, 17):
        if i in muffin_idxs:
            cls = "muffin"
        else:
            cls = "chihuahua"
        image_list.append({"source": os.path.join(image_base_url, f"test{i}.png"),
                           "target": os.path.join(download_dir, cls)})

    download_images(image_list, download_dir)

    initial_preprocess(download_dir)

    total_images = 100
    base_seed = 42

    configs = []
    for bis in [1,2,4,8]:
        bis = bis * 1024
        for gs in [1,2,4,8, 16]:
            configs.append({"big_image_size": bis, "grid_size": gs, "num_samples_class_0": 1})
            if gs > 1:
               configs.append({"big_image_size": bis, "grid_size": gs, "num_samples_class_0": int(gs**2 / 2)})

    make_grid_data(configs, download_dir=download_dir, total_images=total_images, save_path_prefix=save_path_prefix, base_seed=base_seed)

    make_vqa_dataset(configs, save_path_prefix=save_path_prefix,
                     variant="both",
                     total_images=total_images,
                     base_seed=base_seed)


