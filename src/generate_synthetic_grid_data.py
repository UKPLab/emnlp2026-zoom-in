import math
import os
import json
import hashlib
import shutil
import uuid
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageFilter
import random
from pathlib import Path
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

def _load_grid_font(cell_size):
    """Load a TrueType font for the cell index labels, falling back to PIL's
    built-in bitmap font if none of the candidate fonts are available."""
    font_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arial.ttf"]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size=cell_size // 8)
        except Exception:
            continue
    print("Warning: no TrueType font found, falling back to PIL default font.")
    return ImageFont.load_default()


def generate_grid_from_jsonl(
        jsonl_path,
        image_source_dir,
        output_path,
        big_image_size=1024,
        grid_size=4,
        num_samples_class_0=4,
        num_output_images=5,
        class_0_name="Muffin",  # Adjust this to match your JSONL class name
        seed=42,
):
    """
    Reads from JSONL and loads images on-demand to create grid composites.

    Reproducible and resumable: every output grid is seeded independently with
    ``seed + img_idx``, so already-rendered grids can be skipped on a restart
    without desynchronising the RNG for the remaining ones. ``test.jsonl`` is
    written atomically at the very end and therefore doubles as the
    "config complete" marker used by :func:`make_grid_data`.
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

    # Sort the per-class file lists so that a fixed seed always maps to the same
    # physical image regardless of the order the JSONL happened to be built in.
    class_map[0].sort()
    class_map[1].sort()

    # Setup output directories
    images_out_dir = os.path.join(output_path, "images")
    os.makedirs(images_out_dir, exist_ok=True)

    total_cells = grid_size * grid_size
    cell_size = big_image_size // grid_size

    font = _load_grid_font(cell_size)

    all_metadata = []

    for img_idx in range(num_output_images):
        # Per-image seed -> each grid is independently reproducible, which makes
        # the (multi-hour) generation safe to resume at any point.
        random.seed(seed + img_idx)
        np.random.seed((seed + img_idx) % (2 ** 32))

        # Determine cell labels for this big image first: this is cheap and is
        # needed to (re)build the metadata even when the image already exists.
        cell_indices = list(range(total_cells))
        if total_cells == 1:
            if img_idx < num_output_images // 2:
                class_0_cells = []
            else:
                class_0_cells = [0]
        else:
            class_0_cells = random.sample(cell_indices, min(num_samples_class_0, total_cells))

        image_labels_map = {i: (0 if i in class_0_cells else 1) for i in range(total_cells)}

        out_name = f"grid_{img_idx}.png"
        out_path = os.path.join(images_out_dir, out_name)
        all_metadata.append({"image": out_name, "labels": image_labels_map})

        # Resume: a completed grid from a previous run is kept as-is. Grids are
        # written atomically below, so any existing file is guaranteed complete.
        if os.path.exists(out_path):
            continue

        big_img = Image.new('RGB', (big_image_size, big_image_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(big_img)

        for i in range(total_cells):
            #print(f"start with cell {i}")
            row, col = divmod(i, grid_size)
            current_label = image_labels_map[i]

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

        # Draw Grid Lines
        for line in range(0, big_image_size + 1, cell_size):
            draw.line([(line, 0), (line, big_image_size)], fill="black", width=3)
            draw.line([(0, line), (big_image_size, line)], fill="black", width=3)

        # Atomic save: write to a temp file then rename, so an interrupted run
        # never leaves a truncated PNG that a resume would mistake for complete.
        tmp_path = out_path + ".tmp"
        big_img.save(tmp_path, format="PNG")  # explicit: temp name has no .png ext
        os.replace(tmp_path, out_path)

    # Atomic write of the metadata; its presence marks the config as complete.
    tmp_meta = os.path.join(output_path, "test.jsonl.tmp")
    with open(tmp_meta, "w") as f:
        json.dump(all_metadata, f, indent=4)
    os.replace(tmp_meta, os.path.join(output_path, "test.jsonl"))


def initial_preprocess(download_dir:str):
    new_image_path = os.path.join(download_dir, "images")
    if os.path.exists(new_image_path):
        shutil.rmtree(new_image_path)
    os.makedirs(new_image_path)
    if os.path.exists(os.path.join(download_dir, "test.jsonl")):
        os.remove(os.path.join(download_dir, "test.jsonl"))
    dataset = []

    classes = ["Chihuahua", "Muffin"]
    idx = 0
    for cls in classes:

        print(cls)
        cls_path = os.path.join(download_dir, cls.lower())
        # sorted() so the img{idx}.jpg naming is deterministic across machines
        # (os.listdir order is filesystem-dependent).
        for img_path in sorted(os.listdir(cls_path)):
            full_image_path = os.path.join(cls_path, img_path)
            print(img_path)
            #image_name = uuid.uuid4().hex + ".jpg"
            image_name = f"img{idx}.jpg"
            shutil.copy(full_image_path, os.path.join(new_image_path, image_name))
            dataset.append({"class": cls, "image": image_name})
            idx += 1

    save_as_jsonl(dataset, os.path.join(download_dir, "test.jsonl"))

def make_grid_data(configs:list, download_dir: str, save_path_prefix: str, total_images: int, base_seed:int = 42):
    for config in tqdm(configs):
        seed = get_seed(base_seed, config)
        output_path = os.path.join(save_path_prefix, f"grid_pixels_{config['big_image_size']}_gridsize_{config['grid_size']}_samples_class_0_{config['num_samples_class_0']}")
        # test.jsonl is written (atomically) only once a config is fully done, so
        # its presence is the completion marker. A merely-existing output_path is
        # NOT enough: it may hold a partially generated config from a killed run.
        if os.path.exists(os.path.join(output_path, "test.jsonl")):
            print(f"Skipping completed {output_path}")
            continue
        os.makedirs(output_path, exist_ok=True)
        print(f"generating data for {config}")
        generate_grid_from_jsonl(
            jsonl_path=os.path.join(download_dir, "test.jsonl"),
            image_source_dir=os.path.join(download_dir, "images"),
            output_path=output_path,
            big_image_size=config['big_image_size'],
            grid_size=config['grid_size'],
            num_samples_class_0=config['num_samples_class_0'],
            num_output_images=total_images,
            class_0_name="Muffin",  # Adjust this to match your JSONL class name
            seed=seed,
        )

# Prepended to every M&C question so the emitted JSONL is already in the
# canonical format (problem/solution) consumed by open_r1.preprocess_data,
# i.e. the paper's own dataset needs no separate converter step.
GRID_PREAMBLE = ("In the image you see a grid, whose cells are numbered from left to "
                 "right and top to bottom. In each cell, the cell's index is printed "
                 "in the upper left corner. ")


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
                    entry["problem"] = GRID_PREAMBLE + query
                    entry["solution"] = correct_tag
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

                    entry["problem"] = GRID_PREAMBLE + query
                    entry["solution"] = outliers[0]
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
    # NOTE: the built-in hash() is salted per-process (PYTHONHASHSEED) and would
    # give a different seed on every run/machine, silently breaking --base_seed.
    # Use a stable cryptographic hash so seeds are fully reproducible everywhere.
    config_hash = int(hashlib.sha256(config_str.encode("utf-8")).hexdigest(), 16)
    config_seed = base_seed + config_hash % offset
    return config_seed

# If the primary host is unreachable, images are fetched from this Wayback
# Machine snapshot instead (prepend it to the original URL).
WAYBACK_PREFIX = "https://web.archive.org/web/20240417234016im_/"


def _curl_to_file(url: str, output_path, follow_redirects: bool = False) -> None:
    """Download a single URL to ``output_path`` with curl, raising on failure."""
    redirect_flags = ['-L'] if follow_redirects else ['--max-redirs', '0']
    subprocess.run(
        ['curl', '-f', *redirect_flags, '-o', str(output_path), url],
        check=True,
        capture_output=True,
        text=True,
    )


def download_images(images: list[dict], directory: str = ".") -> None:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        target_dir_for_class = target_dir / image["target"]
        target_dir_for_class.mkdir(parents=True, exist_ok=True)
        url = image["source"]

        filename = Path(urlparse(url).path).name or "image.png"
        output_path = target_dir_for_class / filename

        # Resume: skip files already fetched by a previous run.
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"↺ Skipping existing: {output_path}")
            continue

        # Try the primary source first, then the Wayback Machine backup mirror.
        # (Wayback may redirect to the nearest snapshot, so allow redirects there.)
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        sources = [(url, False), (WAYBACK_PREFIX + url, True)]
        for src_url, follow_redirects in sources:
            try:
                _curl_to_file(src_url, tmp_path, follow_redirects=follow_redirects)
                # Atomic: the final file only appears once fully downloaded.
                os.replace(tmp_path, output_path)
                print(f"✓ Downloaded: {src_url} -> {output_path}")
                break
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to download {src_url}: {e}")
                if e.stderr:
                    print(f"  stderr: {e.stderr}")
        else:
            # Neither source worked -> clean up any partial temp file and abort.
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                f"Could not download {url} from the primary source or the "
                f"Wayback Machine backup ({WAYBACK_PREFIX + url})."
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic grid data for M&C dataset')
    parser.add_argument('--download_dir', type=str, default=None,
                        help='Path to the raw dataset directory')
    parser.add_argument('--save_path_prefix', type=str, default=None,
                        help='Path prefix where the generated splits will be saved')
    parser.add_argument('--base_seed', type=int, default=42, help='Base seed for reproducibility')
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
        # Build the URL with a plain f-string (os.path.join would insert
        # backslashes on Windows); target is the per-class subfolder name.
        image_list.append({"source": f"{image_base_url}test{i}.png",
                           "target": cls})

    download_images(image_list, download_dir)

    initial_preprocess(download_dir)

    total_images = 100

    configs = []
    for bis in [1,2,4,8]:
        bis = bis * 1024
        for gs in [1,2,4,8, 16]:
            configs.append({"big_image_size": bis, "grid_size": gs, "num_samples_class_0": 1})
            if gs > 1:
               configs.append({"big_image_size": bis, "grid_size": gs, "num_samples_class_0": int(gs**2 / 2)})

    make_grid_data(configs, download_dir=download_dir, total_images=total_images, save_path_prefix=save_path_prefix, base_seed=args.base_seed)

    make_vqa_dataset(configs, save_path_prefix=save_path_prefix,
                     variant="both",
                     total_images=total_images,
                     base_seed=args.base_seed)


