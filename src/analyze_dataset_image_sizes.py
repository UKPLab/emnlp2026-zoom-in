from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SizeStats:
    count: int
    mean_width: float
    mean_height: float
    mean_pixels: float
    median_width: float
    median_height: float
    median_pixels: float


def iter_image_files(directory: Path) -> Iterable[Path]:
    """
    Yields files that are likely images (by extension). This is a quick filter;
    Pillow will do the real validation when opening.
    """
    exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp",
        ".ppm", ".pgm", ".pbm", ".pnm", ".ico",
    }
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def collect_image_sizes(directory: Path) -> list[tuple[int, int]]:
    """
    Returns a list of (width, height) for all readable images under `directory`.
    Skips unreadable/corrupt files.
    """
    sizes: list[tuple[int, int]] = []

    for path in iter_image_files(directory):
        try:
            with Image.open(path) as im:
                w, h = im.size
            if w > 0 and h > 0:
                sizes.append((int(w), int(h)))
        except Exception:
            # Skip files Pillow can't open/identify
            continue

    return sizes


def compute_stats(sizes: list[tuple[int, int]]) -> SizeStats:
    if not sizes:
        raise ValueError("No readable images found; cannot compute stats.")

    widths = np.array([w for w, _ in sizes], dtype=np.float64)
    heights = np.array([h for _, h in sizes], dtype=np.float64)
    pixels = widths * heights

    return SizeStats(
        count=len(sizes),
        mean_width=float(widths.mean()),
        mean_height=float(heights.mean()),
        mean_pixels=float(pixels.mean()),
        median_width=float(np.median(widths)),
        median_height=float(np.median(heights)),
        median_pixels=float(np.median(pixels)),
    )


def main() -> None:
    prefix_path = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets"
    dataset_paths = [
        "pixel_reasoner/eval/V_Star/",
        "pixel_reasoner/eval/Infographics_VQA",
        "focusreason/HR_Bench_4k/",
        "focusreason/HR_Bench_8k/",
        "focusreason/MME-RealWorld/"
    ]
    for dataset_path in dataset_paths:


        full_path = os.path.join(prefix_path, dataset_path, "images")

        # Set this to the directory you want to analyze:
        images_dir = Path(full_path).expanduser().resolve()

        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist or is not a directory: {images_dir}")

        sizes = collect_image_sizes(images_dir)
        stats = compute_stats(sizes)

        print(f"Directory: {images_dir}")
        print(f"Images counted: {stats.count}")
        print("--- Mean ---")
        print(f"Width  (px): {stats.mean_width:.2f}")
        print(f"Height (px): {stats.mean_height:.2f}")
        print(f"Pixels (w*h): {stats.mean_pixels:.2f}")
        print("--- Median ---")
        print(f"Width  (px): {stats.median_width:.2f}")
        print(f"Height (px): {stats.median_height:.2f}")
        print(f"Pixels (w*h): {stats.median_pixels:.2f}")


if __name__ == "__main__":
    main()