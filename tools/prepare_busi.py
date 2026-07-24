"""Prepare BUSI images and masks for training."""

import argparse
import re
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np


def _collect_images(category_dir: Path) -> List[Path]:
    return sorted(
        p for p in category_dir.glob("*.png")
        if "_mask" not in p.stem
    )


def _find_masks(image_path: Path) -> List[Path]:
    stem = image_path.stem
    parent = image_path.parent

    primary = parent / f"{stem}_mask.png"
    masks = [primary] if primary.exists() else []

    i = 1
    while True:
        extra = parent / f"{stem}_mask_{i}.png"
        if not extra.exists():
            break
        masks.append(extra)
        i += 1

    return masks


def _merge_masks(mask_paths: List[Path]) -> np.ndarray:
    merged = None
    for mp in mask_paths:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Cannot read mask: {mp}")
        m = (m > 0).astype(np.uint8)
        merged = m if merged is None else np.bitwise_or(merged, m)
    if merged is None:
        raise ValueError("No masks provided to _merge_masks")
    return (merged * 255).astype(np.uint8)


def _sanitize_stem(raw_stem: str) -> str:
    match = re.search(r"\((\d+)\)$", raw_stem.strip())
    if not match:
        return re.sub(r"[^a-zA-Z0-9]", "_", raw_stem).strip("_")
    prefix = raw_stem[: match.start()].strip()
    number = int(match.group(1))
    return f"{prefix}_{number:03d}"


def process(busi_root: Path, output_dir: Path) -> None:
    images_out = output_dir / "images"
    masks_out  = output_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    categories = ["benign", "malignant"]
    n_total = n_merged = 0

    for category in categories:
        cat_dir = busi_root / category
        if not cat_dir.is_dir():
            print(f"[SKIP] Directory not found: {cat_dir}")
            continue

        images = _collect_images(cat_dir)
        print(f"\n[{category}] {len(images)} images")

        for img_path in images:
            stem_out = _sanitize_stem(img_path.stem)
            dest_img  = images_out / f"{stem_out}.png"
            dest_mask = masks_out  / f"{stem_out}.png"

            if dest_img.exists():
                print(f"  [skip] {dest_img.name} already exists")
            else:
                shutil.copy2(img_path, dest_img)

            mask_paths = _find_masks(img_path)
            if not mask_paths:
                print(f"  [WARN] No mask found for {img_path.name}, skipping sample")
                dest_img.unlink(missing_ok=True)
                continue

            if len(mask_paths) > 1:
                n_merged += 1
                print(f"  {img_path.name}: merging {len(mask_paths)} masks -> {dest_mask.name}")

            merged = _merge_masks(mask_paths)
            cv2.imwrite(str(dest_mask), merged)
            n_total += 1

    print(f"\nDone: {n_total} samples written to {output_dir}")
    print(f"  {n_merged} samples had multiple masks that were merged")
    print(f"\nNext step:")
    print(f"  python tools/prepare_dataset.py --preset busi \\")
    print(f"      --dataset-root {output_dir} --output-dir {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess BUSI: merge multi-mask images and flatten directory structure."
    )
    parser.add_argument("--busi-root", required=True,
                        help="Path to raw BUSI directory (contains benign/ and malignant/)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for images/ and masks/ (default: same as --busi-root)")
    args = parser.parse_args()

    busi_root  = Path(args.busi_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else busi_root

    if not busi_root.is_dir():
        raise NotADirectoryError(f"Not a directory: {busi_root}")

    process(busi_root, output_dir)


if __name__ == "__main__":
    main()
