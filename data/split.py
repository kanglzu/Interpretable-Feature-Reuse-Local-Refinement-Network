"""Manifest and cross-validation split utilities."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def build_manifest(
    image_dir: str,
    label_dir: str,
    file_ending: str,
    dataset_name: str,
    channel_names: Dict[str, str],
    labels: Dict[str, int],
    description: str = "",
    label_ending: Optional[str] = None,
    label_suffix: str = "",
) -> dict:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    if label_ending is None:
        label_ending = file_ending

    stems = sorted(
        p.stem for p in image_dir.glob(f"*{file_ending}")
    )
    if not stems:
        raise FileNotFoundError(
            f"No files with ending '{file_ending}' found in {image_dir}"
        )

    samples = []
    for stem in stems:
        img_path = image_dir / f"{stem}{file_ending}"
        lbl_path = label_dir / f"{stem}{label_suffix}{label_ending}"
        if not lbl_path.exists():
            raise FileNotFoundError(f"Label not found for sample '{stem}': {lbl_path}")
        samples.append({
            "id": stem,
            "image": str(img_path),
            "label": str(lbl_path),
        })

    return {
        "name": dataset_name,
        "description": description,
        "channel_names": {str(k): v for k, v in channel_names.items()},
        "labels": {k: int(v) for k, v in labels.items()},
        "num_classes": len(labels),
        "file_ending": file_ending,
        "num_samples": len(samples),
        "samples": samples,
    }


def save_manifest(manifest: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _kfold_split(ids: List[str], n_folds: int) -> List[Dict]:
    n = len(ids)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
    if n < n_folds:
        raise ValueError(f"Cannot split {n} samples into {n_folds} folds.")

    base_size, remainder = divmod(n, n_folds)
    folds = []
    cursor = 0
    for i in range(n_folds):
        fold_size = base_size + (1 if i < remainder else 0)
        val_start = cursor
        val_end = cursor + fold_size
        val_ids = ids[val_start:val_end]
        train_ids = ids[:val_start] + ids[val_end:]
        folds.append({"fold": i, "train": train_ids, "val": val_ids})
        cursor = val_end
    return folds


def build_splits(
    manifest: dict,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    all_ids = [s["id"] for s in manifest["samples"]]
    duplicates = _duplicates(all_ids)
    if duplicates:
        raise ValueError(f"Duplicate sample ids in manifest: {duplicates[:10]}.")

    rng = random.Random(seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)

    folds = _kfold_split(shuffled, n_folds)

    return {
        "dataset": manifest["name"],
        "seed": seed,
        "n_folds": n_folds,
        "total": len(all_ids),
        "n_train_val": len(shuffled),
        "folds": folds,
    }


def save_splits(splits: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(splits_json: str, fold: int) -> Tuple[List[str], List[str]]:
    with open(splits_json) as f:
        splits = json.load(f)

    if fold >= splits["n_folds"]:
        raise ValueError(
            f"Requested fold {fold} but splits.json only has {splits['n_folds']} folds."
        )

    entry = splits["folds"][fold]
    train_ids = list(entry["train"])
    val_ids = list(entry["val"])
    train_dups = _duplicates(train_ids)
    val_dups = _duplicates(val_ids)
    if train_dups:
        raise ValueError(f"Duplicate train ids in fold {fold}: {train_dups[:10]}.")
    if val_dups:
        raise ValueError(f"Duplicate val ids in fold {fold}: {val_dups[:10]}.")
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(
            f"Data leakage risk: fold {fold} has {len(overlap)} ids in both "
            f"train and val, e.g. {overlap[:10]}."
        )
    return train_ids, val_ids


def load_test_ids(splits_json: str) -> List[str]:
    with open(splits_json) as f:
        splits = json.load(f)
    return splits.get("test", [])


def load_manifest(manifest_json: str) -> dict:
    with open(manifest_json) as f:
        return json.load(f)


def get_sample_dict(manifest: dict) -> Dict[str, dict]:
    sample_ids = [s["id"] for s in manifest["samples"]]
    duplicates = _duplicates(sample_ids)
    if duplicates:
        raise ValueError(f"Duplicate sample ids in manifest: {duplicates[:10]}.")
    return {s["id"]: s for s in manifest["samples"]}


def _duplicates(values: List[str]) -> List[str]:
    seen = set()
    dupes = []
    dupes_seen = set()
    for value in values:
        if value in seen and value not in dupes_seen:
            dupes.append(value)
            dupes_seen.add(value)
        seen.add(value)
    return dupes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate manifest.json and splits.json")
    parser.add_argument("--image-dir", required=True, help="Directory with input images")
    parser.add_argument("--label-dir", required=True, help="Directory with label masks")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSON files")
    parser.add_argument("--dataset-name", default="MyDataset")
    parser.add_argument("--file-ending", default=".png")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_manifest(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        file_ending=args.file_ending,
        dataset_name=args.dataset_name,
        channel_names={"0": "channel_0"},
        labels={"background": 0, "foreground": 1},
    )

    out_dir = Path(args.output_dir)
    save_manifest(manifest, out_dir / "manifest.json")

    splits = build_splits(manifest, n_folds=args.n_folds, seed=args.seed)
    save_splits(splits, out_dir / "splits.json")

    print(f"Manifest: {manifest['num_samples']} samples -> {out_dir / 'manifest.json'}")
    print(
        f"Splits:   {splits['n_train_val']} samples "
        f"({args.n_folds}-fold CV, no test set) -> {out_dir / 'splits.json'}"
    )
