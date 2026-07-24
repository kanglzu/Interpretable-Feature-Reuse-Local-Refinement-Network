"""Generate manifests and cross-validation splits."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.split import build_manifest, build_splits, save_manifest, save_splits


PRESETS = {
    "cvc": {
        "dataset_name": "CVC-ClinicDB",
        "image_subdir": "Original",
        "label_subdir": "Ground Truth",
        "file_ending": ".png",
        "label_ending": None,    # same as file_ending
        "label_suffix": "",
        "channel_names": ["R", "G", "B"],
        "label_names": ["background", "polyp"],
    },
    "isic": {
        "dataset_name": "ISIC2018",
        "image_subdir": "ISIC2018_Input",
        "label_subdir": "ISIC2018_GroundTruth",
        "file_ending": ".jpg",
        "label_ending": ".png",
        "label_suffix": "_segmentation",
        "channel_names": ["R", "G", "B"],
        "label_names": ["background", "lesion"],
    },
    "busi": {
        "dataset_name": "BUSI",
        "image_subdir": "images",
        "label_subdir": "masks",
        "file_ending": ".png",
        "label_ending": None,    # same as file_ending (.png)
        "label_suffix": "",
        "channel_names": ["R", "G", "B"],
        "label_names": ["background", "lesion"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate manifest.json and splits.json")

    parser.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                        help="Use a built-in preset (cvc, isic, busi). "
                             "Only --dataset-root and --output-dir are required with a preset.")
    parser.add_argument("--dataset-root", default=None,
                        help="Root directory of the dataset (used with --preset)")

    parser.add_argument("--image-dir", default=None, help="Directory containing input images")
    parser.add_argument("--label-dir", default=None,
                        help="Directory containing label masks")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSON files")
    parser.add_argument("--dataset-name", default="MyDataset")
    parser.add_argument("--description", default="")
    parser.add_argument("--file-ending", default=".png",
                        help="File extension for images")
    parser.add_argument("--label-ending", default=None,
                        help="File extension for labels (defaults to --file-ending)")
    parser.add_argument("--label-suffix", default="",
                        help="Suffix appended to image stem for label filename "
                             "(e.g. '_segmentation' for ISIC2018)")
    parser.add_argument("--channel-names", nargs="+", default=["R", "G", "B"],
                        help="Channel modality names, e.g. R G B or FLAIR T1")
    parser.add_argument("--label-names", nargs="+", default=["background", "foreground"],
                        help="Class names in order (index 0 = background)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--test-prefix", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.preset:
        preset = PRESETS[args.preset]
        if not args.dataset_root:
            raise ValueError("--dataset-root is required when using --preset")
        root = Path(args.dataset_root)
        image_dir = str(root / preset["image_subdir"])
        label_dir = str(root / preset["label_subdir"])
        dataset_name = preset["dataset_name"]
        file_ending = preset["file_ending"]
        label_ending = preset["label_ending"]
        label_suffix = preset["label_suffix"]
        channel_names_list = preset["channel_names"]
        label_names_list = preset["label_names"]
        description = args.description
        print(f"[Preset] Using built-in preset: {args.preset}")
    else:
        if not args.image_dir or not args.label_dir:
            raise ValueError("--image-dir and --label-dir are required (or use --preset)")
        image_dir = args.image_dir
        label_dir = args.label_dir
        dataset_name = args.dataset_name
        file_ending = args.file_ending
        label_ending = args.label_ending
        label_suffix = args.label_suffix
        channel_names_list = args.channel_names
        label_names_list = args.label_names
        description = args.description

    channel_names = {str(i): name for i, name in enumerate(channel_names_list)}
    labels = {name: idx for idx, name in enumerate(label_names_list)}

    manifest = build_manifest(
        image_dir=image_dir,
        label_dir=label_dir,
        file_ending=file_ending,
        dataset_name=dataset_name,
        channel_names=channel_names,
        labels=labels,
        description=description,
        label_ending=label_ending,
        label_suffix=label_suffix,
    )
    manifest_path = out_dir / "manifest.json"
    save_manifest(manifest, str(manifest_path))
    print(f"Manifest -> {manifest_path}  ({manifest['num_samples']} samples)")

    splits = build_splits(
        manifest=manifest,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    splits_path = out_dir / "splits.json"
    save_splits(splits, str(splits_path))
    print(
        f"Splits  -> {splits_path}  "
        f"({splits['n_train_val']} samples, {args.n_folds}-fold CV, no test set)"
    )

    print("\nExample config snippet:")
    print(f'  data:')
    print(f'    manifest_json: "{manifest_path}"')
    print(f'    splits_json:   "{splits_path}"')


if __name__ == "__main__":
    main()
