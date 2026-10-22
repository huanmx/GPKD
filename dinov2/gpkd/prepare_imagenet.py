"""Create DINOv2 ImageNet metadata from existing train/val class folders."""

import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def prepare(root, extra, splits=("train", "val")):
    import numpy as np

    root, extra = Path(root), Path(extra)
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    if not classes:
        raise ValueError("No class folders found in <root>/train")
    class_to_idx = {name: i for i, name in enumerate(classes)}
    extra.mkdir(parents=True, exist_ok=True)
    for split in splits:
        split_root = root / split
        found = sorted(p.name for p in split_root.iterdir() if p.is_dir())
        if found != classes:
            raise ValueError(f"{split} class directories differ from train")
        rows = []
        for class_id in classes:
            for path in sorted((split_root / class_id).iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    actual_index = int(path.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Expected original ImageNet filename: {path}") from exc
                prefix = class_id if split == "train" else "ILSVRC2012_val"
                expected = f"{prefix}_{actual_index:08d}.JPEG"
                if path.name != expected:
                    raise ValueError(f"DINOv2 requires {expected}; got {path.name}")
                rows.append((actual_index, class_to_idx[class_id], class_id, class_id))
        if not rows:
            raise ValueError(f"No images found in {split_root}")
        width = max(map(len, classes))
        dtype = [("actual_index", "<u4"), ("class_index", "<u4"),
                 ("class_id", f"U{width}"), ("class_name", f"U{width}")]
        # class_name defaults to the synset; names are not needed by distillation.
        np.save(extra / f"entries-{split.upper()}.npy", np.array(rows, dtype=dtype))
        np.save(extra / f"class-ids-{split.upper()}.npy", np.array(classes))
        np.save(extra / f"class-names-{split.upper()}.npy", np.array(classes))
        print(f"{split}: {len(rows)} images, {len(classes)} classes -> {extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="ImageNet directory containing train/ and val/")
    parser.add_argument("--extra", required=True, help="Output metadata directory")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    args = parser.parse_args()
    prepare(args.root, args.extra, args.splits)


if __name__ == "__main__":
    main()
