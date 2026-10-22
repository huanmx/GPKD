"""The on-disk interface shared by extraction, validation and distillation."""

from pathlib import Path


def token_relative_path(image_path, image_root):
    """Map <image_root>/<class>/<image> to <class>/<image_stem>.pt."""
    return Path(image_path).resolve().relative_to(Path(image_root).resolve()).with_suffix(".pt")


def token_candidates(token_root, image_relpath, index=None):
    """Accept the canonical class tree and the original DINOv2 split tree."""
    root = Path(token_root)
    candidates = []
    if image_relpath is not None:
        rel = Path(image_relpath)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Expected a relative image path, got {rel}")
        if len(rel.parts) >= 2:
            candidates.append(root / rel.parts[-2] / (rel.stem + ".pt"))
        candidates.extend((root / rel.with_suffix(".pt"), root / (rel.stem + ".pt")))
    if index is not None:
        candidates.append(root / f"{index}.pt")
    return list(dict.fromkeys(candidates))


def find_token_path(token_root, image_relpath, index=None):
    for path in token_candidates(token_root, image_relpath, index):
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing teacher token: index={index}, image={image_relpath}, token_root={token_root}"
    )
