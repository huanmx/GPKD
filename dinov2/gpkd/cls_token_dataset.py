import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from dinov2.data.collate import collate_data_and_cast
from .token_paths import find_token_path


class DatasetWithTeacherClsToken(Dataset):
    """Wrap an existing dataset and return (image, teacher_cls_token, label).

    Tokens are read from <cls_token_root>/<class_name>/<image_stem>.pt.
    """

    def __init__(self, dataset: Dataset, cls_token_root: str, expected_dim: Optional[int] = None):
        self.dataset = dataset
        self.cls_token_root = Path(cls_token_root)
        self.expected_dim = expected_dim
        if not self.cls_token_root.is_dir():
            raise FileNotFoundError(f"Teacher token directory does not exist: {self.cls_token_root}")

    def __len__(self):
        return len(self.dataset)

    def _get_relpath(self, index: int) -> Optional[str]:
        # Preferred API for most dinov2 datasets
        if hasattr(self.dataset, "get_image_relpath"):
            relpath = self.dataset.get_image_relpath(index)
            root = getattr(self.dataset, "root", None)
            if root is not None and os.path.isabs(relpath):
                try:
                    return str(Path(relpath).relative_to(root))
                except ValueError:
                    return str(Path(relpath).name)
            return str(relpath)

        # Fallback for dinov2 ImageNet dataset
        if hasattr(self.dataset, "split") and hasattr(self.dataset, "_get_entries"):
            entries = self.dataset._get_entries()
            entry = entries[index]
            actual_index = int(entry["actual_index"])

            class_id = None
            dtype_names = getattr(getattr(entries, "dtype", None), "names", None)
            if dtype_names is not None and "class_id" in dtype_names:
                class_id = str(entry["class_id"])
            elif hasattr(self.dataset, "get_class_id"):
                class_id = self.dataset.get_class_id(index)

            if hasattr(self.dataset.split, "get_image_relpath"):
                return str(self.dataset.split.get_image_relpath(actual_index, class_id))

        return None

    def _find_token_path(self, index: int) -> Path:
        return find_token_path(self.cls_token_root, self._get_relpath(index), index)

    @staticmethod
    def _extract_tensor(obj: Any) -> torch.Tensor:
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for key in ["cls_token", "cls_tokens", "token", "tokens", "x_norm_clstoken", "cls"]:
                if key in obj and torch.is_tensor(obj[key]):
                    return obj[key]
            for v in obj.values():
                if torch.is_tensor(v):
                    return v
        raise ValueError("Unsupported cls token file format, expected Tensor or dict containing Tensor")

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        token_obj = torch.load(self._find_token_path(index), map_location="cpu", weights_only=True)
        teacher_cls_token = self._extract_tensor(token_obj).float().reshape(-1)
        if self.expected_dim is not None and teacher_cls_token.numel() != self.expected_dim:
            raise ValueError(f"Token dimension {teacher_cls_token.numel()} != student dimension {self.expected_dim}")
        if not torch.isfinite(teacher_cls_token).all():
            raise ValueError(f"Non-finite teacher token for sample {index}")
        label = torch.tensor(self.dataset.get_target(index))
        return image, teacher_cls_token, label


def _collate_labels(labels):
    if len(labels) == 0:
        return labels

    first = labels[0]
    if torch.is_tensor(first):
        try:
            return torch.stack(labels, dim=0)
        except Exception:
            return labels

    if isinstance(first, (int, bool)):
        return torch.tensor(labels, dtype=torch.long)

    return labels


def collate_data_and_cast_with_teacher_cls_token(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
):
    image_samples = [(sample[0], 0) for sample in samples_list]
    batch = collate_data_and_cast(
        image_samples,
        mask_ratio_tuple=mask_ratio_tuple,
        mask_probability=mask_probability,
        dtype=dtype,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
    )

    batch["teacher_cls_tokens"] = torch.stack([sample[1] for sample in samples_list], dim=0).to(dtype)
    batch["labels"] = _collate_labels([sample[2] for sample in samples_list])
    return batch
