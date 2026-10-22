# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
HDF5-backed ImageNet dataset (raw JPEG bytes in one big file).

Drop-in replacement for the file-per-image `ImageNet` dataset, reading from the
`.h5` produced by `pack_imagenet_h5.py`. Because the whole split lives in a
single file opened once per worker, there are no per-sample open()/stat() calls
-- eliminating the network-filesystem metadata storm that turns individual ranks
into stragglers.

It subclasses `ExtendedVisionDataset`, so decoding/transform/`__getitem__` are
inherited unchanged: we only override `get_image_data()` (return raw bytes from
the H5) and `get_target()`. This means the DINO augmentation pipeline and the
teacher-cls-token wrappers keep working with zero changes.

The samples are laid out by the base dataset's index (row i == base_dataset[i]),
so this stays aligned with the memmap teacher-token file which is also indexed
by dataset index.
"""

import json
import logging
import os
from typing import Any, Optional

import h5py
import numpy as np

from dinov2.data.datasets.extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")


class ImageNetH5(ExtendedVisionDataset):
    """ImageNet backed by a single packed HDF5 file of raw JPEG bytes.

    Args:
        h5_path: path to the packed `.h5` (from pack_imagenet_h5.py).
        transform / target_transform / transforms: same as any VisionDataset.
        validate: if True (default), cross-check num_samples against the sidecar meta.
    """

    def __init__(
        self,
        *,
        h5_path: str,
        transforms=None,
        transform=None,
        target_transform=None,
        validate: bool = True,
        **kwargs,
    ) -> None:
        # root is unused (data is inside the h5); pass its dir for VisionDataset.
        root = os.path.dirname(os.path.abspath(h5_path))
        super().__init__(root, transforms, transform, target_transform, **kwargs)

        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"Packed ImageNet H5 not found: {h5_path}. "
                f"Run `python -m dinov2.h5_data.pack_imagenet_h5 ...` first."
            )

        # Read cheap metadata up front (length, targets) using a short-lived handle.
        with h5py.File(h5_path, "r") as h5f:
            self._num_samples = int(h5f.attrs["num_samples"])
            self._dataset_str = str(h5f.attrs.get("dataset_str", ""))
            # targets is tiny (N int64) -> load fully into RAM for fast get_target.
            self._targets = np.asarray(h5f["targets"][:], dtype=np.int64)
            # relpaths let the teacher-cls-token wrapper locate per-sample .pt files
            # (its preferred API is get_image_relpath). Loaded once into RAM.
            if "relpaths" in h5f:
                self._relpaths = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in h5f["relpaths"][:]
                ]
            else:
                self._relpaths = None

        if validate:
            self._validate_meta()

        # Per-worker lazy handle: an open h5py.File must NOT be shared across the
        # fork/spawn boundary, so each DataLoader worker opens its own on first use.
        self._h5f: Optional[h5py.File] = None

        logger.info(f"Initialized ImageNetH5 from {h5_path} with {self._num_samples:,d} samples")

    def _validate_meta(self):
        meta_path = self.h5_path + ".meta.json"
        if not os.path.exists(meta_path):
            return
        with open(meta_path, "r") as f:
            meta = json.load(f)
        expected = int(meta.get("num_samples", -1))
        if expected != -1 and expected != self._num_samples:
            raise ValueError(
                f"H5 attr num_samples={self._num_samples} disagrees with meta "
                f"num_samples={expected} for {self.h5_path}"
            )

    def _ensure_open(self):
        if self._h5f is None:
            # swmr allows many reader processes; libver='latest' matches the writer.
            self._h5f = h5py.File(self.h5_path, "r")
            self._bytes_grp = self._h5f["bytes"]

    def get_image_data(self, index: int) -> bytes:
        self._ensure_open()
        arr = self._bytes_grp[str(index)][:]  # uint8 1-D
        return arr.tobytes()

    def get_target(self, index: int) -> Optional[Any]:
        target = int(self._targets[index])
        return None if target < 0 else target

    def get_targets(self) -> np.ndarray:
        return self._targets

    def get_image_relpath(self, index: int) -> str:
        """Original image relpath, e.g. 'train/n01440764/n01440764_10026.JPEG'.

        Preferred API used by DatasetWithTeacherClsToken to locate per-sample
        teacher token files. Returns "" if relpaths were not packed.
        """
        if self._relpaths is None:
            return ""
        return self._relpaths[index]

    def __len__(self) -> int:
        return self._num_samples

    # h5py handles are not picklable; drop them so the dataset can cross the
    # DataLoader worker boundary. Each worker reopens lazily via _ensure_open().
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5f"] = None
        state.pop("_bytes_grp", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5f = None
