0# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Helpers to build the H5-backed ImageNet dataset from a config string."""

import logging
from typing import Callable, Optional

from .imagenet_h5 import ImageNetH5

logger = logging.getLogger("dinov2")


def make_imagenet_h5_dataset(
    *,
    h5_path: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> ImageNetH5:
    """Build an ImageNetH5 dataset mirroring the make_dataset(...) interface.

    Args:
        h5_path: path to the packed `.h5` file (from pack_imagenet_h5.py).
        transform: image transform (the DINO augmentation).
        target_transform: label transform.
    """
    logger.info(f"using H5-packed ImageNet: {h5_path}")
    dataset = ImageNetH5(
        h5_path=h5_path,
        transform=transform,
        target_transform=target_transform,
    )
    logger.info(f"# of dataset samples: {len(dataset):,d}")

    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)
    return dataset
