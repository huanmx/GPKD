# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""HDF5-packed ImageNet for fast data loading on network filesystems (JuiceFS/NFS).

See README.md in this directory for the motivation and usage.
"""

from .imagenet_h5 import ImageNetH5
from .loader import make_imagenet_h5_dataset

__all__ = ["ImageNetH5", "make_imagenet_h5_dataset"]
