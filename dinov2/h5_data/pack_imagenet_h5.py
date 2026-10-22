# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Pack an ImageNet split into a single HDF5 file of raw JPEG bytes.

Why
---
On a network filesystem (JuiceFS / NFS), reading ImageNet as ~1.28M individual
JPEG files means ~1.28M open()/stat() metadata round-trips. Any single slow
round-trip makes one data-loader rank a straggler, and because DINO uses
collective ops (sinkhorn all_reduce, FSDP all-gather) a single slow rank stalls
all the others -- the multi-second per-iteration spikes seen in training.

This script aggregates all images of a split into ONE big `.h5` file, keyed by
the dataset's own index (dataset "i" == the raw JPEG bytes for base_dataset[i]).
At train time the H5 file is opened once and every sample is a read at a file
offset inside it -- no per-sample open()/stat(), so no metadata-storm stragglers.

We store the ORIGINAL, UNDECODED JPEG bytes (not decoded arrays):
  * DINO needs full-resolution images for its random global/local crops, so we
    must not pre-resize.
  * Storing bytes keeps the file ~= the raw dataset size (~140GB), whereas
    storing decoded RGB would explode to multiple TB.

Layout of the output file `<split>.h5`:
  ds "bytes/<index>"     uint8 1-D array : the raw JPEG file content
  ds "targets"           int64 [N]       : class index per sample (-1 if none)
  ds "relpaths"          vlen str [N]    : original image relpath (for debugging)
  attrs: num_samples, dataset_str, split

A sidecar `<split>.h5.meta.json` records {num_samples, dataset_str, ...} so the
runtime dataset can validate it matches the base dataset before training.

Usage
-----
  python -m dinov2.h5_data.pack_imagenet_h5 \
      --dataset-path "ImageNet:split=TRAIN:root=/mnt/.../imagenet:extra=/mnt/.../imagenet_extra" \
      --out-path     /jobutils/imagenet_h5/train.h5 \
      --num-workers  16
"""

import argparse
import json
import logging
import os
from typing import List, Optional, Tuple

import h5py
import numpy as np

from dinov2.data import make_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pack_imagenet_h5")


def _meta_path(out_path: str) -> str:
    return out_path + ".meta.json"


def _build_base_dataset(dataset_path: str):
    """Build the base image dataset WITHOUT any transform.

    We only use get_image_data() (raw bytes) and get_target(); we never decode.
    """
    return make_dataset(dataset_str=dataset_path, transform=None, target_transform=None)


def _get_relpath(dataset, index: int) -> str:
    """Best-effort original relpath for a sample, for debugging/traceability."""
    try:
        if hasattr(dataset, "get_class_id") and hasattr(dataset, "split"):
            entries = dataset._get_entries()
            actual_index = int(entries[index]["actual_index"])
            class_id = dataset.get_class_id(index)
            return str(dataset.split.get_image_relpath(actual_index, class_id))
    except Exception:
        pass
    return ""


def _read_indices(
    dataset_path: str,
    indices: List[int],
) -> List[Tuple[int, bytes, int, str]]:
    """Worker: return [(index, jpeg_bytes, target, relpath), ...] for a shard."""
    dataset = _build_base_dataset(dataset_path)
    out = []
    for index in indices:
        image_bytes = dataset.get_image_data(index)  # raw JPEG bytes
        target = dataset.get_target(index)
        target = -1 if target is None else int(target)
        relpath = _get_relpath(dataset, index)
        out.append((index, image_bytes, target, relpath))
    return out


def _chunked(seq: List[int], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main():
    parser = argparse.ArgumentParser("Pack an ImageNet split into a single HDF5 file")
    parser.add_argument("--dataset-path", required=True, help="Dataset string, same as training, e.g. ImageNet:split=TRAIN:root=...:extra=...")
    parser.add_argument("--out-path", required=True, help="Output .h5 path.")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel reader processes for pulling bytes off the source FS.")
    parser.add_argument("--task-chunk", type=int, default=256, help="How many samples each worker task reads before returning (amortizes IPC).")
    parser.add_argument("--limit", type=int, default=0, help="Debug: only pack the first N samples (0 = all).")
    args = parser.parse_args()

    base = _build_base_dataset(args.dataset_path)
    num_samples = len(base)
    if args.limit > 0:
        num_samples = min(num_samples, args.limit)
    logger.info(f"packing {num_samples:,d} samples from '{args.dataset_path}'")

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    os.makedirs(out_dir or ".", exist_ok=True)

    all_indices = list(range(num_samples))

    # HDF5 writing is single-threaded (one writer process owns the file), but the
    # slow part -- pulling bytes off the network FS -- is parallelized with a pool.
    targets = np.full(num_samples, -1, dtype=np.int64)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    written = 0

    with h5py.File(args.out_path, "w") as h5f:
        h5f.attrs["num_samples"] = num_samples
        h5f.attrs["dataset_str"] = args.dataset_path
        grp = h5f.create_group("bytes")
        relpaths_ds = h5f.create_dataset("relpaths", shape=(num_samples,), dtype=str_dtype)

        def _write_batch(results):
            nonlocal written
            for index, image_bytes, target, relpath in results:
                arr = np.frombuffer(image_bytes, dtype=np.uint8)
                # gzip on already-compressed JPEG buys ~nothing and costs CPU -> store raw.
                grp.create_dataset(str(index), data=arr)
                targets[index] = target
                relpaths_ds[index] = relpath
                written += 1
                if written % 20000 == 0:
                    logger.info(f"  wrote {written:,d}/{num_samples:,d}")

        if args.num_workers <= 1:
            for chunk in _chunked(all_indices, args.task_chunk):
                _write_batch(_read_indices(args.dataset_path, chunk))
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            tasks = list(_chunked(all_indices, args.task_chunk))
            with ctx.Pool(args.num_workers) as pool:
                # imap_unordered keeps the writer fed while workers fetch in parallel.
                for results in pool.imap_unordered(
                    _ReadWorker(args.dataset_path), tasks
                ):
                    _write_batch(results)

        h5f.create_dataset("targets", data=targets)

    meta = {
        "num_samples": num_samples,
        "dataset_str": args.dataset_path,
        "num_written": written,
        "format": "raw_jpeg_bytes_by_index",
    }
    with open(_meta_path(args.out_path), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"DONE: wrote {written:,d} images to {args.out_path}")
    if written != num_samples:
        logger.warning(f"expected {num_samples} but wrote {written}; check for read errors")


class _ReadWorker:
    """Picklable callable for pool.imap (spawn context can't pickle closures)."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def __call__(self, indices: List[int]):
        return _read_indices(self.dataset_path, indices)


if __name__ == "__main__":
    main()
