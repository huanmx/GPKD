# h5_data — HDF5-packed ImageNet for fast loading on JuiceFS / NFS

## Why this exists

Training reads ImageNet as ~1.28M individual JPEG files. On a network filesystem
(this cluster mounts data over **JuiceFS** and **NFS**), every sample triggers an
`open()`/`stat()` metadata round-trip to a remote metadata service. A single slow
round-trip makes one data-loader rank a *straggler*; because DINO uses collective
ops (sinkhorn `all_reduce`, FSDP `all-gather`), that one slow rank stalls all the
others. This is the cause of the multi-second per-iteration spikes (a rank
arriving ~4s late at the first collective barrier while the other 7 wait).

This package packs an entire split into **one HDF5 file of raw JPEG bytes**. The
file is opened once per worker; every sample is a read at a file offset inside it
— no per-sample `open()`/`stat()`, so no metadata-storm stragglers. Same idea as
REPA-E's `images.h5`.

We store the **original, undecoded JPEG bytes** (not decoded arrays) because:
- DINO needs full-resolution images for its random global/local crops, so images
  must not be pre-resized.
- Bytes keep the file ~= raw dataset size (~140GB); decoded RGB would be multiple TB.

Samples are laid out by the base dataset's index (`bytes/i` == `base_dataset[i]`),
so it stays aligned with an index-based memmap teacher-token file.

## Files

- `pack_imagenet_h5.py` — offline: build `<split>.h5` from the file-per-image dataset.
- `imagenet_h5.py` — `ImageNetH5`, a runtime dataset reading from the packed `.h5`.
  Subclasses `ExtendedVisionDataset`, so decode/transform/`__getitem__` are inherited;
  only `get_image_data()` / `get_target()` are overridden.
- `loader.py` — `make_imagenet_h5_dataset(...)`, mirrors `make_dataset(...)`.

## Step 1 — Pack (run once)

```bash
python -m dinov2.h5_data.pack_imagenet_h5 \
    --dataset-path "ImageNet:split=TRAIN:root=/path/to/imagenet:extra=/path/to/imagenet_extra" \
    --out-path     /jobutils/imagenet_h5/train.h5 \
    --num-workers  16
```

- `--dataset-path` must be **identical** to the training dataset string (same
  split/root/extra) so the packed index order matches.
- `--num-workers` parallelizes the slow part (pulling bytes off JuiceFS). The
  writer itself is single-process (one owner of the `.h5`).
- Output: `train.h5` (~140GB) + `train.h5.meta.json`.
- **Write the `.h5` to a local disk** — `/jobutils` is the local SSD on this box
  (`/dev/sda4`, ~268GB free), which fits ImageNet. Avoid writing it back onto
  JuiceFS/NFS.

## Step 2 — Point training at it

Add to your training command / config (`cfg.train.image_h5_path`):

```bash
train.image_h5_path=/jobutils/imagenet_h5/train.h5
```

`train.py` reads images from the H5 when `image_h5_path` is set, and falls back to
the file-per-image dataset when it is empty. The teacher-cls-token wrapper keeps
working: `ImageNetH5.get_image_relpath()` lets it locate per-sample `.pt` files —
or combine with the index-based memmap token file for a fully small-file-free path.

## Notes

- **tmpfs option:** this box also has large tmpfs (`/dev/shm`, `/tmp`). ImageNet at
  ~140GB is too big for comfort there; use `/jobutils` (persistent SSD). The small
  (~2GB) teacher-token file is the one to put on `/dev/shm`.
- **Persistence:** `/jobutils` survives restarts; `/dev/shm` does not (re-copy on
  pod restart).
- **Verify after packing:** compare `find <src> -type f | wc -l` with the packed
  `num_samples`, and spot-check that a few decoded images match the originals.
- **h5py + concurrency:** each DataLoader worker opens its own read-only handle
  lazily (handles are not fork-safe); `ImageNetH5.__getstate__` drops the handle so
  the dataset can cross the worker boundary.
