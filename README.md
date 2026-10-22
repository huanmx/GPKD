<h2 align="center"> <a href="https://openreview.net/forum?id=in565dKueK#discussion">Beyond Generation: Unlocking Discriminative Representations from Diffusion Models (NeurIPS 2026)</a><h5 align="center">


## Overview

GPKD uses the generative process to improve the representations of pretrained visual encoders. Given an image and its class label, a pretrained REG model jointly denoises an image latent and a class token, producing a semantic representation that serves as a global teacher signal for a DINOv2 student.

Our framework combines two complementary objectives:

- **Class-token distillation:** transfer global semantic knowledge from REG-generated class tokens to the student.
- **Masked patch-token distillation:** preserve local structure by matching the student's masked patch representations to targets from a frozen pretrained DINOv2 encoder.

## Installation


Run the following commands from the repository root:

```bash
conda env create -f conda.yaml
conda activate gpkd

pip install -r requirements.txt
```

## Usage

Prepare ImageNet dataset:

```text
imagenet/
├── train/
│   └── n01440764/
│       └── n01440764_00000001.JPEG
└── val/
    └── n01440764/
        └── ILSVRC2012_val_00000293.JPEG
```

Prepare the metadata required by the DINOv2 data loader, or reuse existing metadata:

```bash
python -m dinov2.gpkd.prepare_imagenet \
    --root "$DATA_ROOT" \
    --extra "$IMAGENET_EXTRA"
```

Obtain a pretrained REG SiT-XL/2 checkpoint from the [REG project](https://github.com/Martinser/REG). The token generator also loads the Stable Diffusion VAE `stabilityai/sd-vae-ft-ema`.

### 1. Teacher Token Generation

Generate a teacher class token for each training image:

```bash
accelerate launch --multi_gpu --num_processes "$NUM_GPUS" \
    --module dinov2.gpkd.generation.extract_cls_token \
    --teacher-ckpt "$REG_CHECKPOINT" \
    --data-dir "$DATA_ROOT/train" \
    --output-dir "$TOKEN_DIR" \
    --path-type linear \
    --encoder-depth 8 \
    --cls-dim 768 \
    --num-steps 5 \
    --batch-size 256 \
    --mode ode \
    --noise-time 0.99 \
    --cfg-scale 1.0 \
    --cls-cfg-scale 0.0 \
    --heun
```

The generator uses 256 × 256 center crops and image class labels to condition REG. Tokens are saved as float32 tensors of shape `[768]`, preserving the class-folder layout:

```text
teacher_tokens/
└── n01440764/
    └── n01440764_00000001.pt
```


### 2. Distillation Training

Train the student with the generated class tokens and frozen DINOv2 patch targets:

```bash
torchrun --nproc_per_node=8 -m dinov2.gpkd.train \
    --config-file dinov2/configs/distill/vitb14.yaml \
    --output-dir "$OUTPUT_DIR" \
    "train.dataset_path=ImageNet:split=TRAIN:root=$DATA_ROOT:extra=$IMAGENET_EXTRA" \
    "train.token_path=$TOKEN_DIR" \
    train.batch_size_per_gpu=128 \
    ibot.loss_weight=1.0 \
    optim.epochs=20 \
    optim.warmup_epochs=5
```

### 3. Testing and Evaluation


**Linear evaluation**

```bash
torchrun --nproc_per_node=8 -m dinov2.eval.linear \
    --config-file dinov2/configs/eval/vitb14_pretrain.yaml \
    --pretrained-weights "$EVAL_WEIGHTS" \
    --output-dir "$OUTPUT_DIR/linear_eval" \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA_ROOT:extra=$IMAGENET_EXTRA" \
    --val-dataset "ImageNet:split=VAL:root=$DATA_ROOT:extra=$IMAGENET_EXTRA" \
    --batch-size 256
```

**k-NN evaluation**

```bash
torchrun --nproc_per_node=8 -m dinov2.eval.knn \
    --config-file dinov2/configs/eval/vitb14_pretrain.yaml \
    --pretrained-weights "$EVAL_WEIGHTS" \
    --output-dir "$OUTPUT_DIR/knn_eval" \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA_ROOT:extra=$IMAGENET_EXTRA" \
    --val-dataset "ImageNet:split=VAL:root=$DATA_ROOT:extra=$IMAGENET_EXTRA" \
    --batch-size 256 \
```

## Acknowledgements

Our implementation builds on [DINOv2](https://github.com/facebookresearch/dinov2) and [REG](https://github.com/Martinser/REG). We thank their authors for making the code and pretrained models publicly available.

## Citation

If you find our work useful, please consider citing:

```bibtex
@unpublished{cui2026gpkd,
  title  = {Beyond Generation: Unlocking Discriminative Representations from Diffusion Models},
  author = {Cui, Haowen and Wu, Ge and Ge, Yikai and Gao, Ge and Li, Xiang and Li, Jun and Yang, Jian and Chen, Shuo},
  journal = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year   = {2026},
}
```
