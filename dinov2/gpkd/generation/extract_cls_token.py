import argparse
import json
import os
from pathlib import Path

from dinov2.gpkd.token_paths import token_relative_path


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="ImageFolder root, e.g. ImageNet/train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--teacher-model", default="SiT-XL/2")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per process")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--vae", default="ema", help="ema, mse, or a local/Hugging Face VAE directory")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--teacher-z-dims", default="768")
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="Must match the REG checkpoint architecture")
    parser.add_argument("--path-type", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--cls-dim", type=int, default=768)
    parser.add_argument("--cls-init", choices=("legacy", "gaussian"), default="legacy")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--mode", choices=("ode", "sde"), default="ode")
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=0.85)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cls-cfg-scale", type=float, default=0.0)
    parser.add_argument("--noise-time", type=float, default=0.99)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate tokens instead of resuming")
    return parser


def validate_args(args):
    if args.batch_size < 1 or args.num_steps < 1 or args.num_workers < 0:
        raise ValueError("batch-size and num-steps must be positive; num-workers must be nonnegative")
    if args.resolution < 16 or args.resolution % 8:
        raise ValueError("resolution must be a positive multiple of 8 (at least 16)")
    if not 0.04 <= args.noise_time <= 1:
        raise ValueError("noise-time must be in [0.04, 1] for the REG time grid")
    if not 0 <= args.guidance_low <= args.guidance_high <= 1:
        raise ValueError("guidance interval must lie in [0, 1]")
    if args.cls_dim != int(args.teacher_z_dims.split(",")[0]):
        raise ValueError("cls-dim must match the first teacher-z-dims value in the checkpoint")
    if args.cfg_scale > 1 and args.cfg_prob <= 0:
        raise ValueError("Classifier-free guidance requires a null-class embedding")


def center_crop_arr(pil_image, image_size):
    import numpy as np
    from PIL import Image

    while min(pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX)
    scale = image_size / min(pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC)
    arr = np.array(pil_image)
    y, x = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[y:y + image_size, x:x + image_size])


def make_initial_states(x, args):
    """Preserve legacy RNG order; expose the paper's pure Gaussian alternative."""
    import torch
    import numpy as np

    noises = torch.randn_like(x)
    cls_token = torch.randn(x.shape[0], args.cls_dim, device=x.device)
    noises_cls = torch.randn_like(cls_token)
    time = torch.rand((x.shape[0], 1, 1, 1), device=x.device).to(x.dtype).fill_(args.noise_time)
    if args.path_type == "linear":
        alpha, sigma = 1 - time, time
    else:
        alpha, sigma = torch.cos(time * np.pi / 2), torch.sin(time * np.pi / 2)
    cls_input = (alpha.squeeze(-1).squeeze(-1) * cls_token
                 + sigma.squeeze(-1).squeeze(-1) * noises_cls)
    if args.cls_init == "gaussian":
        cls_input = cls_token
    return alpha * x + sigma * noises, cls_input


def prepare_manifest(output_dir, args, class_to_idx, world_size):
    """Prevent an accidental resume with incompatible generation settings."""
    metadata = {key: value for key, value in vars(args).items()
                if key not in {"output_dir", "overwrite", "num_workers"}}
    metadata.update(format_version=1, class_to_idx=class_to_idx, world_size=world_size)
    path = Path(output_dir) / "_gpkd_tokens.json"
    if path.exists() and not args.overwrite:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != metadata:
            raise ValueError("Token metadata differs. Use a new output directory or --overwrite.")
    elif not path.exists() and any(Path(output_dir).rglob("*.pt")) and not args.overwrite:
        raise ValueError("Untracked tokens already exist. Use a new output directory or --overwrite.")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(args=None):
    args = args or get_args_parser().parse_args()
    validate_args(args)
    # Delayed imports keep --help available without the GPU runtime installed.
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers.models import AutoencoderKL
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import datasets, transforms
    from tqdm.auto import tqdm

    from .models.sit import SiT_models
    from .samplers import euler_maruyama_sampler, euler_ode_sampler

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed, device_specific=True)
    device = accelerator.device
    if args.teacher_model not in SiT_models:
        raise ValueError(f"Unknown REG model: {args.teacher_model}")
    if not Path(args.teacher_ckpt).is_file():
        raise FileNotFoundError(args.teacher_ckpt)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.resolution)), transforms.ToTensor(),
    ])
    base_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if len(base_dataset.classes) != args.num_classes:
        raise ValueError(f"Expected {args.num_classes} class folders; got {len(base_dataset.classes)}")
    # ImageFolder's sorted class order must be the order used to train REG.
    class ImageFolderWithPaths(Dataset):
        def __len__(self):
            return len(base_dataset)

        def __getitem__(self, index):
            image, label = base_dataset[index]
            return image, label, base_dataset.samples[index][0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        prepare_manifest(output_dir, args, base_dataset.class_to_idx, accelerator.num_processes)
    accelerator.wait_for_everyone()

    vae_name = f"stabilityai/sd-vae-ft-{args.vae}" if args.vae in ("ema", "mse") else args.vae
    vae = AutoencoderKL.from_pretrained(vae_name).to(device).eval().requires_grad_(False)
    teacher = SiT_models[args.teacher_model](
        input_size=args.resolution // 8, num_classes=args.num_classes, use_cfg=args.cfg_prob > 0,
        z_dims=[int(z) for z in args.teacher_z_dims.split(",")], encoder_depth=args.encoder_depth,
        fused_attn=args.fused_attn, qk_norm=args.qk_norm,
    ).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    teacher.load_state_dict(state, strict=True)
    teacher.eval().requires_grad_(False)
    del ckpt, state

    # Exact non-overlapping shards avoid Accelerate's padded final batches and
    # concurrent writes for datasets not divisible by the number of processes.
    dataset = Subset(ImageFolderWithPaths(), range(accelerator.process_index,
                                                 len(base_dataset), accelerator.num_processes))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    sampler = euler_ode_sampler if args.mode == "ode" else euler_maruyama_sampler
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, disable=not accelerator.is_local_main_process):
            destinations = [output_dir / token_relative_path(p, args.data_dir) for p in paths]
            missing = [i for i, p in enumerate(destinations) if args.overwrite or not p.is_file()]
            if not missing:
                continue
            images = images[missing].to(device)
            labels = labels[missing].to(device)
            raw = torch.clamp((images * 255).round(), 0, 255)
            posterior = vae.encode(raw.to(torch.float32) / 127.5 - 1).latent_dist
            x = (posterior.mean + posterior.std * torch.randn_like(posterior.mean)) * 0.18215
            latents, cls_latents = make_initial_states(x, args)
            with accelerator.autocast():
                _, tokens = sampler(
                    model=teacher, latents=latents, cls_latents=cls_latents, y=labels,
                    num_steps=args.num_steps, heun=args.heun, cfg_scale=args.cfg_scale,
                    guidance_low=args.guidance_low, guidance_high=args.guidance_high,
                    path_type=args.path_type, args=args, return_cls=True, noise_time=args.noise_time,
                )
            tokens = tokens.float().cpu()
            if tokens.shape != (len(missing), args.cls_dim) or not torch.isfinite(tokens).all():
                raise ValueError(f"Invalid teacher tokens: shape={tuple(tokens.shape)}")
            for token, i in zip(tokens, missing):
                destination = destinations[i]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(f".pt.{os.getpid()}.tmp")
                torch.save(token.clone(), temporary)
                temporary.replace(destination)
    accelerator.wait_for_everyone()
    accelerator.print(f"Extraction completed: {len(base_dataset)} images -> {output_dir}")


if __name__ == "__main__":
    main()
