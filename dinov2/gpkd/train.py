# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import DataAugmentationDINO, CellAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup, write_config
from dinov2.utils.utils import CosineScheduler

from dinov2.gpkd.ssl_meta_arch import SSLMetaArch
from dinov2.gpkd.cls_token_dataset import (
    DatasetWithTeacherClsToken,
    collate_data_and_cast_with_teacher_cls_token,
)
from dinov2.gpkd.checkpoints import unchunk_state_dict

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("GPKD distillation", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    
    parser.add_argument(
        "--token-path",
        "--token_path",
        default="",
        type=str
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def build_backbone_stage2_scheduler(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH
    stage2_start_iter = min(int(getattr(cfg.train, "head_warmup_epochs", 0)) * OFFICIAL_EPOCH_LENGTH, max_iter)
    stage2_total_iters = max(max_iter - stage2_start_iter, 1)

    backbone_base_lr = float(getattr(cfg.optim, "stage2_backbone_base_lr", cfg.optim.lr))
    backbone_stage2_schedule = CosineScheduler(
        base_value=backbone_base_lr,
        final_value=cfg.optim["min_lr"],
        total_iters=stage2_total_iters,
        warmup_iters=0,
        start_warmup_value=backbone_base_lr,
    )
    return backbone_stage2_schedule, stage2_start_iter


def get_training_stage(cfg, iteration):
    stage1_epochs = int(getattr(cfg.train, "head_warmup_epochs", 0))
    if stage1_epochs <= 0:
        return "joint_finetune"
    current_epoch = iteration // cfg.train.OFFICIAL_EPOCH_LENGTH
    if current_epoch < stage1_epochs:
        return "head_warmup"
    return "joint_finetune"


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr, stage, backbone_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        is_backbone = param_group.get("is_backbone", False)
        param_group["weight_decay"] = wd * wd_multiplier

        if is_backbone:
            base_lr = 0.0 if stage == "head_warmup" else backbone_lr
        else:
            base_lr = last_layer_lr if is_last_layer else lr

        param_group["lr"] = base_lr * lr_multiplier


def do_test(cfg, model, iteration):
    eval_dir = os.path.join(cfg.train.output_dir, "eval", str(iteration))
    # state_dict must run on every rank because FSDP gathers parameters.
    for name, module in (("teacher", model.teacher), ("student", model.student)):
        state = module.state_dict()
        if distributed.is_main_process():
            os.makedirs(eval_dir, exist_ok=True)
            # DINOv2 evaluation expects a 'teacher' key even for student exports.
            torch.save({"teacher": unchunk_state_dict(state)},
                       os.path.join(eval_dir, f"{name}_checkpoint.pth"))
        del state


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)
    backbone_stage2_schedule, stage2_start_iter = build_backbone_stage2_scheduler(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    if cfg.train.cell_augmentation:
        data_transform = CellAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )
    else:
        data_transform = DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )

    collate_fn = partial(
        collate_data_and_cast_with_teacher_cls_token,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    dataset = DatasetWithTeacherClsToken(
        dataset=dataset,
        cls_token_root=cfg.train.token_path,
        expected_dim=model.embed_dim,
    )
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=cfg.train.seed,
        sampler_type=sampler_type,
        sampler_advance=start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
        # persistent_workers=True,
        # prefetch_factor=6,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"
    current_stage = None

    for data in metric_logger.log_every(
        data_loader,
        100,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration >= max_iter:
            break

        # apply schedules

        # torch.cuda.synchronize()
        # st_time = time.time()

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        stage = get_training_stage(cfg, iteration)
        if stage == "head_warmup":
            ratio = float(getattr(cfg.optim, "stage1_head_base_lr", cfg.optim.base_lr)) / cfg.optim.base_lr
            lr *= ratio
            last_layer_lr *= ratio
        backbone_lr = 0.0
        if iteration >= stage2_start_iter:
            backbone_lr = backbone_stage2_schedule[iteration - stage2_start_iter]

        if stage != current_stage:
            current_stage = stage
            model.set_training_stage(stage)
            logger.info(f"Switch training stage to: {stage}")

        apply_optim_scheduler(
            optimizer,
            lr,
            wd,
            last_layer_lr,
            stage,
            backbone_lr=backbone_lr,
        )

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)
        
        # torch.cuda.synchronize()
        # loss_time = time.time()
        

        # clip gradients
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update
        model.update_teacher(mom)

        # torch.cuda.synchronize()
        # backward_time = time.time()

        # logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if not all(math.isfinite(value) for value in loss_dict_reduced.values()):
            raise FloatingPointError(f"Non-finite training loss: {loss_dict_reduced}")
        losses_reduced = sum(loss for k, loss in loss_dict_reduced.items() if k != 'teacher_acc' and k != 'student_acc')

        current_head_lr = 0.0
        current_backbone_lr = 0.0
        for param_group in optimizer.param_groups:
            if param_group.get("is_backbone", False):
                current_backbone_lr = max(current_backbone_lr, float(param_group["lr"]))
            else:
                current_head_lr = max(current_head_lr, float(param_group["lr"]))
                
        if iteration % 100 == 0:
            logger.info("lr: {}, head_lr: {}, backbone_lr: {}".format(lr, current_head_lr, current_backbone_lr))

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(head_lr=current_head_lr)
        metric_logger.update(backbone_lr=current_backbone_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # torch.cuda.synchronize()
        # log_time = time.time()

        # checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1

        # torch.cuda.synchronize()
        # save_time = time.time()
        # print('loss time: {}, backward time: {}, log time: {}, save time: {}'.format(loss_time - st_time, backward_time - loss_time, log_time - backward_time, save_time - log_time))
    
    metric_logger.synchronize_between_processes()
    do_test(cfg, model, "final")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    if args.token_path:
        args.opts.append(f"train.token_path={args.token_path}")
    cfg = setup(args)
    if not args.eval_only and (not cfg.train.token_path or str(cfg.train.token_path) == "None"):
        raise ValueError("Set train.token_path or --token-path to the generated teacher token directory")
    if int(cfg.train.OFFICIAL_EPOCH_LENGTH) == 0:
        dataset = make_dataset(dataset_str=cfg.train.dataset_path)
        global_batch = cfg.train.batch_size_per_gpu * distributed.get_global_size()
        cfg.train.OFFICIAL_EPOCH_LENGTH = max(1, math.ceil(len(dataset) / global_batch))
        logger.info(f"Derived epoch length: {cfg.train.OFFICIAL_EPOCH_LENGTH} iterations")
        del dataset
    if cfg.optim.epochs <= 0 or cfg.train.OFFICIAL_EPOCH_LENGTH <= 0:
        raise ValueError("epochs and OFFICIAL_EPOCH_LENGTH must be positive (or use 0 for automatic epoch length)")
    write_config(cfg, cfg.train.output_dir)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
