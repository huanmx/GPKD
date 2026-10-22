torchrun --nproc_per_node=8 -m dinov2.gpkd.train \
    --config-file dinov2/configs/distill/vitb14.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    train.batch_size_per_gpu=128 train.token_path=<PATH/TO/TOKEN/DIR> \
    ibot.loss_weight=1.0 optim.epochs=20 optim.warmup_epochs=5



