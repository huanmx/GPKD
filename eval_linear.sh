torchrun --nproc_per_node=8 -m dinov2.eval.linear \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    --config-file dinov2/configs/eval/vitb14_pretrain.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --batch-size 256
