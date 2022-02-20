_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/gate_segmentation.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(align_corners=True, in_channels=512, channels=128, num_classes=2),
    auxiliary_head=dict(align_corners=True, in_channels=256, channels=64, num_classes=2))
# I removed test_cfg, is this needed?
