# -*- coding: utf-8 -*-
'''
@author: zhjp   2021/9/10
@file: manual_augment.py
'''
# dataset settings

dataset_type = 'HLKTDataset'
data_root = '/home/softlink/dataset/HLKT-v5/'

img_norm_cfg = dict(
    mean=[65.957319, 65.957319, 65.957319],
    std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline_baseline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
train_pipeline_singlescale = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),  # +++++++++
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
img_scale_size = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672)
train_pipeline_multiscale = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        ratio_range=None,
        img_scale=[(value, value) for value in img_scale_size],  # +++++++++
        multiscale_mode='value',  # range or value
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),  # +++++++++
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_pipeline_albmscale = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         ratio_range=None,
         img_scale=[(value, value) for value in img_scale_size],  # +++++++++
         multiscale_mode='value',  # range or value
         keep_ratio=True),
    dict(type='Albu',  # +++++++++
         transforms=[
             dict(type='RandomBrightnessContrast',
                  brightness_limit=[-0.3, 0.3],
                  contrast_limit=[-0.3, 0.3], p=0.3),
             dict(type='GaussNoise',
                  var_limit=(30.0, 80.0),
                  mean=0, per_channel=True,
                  always_apply=False, p=0.3),
             dict(type='MotionBlur', blur_limit=(3, 12), p=0.3),
             dict(type='Cutout',
                  num_holes=72, max_h_size=8,
                  max_w_size=8, fill_value=0,
                  always_apply=False, p=0.3)],
         bbox_params=None,
         keymap=None,
         update_pad_shape=False,
         skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),  # +++++++++
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline_singlescale = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline_multiscale = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(640, 640), (640, 320)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline_albument = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Albu',  # +++++++++
                 transforms=[
                     dict(type='RandomBrightnessContrast',
                          brightness_limit=[-0.3, 0.3],
                          contrast_limit=[-0.3, 0.3], p=0.3),
                     dict(type='GaussNoise',
                          var_limit=(30.0, 80.0),
                          mean=0, per_channel=True,
                          always_apply=False, p=0.3),
                     dict(type='MotionBlur', blur_limit=(3, 12), p=0.3),
                     dict(type='Cutout',
                          num_holes=72, max_h_size=8,
                          max_w_size=8, fill_value=0,
                          always_apply=False, p=0.3)],
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_pipeline = [train_pipeline_baseline,
                  train_pipeline_singlescale,
                  train_pipeline_multiscale,
                  train_pipeline_albmscale][3]
test_pipeline = [test_pipeline_singlescale,
                 test_pipeline_multiscale,
                 test_pipeline_albument][2]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/detection_train.json',
        img_prefix=data_root + 'Images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/detection_val.json',
        img_prefix=data_root + 'Images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/detection_val.json',
        img_prefix=data_root + 'Images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
