# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/softlink/dataset/COCO2017/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# TODO 按比例缩减图像尺寸为原来的一半 666/400 = 666.5/400 = 1.66625，均值方差重新计算？？
# TODO 缺400~666的正方形，比例覆盖不完全，是否影响检测？？
# 训练设置
img_scale_size = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640]    # 用于多尺度训练，value模式
# img_scale_size=[(640, value) for value in img_scale_size],                # 尺寸大部分集中于min/max中
# img_scale_size=[(w, h) for w in img_scale_size for h in img_scale_size],  # 尺寸分散于所有尺寸中.
# 测试设置
max_min_size = (640, 512)           # 用于测试，==min(大/长，小/短)*(长，短)，小值不可太小，否则小/短始终小于大/长，应该≥512！
# ① keep_ratio=False,则转换后尺寸为: 固定的 (W, H) = (max_min_size[0], max_min_size[1])
# ② keep_ratio=True, 则转换后尺寸为: 当(大/长，小/短)缩放比接近时 max(W, H)<=max_size, min(W,H)可＜min_size!
#                                  当(大/长，小/短)缩放比过大时 min(W, H)>=min_size, 但max(W, H)不可＞max_size!
train_pipeline_singlescale = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=max_min_size,
         multiscale_mode='range',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
train_pipeline_multiscale = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         # img_scale=[(640, value) for value in img_scale_size],
         img_scale=[(w, h) for w in img_scale_size for h in img_scale_size],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_pipeline_phdrccms = [
    # copy form centernet_resnet18_dcnv2_140e_coco.py  #黑Pad的太多
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='RandomCenterCropPad',
         crop_size=(512, 512),
         ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
         mean=[0, 0, 0],
         std=[1, 1, 1],
         to_rgb=True,
         test_pad_mode=None),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
train_pipeline_albmscale = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=[(w, h) for w in img_scale_size for h in img_scale_size],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='Albu',  # +++++++++
         transforms=[
             dict(type='OneOf',
                  transforms=[
                      dict(type='RandomBrightnessContrast',
                           brightness_limit=[-0.3, 0.3],
                           contrast_limit=[-0.3, 0.3], p=1),
                      dict(type='ColorJitter',
                           brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.2,
                           always_apply=False, p=1),
                  ], p=0.3),
             dict(type='OneOf',
                  transforms=[
                      dict(type='GaussNoise',
                           var_limit=(30.0, 80.0),
                           mean=0, per_channel=True,
                           always_apply=False, p=1),
                      dict(type='MotionBlur',
                           blur_limit=(3, 12), p=1),
                  ], p=0.3),
             dict(type='Cutout',
                  num_holes=72, max_h_size=16,
                  max_w_size=16, fill_value=123,
                  always_apply=False, p=0.3)],
         bbox_params=None,
         keymap=None,
         update_pad_shape=False,
         skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline_singlescale = [
    # 与 train_pipeline_singlescale效果相同
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=max_min_size,
         flip=False,
         transforms=[
             dict(type='Resize',
                  multiscale_mode='range',
                  keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]
test_pipeline_multiscale = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         # img_scale=[(640, value) for value in img_scale_size],
         img_scale=[(w, h) for w in img_scale_size for h in img_scale_size],
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

train_pipeline = [train_pipeline_singlescale,
                  train_pipeline_multiscale,
                  train_pipeline_phdrccms,
                  train_pipeline_albmscale][0]
test_pipeline = [test_pipeline_singlescale,
                 test_pipeline_multiscale][0]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
