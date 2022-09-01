# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/softlink/dataset/COCO2017/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# TODO 按比例缩减图像尺寸为原来的一半 666/400 = 666.5/400 = 1.66625，均值方差重新计算？？
# TODO 缺400~666的正方形，比例覆盖不完全，是否影响检测？？
# img_scale_size = [320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 500, 516, 532, 548, 564, 580, 596]
img_scale_size = [320, 352, 384, 416, 448, 500, 532, 564, 596, 628, 666]
img_minmax_size = [320, 666]
train_pipeline_singlescale = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(666, 400), keep_ratio=True),
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
         img_scale=[(666, value) for value in img_scale_size],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
alb_train_p0 = [0.0, 0.0, 0.0, 0.0, 0.0]
alb_train_p2 = [0.3, 0.3, 0.3, 0.3, 0.6]
alb_train_p3 = [0.8, 0.8, 0.8, 0.8, 0.8]
alb_train_p4 = [0.6, 0.6, 0.6, 0.6, 0.8]
alb_train_p = alb_train_p4.copy()
train_pipeline_albmscale = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=[(666, value) for value in img_scale_size],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='Albu',  # +++++++++
         transforms=[
             dict(type='OneOf',  # 亮度、对比度、饱和度、色调、色彩
                  transforms=[
                      dict(type='RandomBrightnessContrast',
                           brightness_limit=[-0.3, 0.3],
                           contrast_limit=[-0.3, 0.3], p=1),
                      dict(type='ColorJitter',
                           brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.2,
                           always_apply=False, p=1),
                      dict(type='HueSaturationValue',
                           hue_shift_limit=20, sat_shift_limit=30,
                           val_shift_limit=20, always_apply=False, p=1),
                  ], p=alb_train_p[0]),
             dict(type='OneOf',  # 噪声
                  transforms=[
                      dict(type='ISONoise',
                           color_shift=(0.01, 0.05),
                           intensity=(0.1, 0.5),
                           always_apply=False, p=1),
                      dict(type='GaussNoise',
                           var_limit=(30.0, 80.0),
                           mean=0, per_channel=True,
                           always_apply=False, p=1),
                      dict(type='MultiplicativeNoise',
                           multiplier=(0.9, 1.1),
                           per_channel=False,
                           elementwise=False,
                           always_apply=False, p=1),
                  ], p=alb_train_p[1]),
             dict(type='OneOf',  # 模糊
                  transforms=[
                      dict(type='Blur', blur_limit=(3, 7), p=1),
                      dict(type='MedianBlur', blur_limit=(3, 7), p=1),
                      dict(type='MotionBlur', blur_limit=(3, 12), p=1),
                      dict(type='GlassBlur', sigma=0.7, max_delta=2,
                           iterations=2, always_apply=False, p=1),
                  ], p=alb_train_p[2]),
             dict(type='OneOf',  # 区块
                  transforms=[
                      dict(type='Cutout',
                           num_holes=72, max_h_size=8,
                           max_w_size=8, fill_value=0,
                           always_apply=False, p=1),
                  ], p=alb_train_p[3]),
             dict(type='OneOf',  # 烟雾、雨点、雪花、耀斑、阴影
                  transforms=[
                      dict(type='RandomFog',
                           fog_coef_lower=0.1,
                           fog_coef_upper=0.5,
                           alpha_coef=0.08,
                           always_apply=False, p=1),
                      dict(type='RandomRain',
                           slant_lower=-10, slant_upper=10,
                           drop_length=8, drop_width=1, drop_color=(200, 200, 200),
                           rain_type="drizzle",
                           blur_value=3,
                           brightness_coefficient=0.9,
                           always_apply=False, p=1),
                      dict(type='RandomSnow',
                           snow_point_lower=0.1,
                           snow_point_upper=0.3,
                           brightness_coeff=2.5,
                           always_apply=False, p=1),
                      dict(type='RandomSunFlare',
                           flare_roi=(0, 0, 1, 0.5),
                           angle_lower=0, angle_upper=1,
                           num_flare_circles_lower=6,
                           num_flare_circles_upper=10,
                           src_radius=400,
                           src_color=(255, 255, 255),
                           always_apply=False, p=1),
                      dict(type='RandomShadow',
                           shadow_roi=(0, 0.5, 1, 1),
                           num_shadows_lower=1,
                           num_shadows_upper=3,
                           shadow_dimension=5,
                           always_apply=False, p=1),
                  ], p=alb_train_p[4])],
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
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(666, 400),
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
    dict(type='MultiScaleFlipAug',
         img_scale=[(666, 400), (666, 512)],
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
alb_test_p0 = [0.0, 0.0, 0.0, 0.0, 0.0]
alb_test_p2 = [0.3, 0.3, 0.3, 0.3, 0.6]
alb_test_p3 = [0.8, 0.8, 0.8, 0.8, 0.8]
alb_test_p4 = [0.6, 0.6, 0.6, 0.6, 0.8]
alb_test_p = alb_test_p3.copy()
test_pipeline_albument = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(666, 400),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='Albu',  # +++++++++
                  transforms=[
                      dict(type='OneOf',  # 亮度、对比度、饱和度、色调、色彩
                           transforms=[
                               dict(type='RandomBrightnessContrast',
                                    brightness_limit=[-0.3, 0.3],
                                    contrast_limit=[-0.3, 0.3], p=1),
                               dict(type='ColorJitter',
                                    brightness=0.2, contrast=0.2,
                                    saturation=0.2, hue=0.2,
                                    always_apply=False, p=1),
                               dict(type='HueSaturationValue',
                                    hue_shift_limit=20, sat_shift_limit=30,
                                    val_shift_limit=20, always_apply=False, p=1),
                           ], p=alb_test_p[0]),
                      dict(type='OneOf',  # 噪声
                           transforms=[
                               dict(type='ISONoise',
                                    color_shift=(0.01, 0.05),
                                    intensity=(0.1, 0.5),
                                    always_apply=False, p=1),
                               dict(type='GaussNoise',
                                    var_limit=(30.0, 80.0),
                                    mean=0, per_channel=True,
                                    always_apply=False, p=1),
                               dict(type='MultiplicativeNoise',
                                    multiplier=(0.9, 1.1),
                                    per_channel=False,
                                    elementwise=False,
                                    always_apply=False, p=1),
                           ], p=alb_test_p[1]),
                      dict(type='OneOf',  # 模糊
                           transforms=[
                               dict(type='Blur', blur_limit=(3, 7), p=1),
                               dict(type='MedianBlur', blur_limit=(3, 7), p=1),
                               dict(type='MotionBlur', blur_limit=(3, 12), p=1),
                               dict(type='GlassBlur', sigma=0.7, max_delta=2,
                                    iterations=2, always_apply=False, p=1),
                           ], p=alb_test_p[2]),
                      dict(type='OneOf',  # 区块
                           transforms=[
                               dict(type='Cutout',
                                    num_holes=72, max_h_size=8,
                                    max_w_size=8, fill_value=0,
                                    always_apply=False, p=1),
                           ], p=alb_test_p[3]),
                      dict(type='OneOf',  # 烟雾、雨点、雪花、耀斑、阴影
                           transforms=[
                               dict(type='RandomFog',
                                    fog_coef_lower=0.1,
                                    fog_coef_upper=0.5,
                                    alpha_coef=0.08,
                                    always_apply=False, p=1),
                               dict(type='RandomRain',
                                    slant_lower=-10, slant_upper=10,
                                    drop_length=8, drop_width=1, drop_color=(200, 200, 200),
                                    rain_type="drizzle",
                                    blur_value=3,
                                    brightness_coefficient=0.9,
                                    always_apply=False, p=1),
                               dict(type='RandomSnow',
                                    snow_point_lower=0.1,
                                    snow_point_upper=0.3,
                                    brightness_coeff=2.5,
                                    always_apply=False, p=1),
                               dict(type='RandomSunFlare',
                                    flare_roi=(0, 0, 1, 0.5),
                                    angle_lower=0, angle_upper=1,
                                    num_flare_circles_lower=6,
                                    num_flare_circles_upper=10,
                                    src_radius=400,
                                    src_color=(255, 255, 255),
                                    always_apply=False, p=1),
                               dict(type='RandomShadow',
                                    shadow_roi=(0, 0.5, 1, 1),
                                    num_shadows_lower=1,
                                    num_shadows_upper=3,
                                    shadow_dimension=5,
                                    always_apply=False, p=1),
                           ], p=alb_test_p[4])],
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
train_pipeline = [train_pipeline_singlescale,
                  train_pipeline_multiscale,
                  train_pipeline_albmscale][2]
test_pipeline = [test_pipeline_singlescale,
                 test_pipeline_multiscale,
                 test_pipeline_albument][0]

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
