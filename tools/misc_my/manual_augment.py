# -*- coding: utf-8 -*-  
'''
@author: zhjp   2021/10/10 下午5:59
@file: manual_augment.py
'''

# 在check_dataset_augment.py中进行数据增广测试使用
# 需要train_pipeline格式，test_pipeline格式报错。

import cv2
import numpy as np
from random import random, choice, choices
import albumentations as A

def get_manual_augment(which='v1'):
    # 无Albu的版本
    pipeline_v0 = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(type='RandomFlip', flip_ratio=0.5),
        #### dict(type='Normalize', **img_norm_cfg),    # no need in check_dataset_augment.py
        # dict(type='Pad', size_divisor=32),
        #### dict(type='DefaultFormatBundle'),          # no need in check_dataset_augment.py
        #### dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])  # no need in check_dataset_augment.py
    ]

    # 根据官方Albu-DOC的完整版
    pipeline_v1 = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(
        #     type='Resize',
        #     img_scale=[(666, value) for value in img_scale_size],
        #     multiscale_mode='value',
        #     keep_ratio=True),
        # dict(type='RandomCrop',
        #      crop_size=(0.5, 0.5),
        #      crop_type='relative_range'),
        dict(type='Albu',
             # https://github.com/albumentations-team/albumentations
             # https://albumentations.ai/docs/
             # https://albumentations-demo.herokuapp.com/
             # https://albumentations.ai/docs/api_reference/augmentations/transforms/
             # https://albumentations.ai/docs/api_reference/full_reference/
             transforms=[
                 dict(type='OneOf',  # 几何、形状 ！！基准框失准！！
                      transforms=[
                          dict(type='HorizontalFlip', p=1),
                          dict(type='ShiftScaleRotate',
                               shift_limit=0.0625,
                               scale_limit=0.5,
                               rotate_limit=30,
                               interpolation=1, p=1),
                          dict(type='Perspective',
                               scale=(0.05, 0.1), keep_size=True,
                               pad_mode=0, pad_val=0, mask_pad_val=0,
                               fit_output=False, interpolation=1,
                               always_apply=False, p=1),
                          dict(type='Affine',
                               scale=None, translate_percent=None,
                               translate_px=None, rotate=None,
                               shear=None, interpolation=1,
                               cval=0,  # mask_interpolation=0,
                               cval_mask=0, mode=0, fit_output=False,
                               always_apply=False, p=0.5),
                          dict(type='RandomSizedCrop',
                               min_max_height=(320, 320),
                               height=640, width=640,
                               w2h_ratio=1.0,
                               interpolation=1,
                               always_apply=False, p=1),
                      ], p=1),
                 dict(type='OneOf',  # 亮度、对比度、色彩
                      transforms=[
                          dict(type='Sharpen',
                               alpha=(0.2, 0.5), lightness=(0.5, 1.0),
                               always_apply=False, p=1),
                          dict(type='Solarize',
                               threshold=128, always_apply=False, p=1),
                          dict(type='RandomBrightness',
                               limit=0.2, always_apply=False, p=1),
                          dict(type='RandomContrast',
                               limit=0.2, always_apply=False, p=1),
                          dict(type='RandomBrightnessContrast',
                               brightness_limit=[0.1, 0.5],
                               contrast_limit=[0.1, 0.5], p=1),
                          dict(type='ColorJitter',
                               brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.2,
                               always_apply=False, p=1),
                          dict(type='HueSaturationValue',
                               hue_shift_limit=20, sat_shift_limit=30,
                               val_shift_limit=20, always_apply=False, p=1),
                          dict(type='RandomToneCurve',
                               scale=0.1, always_apply=False, p=1),
                          # dict(type='OpticalDistortion',
                          #      distort_limit=0.05, shift_limit=0.05,
                          #      interpolation=1, border_mode=4,
                          #      value=None, mask_value=None,
                          #      always_apply=False, p=1),       # canot used
                      ], p=1),
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
                      ], p=1),
                 dict(type='OneOf',  # 模糊
                      transforms=[
                          dict(type='Blur', blur_limit=(3, 7), p=1),
                          dict(type='MedianBlur', blur_limit=(3, 7), p=1),
                          dict(type='MotionBlur', blur_limit=(3, 12), p=1),
                          dict(type='GlassBlur', sigma=0.7, max_delta=2, iterations=2, always_apply=False, p=1),
                      ], p=1),
                 dict(type='OneOf',  # 通道操作
                      transforms=[
                          dict(type='ChannelShuffle', p=0.5),
                          dict(type='ChannelDropout',
                               channel_drop_range=(1, 1),
                               fill_value=0,
                               always_apply=False, p=0.5),
                          dict(type='RGBShift',
                               r_shift_limit=20,
                               g_shift_limit=20,
                               b_shift_limit=20,
                               always_apply=False,
                               p=0.5),
                          dict(type='Equalize',
                               mode='cv', by_channels=True,
                               mask=None, mask_params=(),
                               always_apply=False, p=0.5),
                      ], p=1),
                 dict(type='OneOf',  # 区块操作
                      transforms=[
                          dict(type='Cutout',
                               num_holes=choice([8, 10, 12, 24]),
                               max_h_size=choice([8, 16, 24]),
                               max_w_size=choice([8, 16, 24]),
                               fill_value=0,
                               always_apply=False, p=1),
                          # dict(type='MaskDropout',
                          #      max_objects=1,
                          #      image_fill_value=0, mask_fill_value=0,
                          #      always_apply=False, p=1),     # canot use
                          # dict(type='GridDropout',
                          #      ratio=0.5, unit_size_min=None,
                          #      unit_size_max=None, holes_number_x=None,
                          #      holes_number_y=None, shift_x=0, shift_y=0,
                          #      random_offset=False, fill_value=0, mask_fill_value=None,
                          #      always_apply=False, p=1),     # canot use
                          # dict(type='GridDistortion',
                          #      num_steps=5, distort_limit=0.3,
                          #      interpolation=1, border_mode=4,
                          #      value=None, mask_value=None,
                          #      always_apply=False, p=1),    # canot use
                          # dict(type='CoarseDropout',
                          #      max_holes=8, max_height=8,
                          #      max_width=8, min_holes=None,
                          #      min_height=None, min_width=None,
                          #      fill_value=0, mask_fill_value=None,
                          #      always_apply=False, p=1),    # canot use
                      ], p=1),
                 dict(type='OneOf',  # 烟雾、雨点、雪花、耀斑、阴影
                      transforms=[
                          dict(type='RandomGamma',
                               gamma_limit=(80, 120), eps=None,
                               always_apply=False, p=0),  # no need
                          dict(type='RandomFog',
                               fog_coef_lower=0.1,
                               fog_coef_upper=0.5,
                               alpha_coef=0.08,
                               always_apply=False, p=1),
                          dict(type='RandomRain',
                               slant_lower=-10, slant_upper=10,
                               drop_length=choice([6, 8, 10]), drop_width=1, drop_color=(200, 200, 200),
                               rain_type=choice([None, "drizzle", "heavy", "torrential"]),
                               blur_value=choice([1, 3, 5, 7]),
                               brightness_coefficient=choice([1, 0.9, 0.8, 0.7]),
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
                               src_radius=choice([300, 400, 500]),
                               src_color=(255, 255, 255),
                               always_apply=False, p=1),
                          dict(type='RandomShadow',
                               shadow_roi=(0, 0.5, 1, 1),
                               num_shadows_lower=1,
                               num_shadows_upper=3,
                               shadow_dimension=5,
                               always_apply=False, p=1),
                      ], p=1),
             ],
             bbox_params=None,
             keymap=None,
             update_pad_shape=False,
             skip_img_without_anno=False),
        # dict(type='CutOut',
        #      n_holes=(0, 8),
        #      cutout_shape=(32, 32),
        #      cutout_ratio=None,
        #      fill_in=(0, 0, 0)),
        # dict(type='Mosaic',
        #      img_scale=(640, 640),
        #      center_ratio_range=(0.5, 1.5),
        #      pad_val=114),
        # dict(type='MixUp',
        #      img_scale=(640, 640),
        #      ratio_range=(0.5, 1.5),
        #      flip_ratio=0.5,
        #      pad_val=114,
        #      max_iters=15,),
        # dict(type='RandomAffine',
        #      max_rotate_degree=30.0,
        #      max_translate_ratio=0.1,
        #      scaling_ratio_range=(0.5, 1.5),
        #      max_shear_degree=5.0,
        #      border=(0, 0),
        #      border_val=(114, 114, 114),
        #      min_bbox_size=2,
        #      min_area_ratio=0.2,
        #      max_aspect_ratio=20),
        # dict(type='RandomFlip', flip_ratio=0.5),
        #### dict(type='Normalize', **img_norm_cfg),    # no need in check_dataset_augment.py
        dict(type='Pad', size_divisor=32),
        #### dict(type='DefaultFormatBundle'),    # no need in check_dataset_augment.py
        #### dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])  # no need
    ]

    # v1的简化版，适用于HLKT
    pipeline_v2 = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Albu',
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
                      ], p=0),
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
                      ], p=0),
                 dict(type='OneOf',  # 模糊
                      transforms=[
                          dict(type='Blur', blur_limit=(3, 7), p=1),
                          dict(type='MedianBlur', blur_limit=(3, 7), p=1),
                          dict(type='MotionBlur', blur_limit=(3, 12), p=1),
                          dict(type='GlassBlur', sigma=0.7, max_delta=2,
                               iterations=2, always_apply=False, p=1),
                      ], p=0),
                 dict(type='OneOf',  # 区块操作
                      transforms=[
                          dict(type='Cutout',
                               num_holes=72 or choice([8, 10, 12, 24]),
                               max_h_size=8 or choice([8, 16, 24]),
                               max_w_size=8 or choice([8, 16, 24]),
                               fill_value=0,
                               always_apply=False, p=1),
                      ], p=0),
                 dict(type='OneOf',  # 烟雾、雨点、雪花、耀斑、阴影
                      transforms=[
                          dict(type='RandomFog',
                               fog_coef_lower=0.1,
                               fog_coef_upper=0.5,
                               alpha_coef=0.08,
                               always_apply=False, p=1),
                          dict(type='RandomRain',
                               slant_lower=-10, slant_upper=10,
                               drop_length=choice([6, 8, 10]), drop_width=1, drop_color=(200, 200, 200),
                               rain_type=choice([None, "drizzle", "heavy", "torrential"]),
                               blur_value=choice([1, 3, 5, 7]),
                               brightness_coefficient=choice([1, 0.9, 0.8, 0.7]),
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
                               src_radius=choice([300, 400, 500]),
                               src_color=(255, 255, 255),
                               always_apply=False, p=1),
                          dict(type='RandomShadow',
                               shadow_roi=(0, 0.5, 1, 1),
                               num_shadows_lower=1,
                               num_shadows_upper=3,
                               shadow_dimension=5,
                               always_apply=False, p=1),
                      ], p=1),
             ],
             bbox_params=None,
             keymap=None,
             update_pad_shape=False,
             skip_img_without_anno=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        #### dict(type='Normalize', **img_norm_cfg),    # no need in check_dataset_augment.py
        dict(type='Pad', size_divisor=32),
        #### dict(type='DefaultFormatBundle'),          # no need in check_dataset_augment.py
        #### dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])  # no need
    ]

    # 调试和查看单个变换
    pipeline_v3 = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        # dict(
        #     type='Resize',
        #     img_scale=[(666, value) for value in img_scale_size],
        #     multiscale_mode='value',
        #     keep_ratio=True),
        # dict(type='RandomCrop',
        #      crop_size=(0.5, 0.5),
        #      crop_type='relative_range'),
        dict(type='Albu',
             # https://github.com/albumentations-team/albumentations
             # https://albumentations.ai/docs/
             # https://albumentations-demo.herokuapp.com/
             # https://albumentations.ai/docs/api_reference/augmentations/transforms/
             # https://albumentations.ai/docs/api_reference/full_reference/
             transforms=[
                 # dict(type='HorizontalFlip', p=1),
                 # dict(type='RandomSnow',
                 #      snow_point_lower=0.1,
                 #      snow_point_upper=0.3,
                 #      brightness_coeff=2.5,
                 #      always_apply=False, p=1),
                 # dict(type='Cutout',
                 #      num_holes=72 or choice([8, 10, 12, 24]),
                 #      max_h_size=8 or choice([8, 16, 24]),
                 #      max_w_size=8 or choice([8, 16, 24]),
                 #      fill_value=0,
                 #      always_apply=False, p=1),
                 # dict(type='ShiftScaleRotate',
                 #      shift_limit=0.0625,
                 #      scale_limit=0.5,
                 #      rotate_limit=30,
                 #      interpolation=1, p=1),
                 # dict(type='Perspective',
                 #      scale=(0.05, 0.1), keep_size=True,
                 #      pad_mode=0, pad_val=0, mask_pad_val=0,
                 #      fit_output=False, interpolation=1,
                 #      always_apply=False, p=1),
                 # dict(type='Affine',
                 #      scale=None, translate_percent=None,
                 #      translate_px=None, rotate=None,
                 #      shear=None, interpolation=1,
                 #      cval=0,  # mask_interpolation=0,
                 #      cval_mask=0, mode=0, fit_output=False,
                 #      always_apply=False, p=0.5),
                 # dict(type='RandomSizedCrop',
                 #      min_max_height=(320, 320),
                 #      height=640, width=640,
                 #      w2h_ratio=1.0,
                 #      interpolation=1,
                 #      always_apply=False, p=1),
             ],
             bbox_params=None,      #A.BboxParams(format='coco'),
             keymap=None,
             update_pad_shape=False,
             skip_img_without_anno=False),
        # dict(type='RandomFlip', flip_ratio=0.5),
        #### dict(type='Normalize', **img_norm_cfg),    # no need in check_dataset_augment.py
        dict(type='Pad', size_divisor=32),
        #### dict(type='DefaultFormatBundle'),    # no need in check_dataset_augment.py
        #### dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])  # no need
    ]

    pipeline = eval(f'pipeline_{which}')
    return pipeline
