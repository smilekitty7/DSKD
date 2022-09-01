# -*- coding: utf-8 -*-  
'''
@author: zhjp   2022/1/13 上午9:28
@file: get_dataset.py
'''
import random
import albumentations as A
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,
    RandomRain, RandomFog, RandomSnow, RandomShadow, RandomSunFlare)
from random import randint, choice, random


def get_dataset(dataset='coco', trainval='val'):
    if dataset == 'hlkt':
        imgs_dir = {
            'train': '/home/softlink/dataset/HLKT-v5/Images/',
            'val': '/home/softlink/dataset/HLKT-v5/Images/',
        }
        anno_file = {
            'train': '/home/softlink/dataset/HLKT-v5/Annotations/detection_train.json',
            'val': '/home/softlink/dataset/HLKT-v5/Annotations/detection_val.json',
        }
    elif dataset == 'wrxt':
        imgs_dir = {
            'train': '/home/softlink/dataset/WRXT/Images/',
            'val': '/home/softlink/dataset/WRXT/Images/',
        }
        anno_file = {
            # 'train': '/home/softlink/dataset/WRXT/Annotations/wrxt_detection_train_v1.json',
            # 'val': '/home/softlink/dataset/WRXT/Annotations/wrxt_detection_val_v1.json',
            'train': '/home/softlink/dataset/WRXT/Annotations/wrxt_detection_train_k6u3.json',
            'val': '/home/softlink/dataset/WRXT/Annotations/wrxt_detection_val_k6u3.json',
        }
    elif dataset == 'coco':
        which = ['', '_mini0.1k', '_mini0.5k', '_mini1k', '_mini2k', '_mini5k', '_mini2w', '_mini3w', '_mini5w']
        trainwhich = '_mini5k'
        valwhich = '_mini1k'
        imgs_dir = {
            'train': '/home/softlink/dataset/COCO2017/train2017/',
            'val': '/home/softlink/dataset/COCO2017/val2017/',
        }
        anno_file = {
            'train': f'/home/softlink/dataset/COCO2017/annotations/instances_train2017{trainwhich}.json',
            'val': f'/home/softlink/dataset/COCO2017/annotations/instances_val2017{valwhich}.json',
        }
    else:
        raise ValueError(f'dataset: {dataset}')
    imgs_dir = imgs_dir[trainval]
    anno_file = anno_file[trainval]
    return imgs_dir, anno_file


transform = A.Compose([
    # A.RandomCrop(width=640, height=640),
    # A.HorizontalFlip(p=1),
    # A.Rotate(p=1),
    # A.RandomCrop(height=480, width=600, p=1),
    # A.RandomBrightnessContrast(p=1),
    # A.ColorJitter(p=1),
    # A.HueSaturationValue(p=1),
    # A.GaussNoise(var_limit=(20.0, 50.0), p=1),
    # A.GaussianBlur(blur_limit=(15, 35), p=1),
    # A.MotionBlur(blur_limit=(15, 35), p=1),
    # A.Cutout(num_holes=18, max_h_size=21, max_w_size=21, fill_value=120, p=1),
    # A.RandomRain(p=1),
    # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=1),
])  # , bbox_params=A.BboxParams(format='coco')


def get_transform(p=1, bbox_format='coco|voc|none'):
    """OK 可以使用"""
    transform = A.Compose([
        # A.RandomCrop(width=640, height=640),
        # A.HorizontalFlip(p=1),
        # A.Rotate(p=1),
        A.RandomCrop(height=360, width=500, p=1),
        # A.RandomBrightnessContrast(brightness_limit=0.7,
        #                            contrast_limit=0.7,
        #                            brightness_by_max=True,
        #                            always_apply=False, p=1),
        # A.ColorJitter(brightness=0.4,
        #               contrast=0.4,
        #               saturation=0.4,
        #               hue=0.4,
        #               always_apply=False, p=0.6),
        # A.HueSaturationValue(hue_shift_limit=40,
        #                      sat_shift_limit=50,
        #                      val_shift_limit=40,
        #                      always_apply=False, p=0.6),
        # A.GaussNoise(var_limit=(20.0, 70.0), p=0.6),
        # A.ISONoise(color_shift=(0.03, 0.09), intensity=(0.3, 0.8), always_apply=False, p=0.6),
        # A.GaussianBlur(blur_limit=(15, 35), p=0.6),
        # A.MotionBlur(blur_limit=(15, 35), p=0.6),
        # A.Cutout(num_holes=randint(15, 40), max_h_size=randint(15, 30),
        #          max_w_size=randint(15, 30), fill_value=randint(100, 220), p=0.6),
        # A.RandomRain(slant_lower=-10,
        #              slant_upper=10,
        #              drop_length=20,
        #              drop_width=1,
        #              drop_color=(200, 200, 200),
        #              blur_value=7,
        #              brightness_coefficient=0.7,
        #              rain_type=None,
        #              always_apply=False, p=0.6),
        # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3,
        #              brightness_coeff=2.5, p=0.6),
        # A.RandomFog(fog_coef_lower=0.5, fog_coef_upper=1,
        #             alpha_coef=0.08, always_apply=False, p=0.6),
        # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5),
        #                  angle_lower=0, angle_upper=1,
        #                  num_flare_circles_lower=6,
        #                  num_flare_circles_upper=10,
        #                  src_radius=400, src_color=(255, 255, 255),
        #                  always_apply=False, p=0.6)
    ])  # , bbox_params=A.BboxParams(format='coco')
    return transform

# def get_data_pipeline(data='qoqo|hlkt'):
#     img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
#                         std=[58.395, 57.12, 57.375], to_rgb=True)
#     min_values = [240, 256, 272, 288, 304, 320,
#                   336, 352, 368, 384, 400][5:]
#     data_pipeline = [
#         # dict(type='LoadImageFromFile'),
#         # dict(type='LoadAnnotations', with_bbox=True),
#         dict(type='Resize',
#              img_scale=[(666, value) for value in min_values],
#              multiscale_mode='value',
#              keep_ratio=True),
#         dict(type='RandomAffine',
#              max_rotate_degree=30.0,
#              max_translate_ratio=0.1,
#              scaling_ratio_range=(1, 1),
#              max_shear_degree=2.0,
#              border=(0, 0),
#              border_val=(114, 114, 114),
#              min_bbox_size=2,
#              min_area_ratio=0.2,
#              max_aspect_ratio=20),
#         dict(type='RandomFlip', flip_ratio=0.5),
#         dict(type='Normalize', **img_norm_cfg),
#         dict(type='Pad', size_divisor=32),
#         dict(type='DefaultFormatBundle'),
#         dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
#     ]
#     # data_pipeline = replace_ImageToTensor(data_pipeline)
#     data_pipeline = Compose(data_pipeline)
#     return data_pipeline
