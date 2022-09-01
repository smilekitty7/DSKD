# -*- coding: utf-8 -*-
'''
@author: zhjp   2021/10/5 上午11:41
@file: check_annotate_detection.py
'''

"""
  按 COCO-Style 的格式，比较Albumentation数据增强前后的标注效果
"""

import os
import shutil
import copy
import random
import numpy as np
import json
import cv2
import asyncio
from argparse import ArgumentParser

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.apis import (async_inference_detector, inference_detector, inference,
                        init_detector, show_result_pyplot)
from get_dataset_augment_cfg import get_dataset, get_transform


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='hlkt || coco || wrxt')
    parser.add_argument('--trainval', help='train || val')
    parser.add_argument('--plotbbox', action='store_true', default=True, help='是否绘制BOX')
    args = parser.parse_args()
    return args


def check_imge_detection(args):
    """
    根据Annotaions加载每一张图片，进行检测，检查数据增强前后的标注框是否正确
    """
    # 加载数据
    imgs_dir, anno_file = get_dataset(args.dataset, args.trainval)

    with open(anno_file, 'r') as f:
        anno = json.load(f)

    images = anno['images']
    annotations = anno['annotations']
    categories = anno['categories']
    print(f'\n当前数据集共有图像 {len(images)}张，共有标注 {len(annotations)}，共有类别 {len(categories)}\n')
    if args.random_imgs:
        random.shuffle(images)

    # 遍历检测
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, imgobj in enumerate(images):
        img_id = imgobj['id']
        img_path = imgs_dir + imgobj['file_name']
        print('图像路径 => ', img_path)
        org_img = cv2.imread(img_path)
        cv2.imshow(f'{args.dataset}-org_img', org_img)

        # 整理所有原始基准框
        org_anno_list = [a for a in annotations if a['image_id'] == img_id]
        org_gt_bboxes = []
        for i, target in enumerate(org_anno_list):
            x1, y1, w, h = [int(x) for x in target['bbox']]
            category_id = target['category_id']
            category_name = [cat['name'] for cat in categories if cat['id'] == category_id][0]
            org_gt_bboxes.append([x1, x1 + w, y1, y1 + h, category_name])

        # 图像增强变换 ALbument Augment
        transform = get_transform(p=1, bbox_format='coco')
        transformed = transform(image=org_img)  # , bboxes=org_gt_bboxes
        trans_image = transformed['image']
        trans_gt_bboxes = org_gt_bboxes # transformed['bboxes']
        cv2.imshow(f'{args.dataset}-trans_image', trans_image)

        # # 绘制原始基准框
        if args.plotbbox:
            for i, bbox in enumerate(org_gt_bboxes):
                x1, x2, y1, y2, cate = int(bbox[0]), int(bbox[1]), \
                                       int(bbox[2]), int(bbox[3]), bbox[4]
                text = str(i) + ':' + cate
                cv2.rectangle(org_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(org_img, text, (x1, int(y1 - 5)), font, 0.4, (255, 255, 255), 1)
                cv2.putText(org_img, f'ImgId: {img_id}', (20, 20), font, 0.4, (0, 255, 0), 1)
                cv2.imshow(f'{args.dataset}-org_img', org_img)

            # # 绘制变换后基准框
            for i, bbox in enumerate(trans_gt_bboxes):
                x1, x2, y1, y2, cate = int(bbox[0]), int(bbox[1]), \
                                       int(bbox[2]), int(bbox[3]), bbox[4]
                text = str(i) + ':' + cate
                cv2.rectangle(trans_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 不区分顶点的坐标方位
                cv2.putText(trans_image, text, (x1, int(y1 - 5)), font, 0.4, (255, 255, 255), 1)
                cv2.putText(trans_image, f'ImgId: {img_id}', (20, 20), font, 0.4, (0, 255, 0), 1)
                cv2.imshow(f'{args.dataset}-trans_image', trans_image)

        cv2.waitKey(args.plot_interval)


if __name__ == '__main__':
    args = parse_args()
    args.dataset = 'wrxt'      # hlkt, coco, wrxt
    args.trainval = 'train'    # train & val
    args.plotbbox = False
    args.random_imgs = False
    args.plot_interval = -5000
    args.device = 'cuda:0'

    print(f'\n输入参数: {args}\n')

    check_imge_detection(args)

