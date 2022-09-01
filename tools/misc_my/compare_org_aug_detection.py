# -*- coding: utf-8 -*-
'''
@author: zhjp   2021/10/5 上午11:41
@file: check_annotate_detection.py
'''

"""
  按 COCO-Style 的格式，比较Albumentation数据增强前后的检测识别效果
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
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def check_imge_detection(args):
    """
    根据Annotaions加载每一张图片，进行检测，检查检测结果与标注框是否相合
    """
    # 加载模型
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print('\n########### Model Architecture #################\n')
    print(model)
    print('\n########### Model Architecture #################\n')

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
        # cv2.imshow(f'{args.dataset}-org_img', org_img)

        # 整理所有原始基准框
        org_anno_list = [a for a in annotations if a['image_id'] == img_id]
        org_gt_bboxes = []
        for i, target in enumerate(org_anno_list):
            x1, y1, w, h = [int(x) for x in target['bbox']]
            category_id = target['category_id']
            category_name = [cat['name'] for cat in categories if cat['id'] == category_id][0]
            org_gt_bboxes.append([x1, x1 + w, y1, y1 + h, category_name])

        # # 绘制原始基准框
        for i, bbox in enumerate(org_gt_bboxes):
            x1, x2, y1, y2, cate = int(bbox[0]), int(bbox[1]), \
                                   int(bbox[2]), int(bbox[3]), bbox[4]
            text = str(i) + ':' + cate
            cv2.rectangle(org_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(org_img, text, (x1, int(y1 - 5)), font, 0.4, (255, 255, 255), 1)
            cv2.putText(org_img, f'ImgId: {img_id}', (20, 20), font, 0.4, (0, 255, 0), 1)
            cv2.imshow(f'{args.dataset}-org_img-gtbox', org_img)

        # 图像增强变换 ALbument Augment
        transform = get_transform(p=1, bbox_format='none')
        transformed = transform(image=org_img)      # , bboxes=org_gt_bboxes
        trans_image = transformed['image']
        trans_gt_bboxes = org_gt_bboxes     # transformed['bboxes']

        # 绘制变换后基准框
        for i, bbox in enumerate(trans_gt_bboxes):
            x1, x2, y1, y2, cate = int(bbox[0]), int(bbox[1]), \
                                   int(bbox[2]), int(bbox[3]), bbox[4]
            text = str(i) + ':' + cate
            cv2.rectangle(trans_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 不区分顶点的坐标方位
            cv2.putText(trans_image, text, (x1, int(y1 - 5)), font, 0.4, (255, 255, 255), 1)
            cv2.putText(trans_image, f'ImgId: {img_id}', (20, 20), font, 0.4, (0, 255, 0), 1)
            cv2.imshow(f'{args.dataset}-trans_image-gtbox', trans_image)

        detect_img = [img_path, org_img, trans_image][1:]

        # 检测识别过程
        for idx, image in enumerate(detect_img):
            det_bboxes, det_labels, det_image = \
                inference_image(model, image.copy(), categories, imgidx=str(idx),
                                score_thr=args.score_thr, dataset=args.dataset)
            print(f'检测结果-detimage-{idx} => {det_bboxes}， {det_labels}\n')
            cv2.imshow(f'{args.dataset}-detimage-{idx}', det_image)

        cv2.waitKey(args.plot_interval)


def inference_image(model, image, categories, imgidx='', score_thr=0.3, dataset='coco'):
    """
    Args:
        model:
        image: 可以是加载后的ndarray或图像路径
        categories:
        imgidx:
    """
    # 检测识别过程
    result = inference_detector(model, image)
    # show_result_pyplot(model, img_path, result, score_thr=score_thr)
    det_bboxes, det_labels = inference_filter(result, score_thr=score_thr)
    # print(f'检测结果 => {det_bboxes}， {det_labels}\n')
    # 绘制检测框
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (bbox, label) in enumerate(zip(det_bboxes, det_labels)):
        x1, y1, x2, y2, score = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), bbox[4]
        cat_id, cat_name = None, 'None'  # 为当前label找到其类别编号和类别名称
        for idx, cat in enumerate(categories):
            if dataset in ['coco']:
                cat_id = idx  # ！cat['id']有跳数，最大为90，因此使用其列表编号！
            elif dataset in ['hlkt', 'wrxt']:
                cat_id = cat['id']
            if cat_id == label:
                cat_name = cat['name']
                break
        text = cat_name + ' ' + str(score)[:4]
        cv2.rectangle(image, (x1, y1), (x2, y2), (1, 255, 1), 1)
        cv2.rectangle(image, (x1, y1), (x1 + len(text) * 7, int(y1 - 18)), (1, 255, 1), cv2.FILLED)
        cv2.putText(image, text, (x1 + 2, int(y1 - 5)), font, 0.4, (10, 11, 160), 1)
    det_image = image
    return det_bboxes, det_labels, det_image


def inference_filter(result, score_thr=0.3):
    """
    xmdet216\mmdet\models\detectors\base.py
    xmdet216\mmdet\core\visualization\image.py
    show_result() & imshow_det_bboxes()
    """
    # 过滤多余的BBOX
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    # print(f'bboxes ==> {bboxes}，labels ==> {labels}')
    return bboxes, labels


if __name__ == '__main__':
    args = parse_args()
    # args.config = '../../configs/xyz_rcnn/xyz_rcnn_pvtb2_fpn_qoqo.py'
    # args.config = '../../configs/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_2x_hlkt.py'
    # args.config = '../../configs/yoloy/yoloy_resnet_8x4_1x_wrxt.py'
    args.config = '../../configs/yolox/yolox_resnet_wrxt.py'
    # args.config = '../../configs/faster_rcnn/faster_resnet_fpn_wrxt.py'
    args.checkpoint = None
    # args.checkpoint = '/home/softlink/Pretrained/xyz_pvtb2_fpn_8x4_1x_qoqo_epoch_12.pth'
    # args.checkpoint = '/home/softlink/Pretrained/sparse_r50_fpn_mstrain_480-640_2x_hlkt_epoch_24.pth'
    args.checkpoint = '/home/softlink/experiments/wrxt-yolox-r18-stst-k6u3/epoch_12.pth'
    # args.checkpoint = '/home/softlink/experiments/wrxt-faster-r18-stst/epoch_12.pth'
    args.dataset = 'wrxt'      # hlkt, coco, wrxt
    args.trainval = 'val'    # train & val
    args.which = 'detect'    # annotate & detect
    args.random_imgs = False
    args.plot_interval = -5000
    args.device = 'cuda:0'
    args.score_thr = 0.3

    print(f'\n输入参数: {args}\n')

    check_imge_detection(args)

