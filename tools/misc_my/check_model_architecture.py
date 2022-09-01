# -*- coding:utf-8 -*-
# @Author: xx
# @Time: 2020/9/18 22:21
# software: PyCharm

"""
   打印查看模型架构
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
import warnings

import mmcv
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from IPython import embed

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file || None')
    parser.add_argument('--device', default='cuda:1', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def check_model_arch(args):
    """
    根据Annotaions加载每一张图片，进行检测，检查检测结果与标注框是否相合。
    """
    # 加载模型
    # build the model from a config file and a checkpoint file

    print('\n########### Model Configs #################\n')
    config = mmcv.Config.fromfile(args.config)
    config.model.pretrained = None
    config.model.train_cfg = None
    # model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    # print(config)
    # embed()
    print('\n########### Model Architecture Start #################\n')
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print(model)
    print('\n########### Model Architecture End #################\n')

if __name__ == '__main__':
    args = parse_args()
    if args.config == None:
        # args.config = '../../configs/agg_rcnn/agg_r50_fpn_qoqo.py'
        # args.config = '../../configs/agg_rcnn/agg_r50_fpn_hlkt.py'
        # args.config = '../../configs/yolof/yolof_r50_c5_hlkt.py'
        # args.config = '../../configs/deformable_detr/deformable_detr_r50_1x_hlkt.py'
        args.config = '../../configs/tridentnet/tridentnet_r50_caffe_1x_coco.py'
        # args.config = '../../configs/sparse_rcnn/sparse_rcnn_reg3.2_fpn_mstrain_hlkt.py'
        # args.config = '../../configs/xyz_rcnn/xyz_rcnn_pvtb2_xpn_hlkt.py'
        # args.config = '../../configs/xyz_rcnn/xyz_rcnn_swint_xpn_hlkt.py'
        # args.config = '../../configs/deformable_detr/deformable_detr_r50_1x_hlkt.py'
        # args.config = 'configs/paa/paa_r50_fpn_1x_hlkt.py'
    if args.checkpoint == None:
        args.checkpoint = None
    args.device = 'cuda:0'
    print(f'\n输入参数: {args}\n')
    check_model_arch(args)

