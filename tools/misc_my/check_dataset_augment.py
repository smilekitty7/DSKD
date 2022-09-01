# -*- coding: utf-8 -*-  
'''
@author: zhjp   2021/10/10 上午9:41
@file: check_dataset_augment.py
'''

import argparse
import copy
import os
import random
from collections import Sequence
from pathlib import Path

import cv2
import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset
from manual_augment import get_manual_augment


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument(
        '--config',
        # default='../../configs/_base_/datasets/hlkt_detection.py',
        # default='../../configs/_base_/datasets/qoqo_detection.py',
        # default='../../configs/_base_/datasets/wrxt_detection.py',
        default='../../configs/_base_/datasets/mini_detection.py',
        # default='../../configs/_base_/datasets/coco_detection_yolox.py',
        # default='../../configs/_base_/datasets/qoqo_detection_albucomplex.py',
        help='train config file path')
    parser.add_argument('--trainval', default='val', action='store_true', help='train|val|test')
    parser.add_argument('--manual-aug', default='v0', help='train|val|test, None(只在train下工作),v0 v1 v2 v3 work in'
                                                       '.tools/misc_my/manual_augment.py')
    parser.add_argument('--shuffle-img', default='False', action='store_true', help='shuffle images')
    parser.add_argument('--show-img', default=1, action='store_true', help='显示原图与变换图,按各自尺寸独立显示2张')
    parser.add_argument('--show-stich', default=1, action='store_true', help='显示原图与变换图，将后者尺寸按原图缩放')
    parser.add_argument('--show-gtbox', default=1, action='store_true', help='显示变换图与其上的基准框，检查变换后基准框是否正确')
    parser.add_argument('--show-interval', default=-5, type=float, help='the interval of show (s)')
    parser.add_argument('--save-img', default=0, action='store_true', help='保存变换后的图像')
    parser.add_argument('--show-random', default=0, type=float, help='随机选择部分图像')
    parser.add_argument('--output-dir', type=str,
                        default='/home/softlink/experiments/common_exp2',
                        help='图像保存路径，If there is no display interface, you can save it')

    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, trainval, skip_type, cfg_options):
    def skip_pipeline_steps(config):
        config['pipeline'] = [x for x in config.pipeline if x['type'] not in skip_type]

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    data_cfg = cfg.data.get(trainval, None)
    while 'dataset' in data_cfg and data_cfg['type'] != 'MultiImageMixDataset':
        data_cfg = data_cfg['dataset']
    if isinstance(data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in data_cfg]
    else:
        skip_pipeline_steps(data_cfg)

    return data_cfg


def main():
    args = parse_args()

    data_cfg = retrieve_data_cfg(args.config, args.trainval, args.skip_type, args.cfg_options)

    if args.manual_aug is not None:
        data_cfg.pipeline = get_manual_augment(which=args.manual_aug)

    if args.output_dir is not None and not os.path.exists(args.output_dir):
        print(f'创建新文件夹 => {args.output_dir}')
        os.makedirs(args.output_dir)

    # debug可查看数据集各类别数量分布
    print(f'data_cfg ===> {data_cfg}')
    dataset = build_dataset(data_cfg)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=1,
    #     dist=False,
    #     shuffle=True)

    progress_bar = mmcv.ProgressBar(len(dataset))

    avg_area_ratio = []
    # 统计各个尺度的BOX数量和百分比，[0,32),[32, 64),[64, 96), [96, 128),...,[512, imgH*W)
    box_area_nums = {32: 0, 64: 0, 96: 0, 128: 0, 160: 0, 192: 0, 224: 0, 256: 0,
                     288: 0, 320: 0, 352: 0, 384: 0, 416: 0, 448: 0, 480: 0, 512: 0}
    box_area_keys = list(box_area_nums.keys())
    for idx, item in enumerate(dataset):

        if random.random() < args.show_random:  # 随机挑选部分图片
            continue

        win_name = f'{args.trainval}'
        filename = Path(item['filename']).name
        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            filename = os.path.join(args.output_dir, Path(item['filename']).name)

        gt_masks = item.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)

        if args.show_img:
            org_img = cv2.imread(item['filename'])
            new_img = item['img'].astype(np.uint8)

            if args.show_gtbox:
                bboxes, labels = item['gt_bboxes'], item['gt_labels']
                class_names = dataset.ALL_CLASSES
                font = cv2.FONT_ITALIC
                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    x1, y1, x2, y2 = bbox.astype(np.int32)
                    label_text = class_names[label] if class_names is not None else f'class {label}'
                    cv2.rectangle(new_img, (x1, y1), (x2, y2), (200, 50, 20), 1)
                    cv2.rectangle(new_img, (x1, y1), (x1 + len(label_text) * 7, int(y1 - 14)), (170, 50, 10),
                                  cv2.FILLED)
                    cv2.putText(new_img, label_text, (x1, int(y1 - 5)), font, 0.4, (255, 255, 255), 1)
                    # 统计框的面积
                    idx_area = [(x2 - x1) * (y2 - y1)//(x * x) for x in box_area_keys]
                    if 0 in idx_area:
                        idx_area = idx_area.index(0)
                    else:
                        idx_area = len(box_area_keys) - 1
                    box_area_nums[box_area_keys[idx_area]] += 1
                # print('\n =>', box_area_nums)

            if args.show_stich:
                # org_new_img = stich_many_imgs(scale=1, imgarray=[org_img, new_img])
                org_new_img = np.hstack([org_img, cv2.resize(new_img, org_img.shape[:2][::-1])])
                cv2.imshow('org_new_img', org_new_img)
                if args.save_img and args.output_dir:
                    quality = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
                    cv2.imwrite(filename, org_new_img, quality)
            else:
                org_ratio = round(org_img.shape[0] / org_img.shape[1], 2)
                new_ratio = round(new_img.shape[0] / new_img.shape[1], 2)
                org_new_ratio = round(org_ratio / new_ratio, 2)
                area_ratio = new_img.shape[0] * new_img.shape[1] / org_img.shape[0] / org_img.shape[1]
                avg_area_ratio.append(area_ratio)
                print(f'{idx}\t\torg_img-{org_img.shape} new_img-{new_img.shape}\t',
                      f'org-new-ratio：{org_ratio} {new_ratio} {org_new_ratio}\t'
                      f'new-org-area-ratio：{round(area_ratio, 2)}\t'
                      f'average_area_ratio: {round(sum(avg_area_ratio) / len(avg_area_ratio), 2)}')
                cv2.imshow('org_img', org_img)
                cv2.imshow('new_img', new_img)
                if args.save_img and args.output_dir:
                    quality = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
                    cv2.imwrite(filename, new_img, quality)

            cv2.waitKey(args.show_interval)

        if args.show_gtbox and None:
            imshow_det_bboxes(
                item['img'],
                item['gt_bboxes'],
                item['gt_labels'],
                gt_masks,
                class_names=dataset.CLASSES,
                win_name=win_name,
                show=args.show_gtbox,
                wait_time=2 or np.abs(args.show_interval),
                out_file=filename,
                bbox_color=(255, 102, 61),
                text_color=(255, 102, 61))

        progress_bar.update()

    total_boxes = max(sum(box_area_nums.values()), 1)
    box_area_percent = {k: round(v/total_boxes, 4)*100 for (k, v) in box_area_nums.items()}
    print(f'\n统计盒子尺寸数量==>', box_area_nums)
    print(f'统计盒子尺寸百分比==>', box_area_percent)
    print(f'统计盒子尺寸数量==>', box_area_nums.values())
    print(f'统计盒子尺寸百分比==>', box_area_percent.values())


if __name__ == '__main__':
    main()
