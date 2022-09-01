# VOC_CATS          # 21 类别
import copy
import random

import numpy as np
from collections import OrderedDict

import torch

VOC_CATS = ['tv', 'chair', 'bottle', 'boat', 'cow', 'dog', 'sheep',
            'horse', 'bird', 'dining table', 'bicycle', 'cat',
            'potted plant', '__background__', 'train', 'car',
            'couch', 'person', 'airplane', 'motorcycle', 'bus']

# OCO_NONVOC_CATS  # 61 类别
COCO_NONVOC_CATS = ['apple', 'backpack', 'banana', 'baseball bat',
                    'baseball glove', 'bear', 'bed', 'bench', 'book', 'bowl',
                    'broccoli', 'cake', 'carrot', 'cell phone', 'clock', 'cup',
                    'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee',
                    'giraffe', 'hair drier', 'handbag', 'hot dog', 'keyboard',
                    'kite', 'knife', 'laptop', 'microwave', 'mouse', 'orange',
                    'oven', 'parking meter', 'pizza', 'refrigerator', 'remote',
                    'sandwich', 'scissors', 'sink', 'skateboard', 'skis',
                    'snowboard', 'spoon', 'sports ball', 'stop sign',
                    'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
                    'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
                    'truck', 'umbrella', 'vase', 'wine glass', 'zebra']

# COCO_CATS = COCO_VOC_CATS+COCO_NONVOC_CATS  # 81 类别
COCO_CATS = ['__background__', 'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear',
             'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car',
             'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut',
             'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog',
             'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven',
             'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors',
             'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase',
             'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
             'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

# MM中按照类别ID进行排序
COCO_CATS_IDSX = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8,
                  'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
                  'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23,
                  'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33,
                  'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39,
                  'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44,
                  'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52,
                  'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59,
                  'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67,
                  'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77,
                  'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85,
                  'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}

# 增量学习中按照类别名的拼音排序
COCO_CATS_IDS = {'airplane': 5, 'apple': 53, 'backpack': 27, 'banana': 52, 'baseball bat': 39,
                 'baseball glove': 40, 'bear': 23, 'bed': 65, 'bench': 15, 'bicycle': 2, 'bird': 16,
                 'boat': 9, 'book': 84, 'bottle': 44, 'bowl': 51, 'broccoli': 56, 'bus': 6, 'cake': 61,
                 'car': 3, 'carrot': 57, 'cat': 17, 'cell phone': 77, 'chair': 62, 'clock': 85, 'couch': 63,
                 'cow': 21, 'cup': 47, 'dining table': 67, 'dog': 18, 'donut': 60, 'elephant': 22,
                 'fire hydrant': 11, 'fork': 48, 'frisbee': 34, 'giraffe': 25, 'hair drier': 89, 'handbag': 31,
                 'horse': 19, 'hot dog': 58, 'keyboard': 76, 'kite': 38, 'knife': 49, 'laptop': 73,
                 'microwave': 78, 'motorcycle': 4, 'mouse': 74, 'orange': 55, 'oven': 79, 'parking meter': 14,
                 'person': 1, 'pizza': 59, 'potted plant': 64,  'refrigerator': 82, 'remote': 75, 'sandwich': 54,
                 'scissors': 87, 'sheep': 20, 'sink': 81, 'skateboard': 41, 'skis': 35, 'snowboard': 36, 'spoon': 50,
                 'sports ball': 37, 'stop sign': 13, 'suitcase': 33, 'surfboard': 42, 'teddy bear': 88,
                 'tennis racket': 43, 'tie': 32, 'toaster': 80, 'toilet': 70, 'toothbrush': 90, 'traffic light': 10,
                 'train': 7, 'truck': 8, 'tv': 72, 'umbrella': 28, 'vase': 86, 'wine glass': 46, 'zebra': 24}
COCO_CATS_IDS = OrderedDict(COCO_CATS_IDS)


COCO_LABEL_MAP = {  # 93, 名称不完整
    0: "unlabeled", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train",
    8: "truck", 9: "boat",
    10: "traffic", 11: "fire", 12: "street", 13: "stop", 14: "parking", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
    19: "horse",
    20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 26: "hat", 27: "backpack",
    28: "umbrella", 29: "shoe",
    30: "eye", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports",
    38: "kite", 39: "baseball",
    40: "baseball", 41: "skateboard", 42: "surfboard", 43: "tennis", 44: "bottle", 45: "plate", 46: "wine", 47: "cup",
    48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted", 65: "bed", 66: "mirror", 67: "dining",
    68: "window", 69: "desk",
    70: "toilet", 71: "door", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell",
    78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 83: "blender", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy", 89: "hair",
    90: "toothbrush", 91: "hair", 92: "banner"
}


def shuffle_dict(x: OrderedDict):
    # list(shuffle_dict(x).keys())
    keys = list(x.keys())
    random.shuffle(keys)
    x = {k: x[k] for k in keys}
    return x


def split_data_category(dataname='CocoDataset', split=(20, 20, 20, 20), order='pingyin',
                        catofset='train|val|fine', trainpart='cur-only', valpart='prev-only|cur-only|prev-cur'):
    # 按任意数目拆分COCO类别为多组
    if dataname == 'CocoDataset':
        CATS_IDS = COCO_CATS_IDS
    else:
        raise NotImplementedError(f'错误的数据集: {dataname}')
    if order == 'shuffle':
        CATS_IDS_ORDERED = shuffle_dict(CATS_IDS)
    elif order == 'pingyin':
        CATS_IDS_ORDERED = copy.copy(CATS_IDS)
    else:
        raise ValueError('不支持的类别排序')
    if isinstance(split, str):
        split = [int(s) for s in split.split('-')]
    assert isinstance(split, (tuple, list))

    print(f'\n执行数据划分 => {dataname} => 共有{len(split)}个子任务, '
          f'各任务类别数：{split} ==> 当前划分：{catofset}')
    coco_keys = list(CATS_IDS_ORDERED.keys())
    coco_vals = list(CATS_IDS_ORDERED.values())
    start, trainsplit, valsplit, finesplit = 0, [], [], []

    for idx, spt in enumerate(split):
        catname = coco_keys[start: start + spt]
        catindex = coco_vals[start: start + spt]
        trainsplit.append(dict({k: v for k, v in zip(catname, catindex)}))
        start = start + spt
    tmpdict = dict()
    for idx, spt in enumerate(trainsplit):
        assert valpart in ['prev-only', 'cur-only', 'prev-cur'], f'错误的验证模式设定:{valpart}'
        if valpart == 'prev-only':
            tmpdict = trainsplit[idx - 1] if idx >= 1 else {}
        elif valpart == 'cur-only':
            tmpdict = spt
        elif valpart == 'prev-cur':
            tmpdict.update(spt)
        valsplit.append(copy.copy(tmpdict))
    tmpdict = dict()
    for idx, spt in enumerate(trainsplit):
        tmpdict.update(spt)
        finesplit.append(copy.copy(tmpdict))
    print(f'trainsplit ==> {trainsplit}')
    print(f'valsplit   ==> {valsplit}')
    print(f'finesplit  ==> {finesplit}')
    if catofset == 'train':
        return trainsplit
    elif catofset == 'val':
        return valsplit
    elif catofset == 'fine':
        return finesplit
    elif catofset == 'train|val':
        return trainsplit, valsplit
    else:
        return trainsplit, valsplit, finesplit


if __name__ == '__main__':
    # seed = 100
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # split = list(np.ones(80).astype(dtype=int))
    split = (2, 3, 1)
    # split = 'cocosplit20'
    split_data_category(split=split, dataname='COCO2017', order='pingyin',
                        valpart='prev-all', catofset='train|val|fine')
