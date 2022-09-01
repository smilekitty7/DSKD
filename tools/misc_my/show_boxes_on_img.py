# -*- coding: utf-8 -*-  
'''
@author: zhjp   2021/12/6 下午10:52
@file: show_boxes_on_img.py.py
'''

import cv2
import numpy as np


def show_boxes_on_img(imgpath, imgwhc, bboxes=None, blabels=None,
                      gtboxes=None, gtlabels=None, match=None,
                      class_names=None, show_label=True, waitkey=-1):
    old_image = cv2.imread(imgpath)
    new_image = cv2.resize(old_image, dsize=imgwhc[:2])
    # cv2.imshow('old image', old_image)

    if blabels is None:
        blabels = [0] * len(bboxes)
    else:
        assert len(bboxes) == len(blabels), \
            f'盒子BBox({len(bboxes)})与BLable({len(blabels)})数量不相等'
    if gtlabels is None:
        gtlabels = [0] * len(gtboxes)
    else:
        assert len(gtboxes) == len(gtlabels), \
            f'盒子GtBox({len(gtboxes)})与GtLable({len(gtlabels)})数量不相等'

    font = cv2.FONT_ITALIC

    for i, (bbox, label) in enumerate(zip(bboxes, blabels)):
        x1, y1, x2, y2 = bbox.astype(np.int32)
        label_text = class_names[label] if class_names is not None else f'class {label}'
        cv2.rectangle(new_image, (x1, y1), (x2, y2), (200, 50, 20), 1)
        if show_label:
            cv2.rectangle(new_image, (x1, y1), (x1 + len(label_text) * 7, int(y1 - 14)), (170, 50, 10), cv2.FILLED)
            cv2.putText(new_image, label_text, (x1, int(y1 - 5)), font, 0.4, (255, 255, 255), 1)

    for i, (bbox, label) in enumerate(zip(gtboxes, gtlabels)):
        x1, y1, x2, y2 = bbox.astype(np.int32)
        label_text = class_names[label] if class_names is not None else f'class {label}'
        cv2.rectangle(new_image, (x1, y1), (x2, y2), (20, 50, 200), 1)
        if show_label:
            cv2.rectangle(new_image, (x1, y1), (x1 + len(label_text) * 7, int(y1 - 14)), (170, 50, 10), cv2.FILLED)
            cv2.putText(new_image, label_text, (x1, int(y1 - 5)), font, 0.4, (255, 255, 255), 1)

    cv2.imshow('new image', new_image)
    cv2.waitKey(waitkey)
