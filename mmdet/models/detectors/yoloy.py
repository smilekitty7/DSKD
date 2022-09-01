# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
from mmdet.core import bbox2result

# for increment learning
import cv2, copy, mmcv
import numpy as np
from torch import Tensor
from typing import Any
from collections import OrderedDict
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw, ImageFont
from ..builder import build_backbone, build_head, build_neck
from .. import build_detector


@DETECTORS.register_module()
class YOLOY(SingleStageDetector):

    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config=None,
                 teacher_ckpt=None,
                 teacher_test_cfg=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLOY, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        self.has_teacher = teacher_config and teacher_ckpt
        self.fp16_enabled = False
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        bbox_head.update(has_teacher=self.has_teacher)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.teacher_test_cfg = teacher_test_cfg

        self.Label2CatNameId = dict()  # {Label: [CatID, CatName], ...}

        # Build teacher model from config file
        if self.has_teacher:
            self.eval_teacher = eval_teacher
            self.teacher_model = self.set_teacher(config=teacher_config, ckptfile=teacher_ckpt, trainval='val')
            print(f'教师模型加载成功，{teacher_ckpt}')
        else:
            self.teacher_model = None
            print(f'教师模型未设置，teacher_config={teacher_config}，teacher_ckpt={teacher_ckpt}')

    def set_teacher(self, config=None, ckptfile=None, model=None, trainval='val'):
        # Build teacher model by student API
        if (config is None or ckptfile is None) and model is None:
            self.has_teacher = False
            print(f'教师模型未设置')
            return None
        if model is not None:
            self.teacher_model = copy.deepcopy(model)
        elif config and ckptfile:
            if isinstance(config, str):
                config = mmcv.Config.fromfile(config)
            self.teacher_model = build_detector(config['model'])
            if ckptfile:
                mmcv.runner.load_checkpoint(self.teacher_model, ckptfile, map_location='cpu')
        else:
            raise NotImplementedError('教师模型wufa设置')
        if trainval == 'val':
            self.eval_teacher = True
            # self.teacher_model.eval()
            self.teacher_model.train(False)
            for name, param in self.teacher_model.named_parameters():
                param.requires_grad = False
        else:
            self.eval_teacher = False
            self.teacher_model.train(True)
        # del teacher of teacher
        if getattr(self.teacher_model, 'teacher_model', None) is not None:
            setattr(self.teacher_model, 'teacher_model', None)
        if getattr(self.teacher_model, 'has_teacher', False):
            setattr(self.teacher_model, 'has_teacher', False)
        self.has_teacher = True
        print(f'教师模型已设置，TrainVal：{trainval}，权值加载：{ckptfile if not model else "byModel"}')
        return self.teacher_model

    def out_teacher(self, img, img_metas, rescale=False):
        assert self.has_teacher, '当前没有教师模型'
        with torch.no_grad():
            # teacher_feat=([b, c, h5, w5], [b, c, h4, w4], ...)，
            # teacher_out=([[b, Prior*Class, h5, w5], [b, Prior*4, h5, w5], ...], [[],[], ...]...)
            # teacher_result=[((ObjNums, 5), (ObjNums, ))b1, ((ObjNums, 5), (ObjNums, ))b2, ...]
            # teacher_result = self.teacher_model(return_loss=False, rescale=True, **img)
            teacher_feat = self.teacher_model.extract_feat(img)
            teacher_out = self.teacher_model.bbox_head.forward(teacher_feat)
            teacher_result = self.teacher_model.bbox_head.get_bboxes(
                *teacher_out, img_metas=img_metas, rescale=False,
                cfg=getattr(self, 'teacher_test_cfg', self.test_cfg)
            )
            # for feat in teacher_feat:
            #     feat.detach_()
            # for out in teacher_out:
            #     for ot in out:
            #         ot.detach_()
            # for i in range(len(img_metas)):
            #     teacher_result[i][0].detach_()
            #     teacher_result[i][1].detach_()
            teacher_bboxes = [result[0][:, 0:4].detach() for result in teacher_result]
            teacher_scores = [result[0][:, 4:5].flatten().detach() for result in teacher_result]
            teacher_labels = [result[1].detach() for result in teacher_result]
            # print(teacher_result)
        return teacher_feat, teacher_out, teacher_result, teacher_labels, teacher_bboxes, teacher_scores

    def set_student(self, ):
        # Frozen
        # Loss
        return self

    def set_datainfo(self, cat2id: dict, cat2label: dict):
        # cat2id: {CatName: CatID, ...} cat2label: {CatID: Label, ...}
        catid2catname = {v: k for k, v in cat2id.items()}
        self.Label2CatNameId = {v: [catid2catname[k], k] for k, v in cat2label.items()}

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        forked from SingleStageDetector.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)

        teacher_feat, teacher_out, teacher_result = None, None, None
        if self.has_teacher:
            teacher_feat, teacher_out, teacher_result, \
            teacher_labels, teacher_bboxes, teacher_scores = self.out_teacher(img, img_metas, rescale=False)
            gt_labels = [torch.cat([teacher_labels[i], gt_labels[i]], dim=0) for i in range(len(img_metas))]
            gt_bboxes = [torch.cat([teacher_bboxes[i], gt_bboxes[i]], dim=0) for i in range(len(img_metas))]

        # for batch_idx, img_meta in enumerate(img_metas):
        #     # target = teacher_result[batch_idx]
        #     target = {'labels': teacher_labels[batch_idx],
        #               'scores': teacher_scores[batch_idx],
        #               'boxes': teacher_bboxes[batch_idx]}
        #     self.draw_boxes_on_img_v1(#img_mat=img[batch_idx],
        #                               img_info=img_meta,
        #                               target=target, target_style='style1',
        #                               coord='x1y1x2y2', isnorm=False, imgsize='new',
        #                               waitKey=-200, window='imgshow', realtodo=1)
        #     # print(target)

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore,
                                              proposal_cfg=None,
                                              teacher_feat=teacher_feat,
                                              teacher_out=teacher_out)
        return losses

    def Xforward_test(self, imgs, img_metas, **kwargs):
        super(YOLOY, self).forward_test(imgs, img_metas, **kwargs)

    def Xsimple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.
        """
        # super(YOLOY, self).simple_test(img, img_metas, rescale=rescale)
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    @auto_fp16(apply_to=('img', ))
    def Xforward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        if self.has_teacher:
            # print('设置教师模型device==>', device)
            self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.has_teacher:
            # print(f'设置教师模型训练验证状态 Eval: {self.eval_teacher}')
            if self.eval_teacher:
                self.teacher_model.train(False)
            else:
                self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model' and self.has_teacher:
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def draw_boxes_on_img_v1(self, img_id=None, img_mat=None, img_info=None,
                             target=Any, target_style='style1',
                             coord='x1y1wh', isnorm=False, imgsize='orig|new',
                             waitKey=200, window='imgshow', realtodo=1):
        # imgsize: 使用原图尺寸或转换后尺寸画图,跟模型中rescale参数协同设定。
        if not realtodo: return
        assert coord in 'x1y1wh|cxcywh|x1y1x2y2'
        img_flip = False
        h_org, w_org, h_new, w_new, w_now, h_now = 0, 0, 0, 0, 0, 0

        print(f'\n加载 Image........')
        if img_id:
            image = self.coco.load_imgs(ids=[img_id])
            target = self.coco.load_anns(ids=[img_id])
        elif img_mat is not None:
            if isinstance(img_mat, Tensor):
                img_mat = ToPILImage()(img_mat)
            image = img_mat
        elif img_info:
            if isinstance(img_info, dict) and 'filename' in img_info:
                # print('img_info=>', img_info)
                img_path = img_info.get('filename', 'error filename')
                img_flip = img_info.get('flip', False)
                h_org, w_org = img_info.get('orig_size', img_info.get('ori_shape', [None] * 3)[:2])
                h_new, w_new = img_info.get('size', img_info.get('img_shape', [None] * 3)[:2])
            else:
                img_path = img_info
            image = Image.open(img_path)
        else:
            raise ValueError('无法加载图片')
        image = image.convert('RGB')
        if imgsize == 'new':
            image = image.resize((w_new, h_new), Image.ANTIALIAS)
        w_now, h_now = image.size
        print(f'图像尺寸信息: [h_org, w_org], [h_new, w_new], [h_now, w_now]'
              f'= {h_org, w_org, h_new, w_new, h_now, w_now}')
        # image.show()

        print(f'加载 Target........')
        if target_style == 'style1':
            # boxes, labels, scores 按找字典传入
            boxes = target['boxes']
            labels = target.get('labels', [0] * len(boxes))
            scores = target.get('scores', [0] * len(boxes))
            boxes = boxes if not isinstance(boxes, Tensor) else boxes.cpu().numpy().tolist()
            labels = labels if not isinstance(labels, Tensor) else labels.cpu().numpy().tolist()
            scores = scores if not isinstance(scores, Tensor) else scores.cpu().numpy().tolist()
            assert len(labels) == len(boxes) and len(scores) == len(boxes)
            target = list(zip(labels, scores, boxes))
        elif target_style == 'mmpred':
            # 直接传入mmde中 target=model.get_bboxes() 的预测输出: [(x1, y1, x2, y2, score), ...][label, ...]
            # [(x1, y1, x2, y2, score, label), ...]=>[((x1, y1, x2, y2), score, label), ...]
            target = [torch.cat([t[0], t[1].unsqueeze(1)], dim=1) for t in target]
            if isinstance(target[0], Tensor):
                target = [t.cpu().numpy().tolist() for t in target]
            target = [[t[5], t[4], t[:4]] for t in target]
        else:
            raise NotImplementedError(f'错误的Taget存放方式, target_style={target_style}')

        print(f'绘制 BBOX........')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf", 18)
        for idx, (label, score, bbox) in enumerate(target):
            # print(label, score, bbox)
            if coord == 'x1y1wh':
                x1, y1, w, h = (int(v) for v in bbox)
                x_min, y_min, x_max, y_max = x1, y1, x1 + w, y1 + h
            elif coord == 'cxcywh':
                cx, cy, w, h = (int(v) for v in bbox)
                x_min, y_min, x_max, y_max = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            elif coord == 'x1y1x2y2':
                x_min, y_min, x_max, y_max = (int(v) for v in bbox)
            else:
                raise NotImplementedError(f'参数错误：coord={coord}')
            if img_flip:
                x_min, y_min, x_max, y_max = w_now - x_max, y_min, w_now - x_min, y_max
            draw.line([(x_min, y_min), (x_min, y_max), (x_max, y_max),
                       (x_max, y_min), (x_min, y_min)], width=1, fill=(0, 0, 255))
            # CategoryName, CategoryID, CategoryLabel
            text = self.Label2CatNameId[label][0] + ['', '|' + str(score)[:5]][score > 0]
            draw.text((x_min, y_min), text, (255, 255, 0), font=font)
        # image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow(f'{window}', image)
        print(f'绘制完成........')
        cv2.waitKey(waitKey)