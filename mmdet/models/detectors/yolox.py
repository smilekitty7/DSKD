# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from ...utils import log_img_scale
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

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
class YOLOX(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Note: Considering the trade-off between training speed and accuracy,
    multi-scale training is temporarily kept. More elegant implementation
    will be adopted in the future.

    Args:
        backbone (nn.Module): The backbone module.
        neck (nn.Module): The neck module.
        bbox_head (nn.Module): The bbox head module.
        train_cfg (obj:`ConfigDict`, optional): The training config
            of YOLOX. Default: None.
        test_cfg (obj:`ConfigDict`, optional): The testing config
            of YOLOX. Default: None.
        pretrained (str, optional): model pretrained path.
            Default: None.
        input_size (tuple): The model default input image size. The shape
            order should be (height, width). Default: (640, 640).
        size_multiplier (int): Image size multiplication factor.
            Default: 32.
        random_size_range (tuple): The multi-scale random range during
            multi-scale training. The real training image size will
            be multiplied by size_multiplier. Default: (15, 25).
        random_size_interval (int): The iter interval of change
            image size. Default: 10.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config=None,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None):
        super(YOLOX, self).__init__(backbone, neck, bbox_head, train_cfg,
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

        log_img_scale(input_size, skip_square=True)
        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self._input_size = input_size
        self._random_size_range = random_size_range
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0

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

    def out_teacher(self, img, img_metas):
        assert self.has_teacher, '当前没有教师模型'
        with torch.no_grad():
            # teacher_feat=([b, c, h5, w5], [b, c, h4, w4], ...)，
            # teacher_out=([[b, Prior*Class, h5, w5], [b, Prior*4, h5, w5], ...], [[],[], ...]...)
            # teacher_result=[((ObjNums, 5), (ObjNums, ))b1, ((ObjNums, 5), (ObjNums, ))b2, ...]
            teacher_feat = self.teacher_model.extract_feat(img)
            teacher_out = self.teacher_model.bbox_head.forward(teacher_feat)
            teacher_result = self.teacher_model.bbox_head.get_bboxes(
                *teacher_out, img_metas=img_metas, rescale=False,
                cfg=self.test_cfg if True else 'cfg'
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
            teacher_scores = [result[0][:, 4:5].squeeze().detach() for result in teacher_result]
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
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Multi-scale training
        img, gt_bboxes = self._preprocess(img, gt_bboxes)
        super(SingleStageDetector, self).forward_train(img, img_metas)

        teacher_feat, teacher_out, teacher_result = None, None, None
        if self.has_teacher:
            teacher_feat, teacher_out, teacher_result, \
            teacher_labels, teacher_bboxes, teacher_scores = self.out_teacher(img, img_metas)
            # gt_labels = [torch.cat([teacher_labels[i], gt_labels[i]], dim=0) for i in range(len(img_metas))]
            # gt_bboxes = [torch.cat([teacher_bboxes[i], gt_bboxes[i]], dim=0) for i in range(len(img_metas))]

        for batch_idx, img_meta in enumerate(img_metas):
            # target = teacher_result[batch_idx]
            target = {'labels': gt_labels[batch_idx],
                      # 'scores': teacher_scores[batch_idx],
                      'boxes': gt_bboxes[batch_idx]}
            self.draw_boxes_on_img_v1(img_mat=img[batch_idx],
                img_info=img_meta, target=target, target_style='style1',
                coord='x1y1x2y2', isnorm=False, waitKey=-200, window='imgshow', realtodo=1)
            # print(target)

        x = self.extract_feat(img)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore,
                                              proposal_cfg=None,
                                              teacher_feat=teacher_feat,
                                              teacher_out=teacher_out)

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize()
        self._progress_in_iter += 1

        return losses

    def _preprocess(self, img, gt_bboxes):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(
                img,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes

    def _random_resize(self):
        tensor = torch.LongTensor(2).cuda()

        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            aspect_ratio = float(
                self._default_input_size[1]) / self._default_input_size[0]
            size = (self._size_multiplier * size,
                    self._size_multiplier * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if self.world_size > 1:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

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

    def draw_boxes_on_img_v1(self, img_id=None, img_mat=None, img_info=None, target=Any, target_style='style1',
                             coord='x1y1wh', isnorm=False, waitKey=200, window='imgshow', realtodo=1):
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
        elif target_style == 'style2':
            # mmde中 target=get_bboxes() 的预测输出: [(x1, y1, x2, y2, score), ...][label, ...]
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