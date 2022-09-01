# Copyright (c) OpenMMLab. All rights reserved.
import copy
import mmcv

from .single_stage import SingleStageDetector
from mmdet.core import bbox2result
from ..builder import DETECTORS
from .base import BaseDetector
import warnings
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16
from .. import build_detector
from ...datasets.data_split import COCO_LABEL_MAP

# for increment learning
import cv2
import numpy as np
from torch import Tensor
from typing import Any
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw, ImageFont
from ..builder import build_backbone, build_head, build_neck
from collections import OrderedDict


@DETECTORS.register_module()
class YOLOF(SingleStageDetector):
    r"""Implementation of `You Only Look One-level Feature
    <https://arxiv.org/abs/2103.09460>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config=None,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 teacher_test_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOF, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.fp16_enabled = False
        self.has_teacher = teacher_config and teacher_ckpt
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
            self.bbox_head.has_teacher = False
            print(f'教师模型未设置')
            return None
        if model is not None:
            self.teacher_model = model
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
            self.teacher_model.has_teacher = False
            self.teacher_model.bbox_head.has_teacher = False
        self.has_teacher = True
        self.bbox_head.has_teacher = True
        print(f'教师模型已设置，TrainVal：{trainval}，权值加载：{ckptfile if not model else "byModel"}')
        return self.teacher_model

    def out_teacher(self, img, img_metas, cat_keepid=True):
        assert self.has_teacher, '当前没有教师模型'
        with torch.no_grad():
            # neck_feat=([b, c, h5, w5], [b, c, h4, w4], ...)
            # head_outs=([[b, Prior*Class, h5, w5], [b, Prior*4, h5, w5], ...], [[],[], ...]...)
            # pred_outs=[((ObjNums, 5), (ObjNums, ), (ObjNums, CatNums))b1, (...)b2, ...]
            # pred_outs = self.teacher_model(return_loss=False, rescale=True, **img)
            neck_feat = self.teacher_model.extract_feat(img)
            head_outs = self.teacher_model.bbox_head.forward(neck_feat)
            pred_outs = self.teacher_model.bbox_head.get_bboxes(
                *head_outs, img_metas=img_metas,
                cfg=getattr(self, 'teacher_test_cfg', self.test_cfg),
                rescale=False, with_nms=True, need_logits=True)
            # for feat in neck_feat:
            #     feat.detach_()
            # for out in head_outs:
            #     for ot in out:
            #         ot.detach_()
            # for i in range(len(img_metas)):
            #     pred_outs[i][0].detach_()
            #     pred_outs[i][1].detach_()
            pred_bboxes = [result[0][:, 0:4].detach() for result in pred_outs]
            pred_scores = [result[0][:, 4:5].flatten().detach() for result in pred_outs]
            pred_labels = [result[1].detach() for result in pred_outs]
            assert not any(torch.cat(pred_labels)==80), print(torch.cat(pred_labels))
            if len(pred_outs[0]) == 4:  # if has logits
                pred_logits = [result[2].detach() for result in pred_outs]
                pred_keepid = [result[3].detach() for result in pred_outs]
            else:
                pred_logits, pred_keepid = None, None
            if cat_keepid:
                # print('pred_keepid 1=>', [(len(pk.tolist())==len(set(pk.tolist())), len(pk.tolist()), len(set(pk.tolist()))) for pk in teacher_info['pred_keepid']])
                # total_boxs = head_outs[0][0].numel()//self.bbox_head.num_classes
                batch_size = head_outs[0][0].numel()//self.bbox_head.num_classes//len(pred_outs)
                pred_keepid = torch.cat([pk + i * batch_size for i, pk in enumerate(pred_keepid)])
                # print('pred_keepid 2=>', len(pred_keepid.tolist())==len(set(pred_keepid.tolist())), len(pred_keepid.tolist()), len(set(pred_keepid.tolist())))
                # print(f'len(pred_keepid) = {len(pred_keepid)}, {pred_keepid} \t')
        return neck_feat, head_outs, pred_keepid, pred_logits, pred_labels, pred_scores, pred_bboxes

    def set_student(self, ckptfile=None):
        if ckptfile is not None:
            print(f'学生模型权值加载：{ckptfile}')
            mmcv.runner.load_checkpoint(self, ckptfile, map_location='cpu')
        # Frozen
        # Loss
        return self

    def load_student(self, ckptfile):
        mmcv.runner.load_checkpoint(self, ckptfile, map_location='cpu')
        # delete prev teacher of student
        if self.teacher_model is not None:
            self.teacher_model = None
            self.has_teacher = False
        return None

    def set_datainfo(self, cat2id: dict, cat2label: dict):
        # cat2id: {CatName: CatID, ...} cat2label: {CatID: Label, ...}
        catid2catname = {v: k for k, v in cat2id.items()}
        self.Label2CatNameId = {v: [catid2catname[k], k] for k, v in cat2label.items()}

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

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
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

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
        super(SingleStageDetector, self).forward_train(img, img_metas)

        teacher_feats, teacher_outs, teacher_keepid, teacher_logits, \
        teacher_labels, teacher_scores, teacher_bboxes = [None] * 7
        if self.has_teacher:
            teacher_feats, teacher_outs, teacher_keepid, teacher_logits, teacher_labels, \
            teacher_scores, teacher_bboxes = self.out_teacher(img, img_metas, cat_keepid=True)

        ###########################
        # for batch_idx, img_meta in enumerate(img_metas):
        #     # target = teacher_result[batch_idx]
        #     target = {'labels': teacher_labels[batch_idx],
        #               'scores': teacher_scores[batch_idx],
        #               'boxes': teacher_bboxes[batch_idx]}
        #     self.draw_boxes_on_img_v1(
        #         img_info=img_meta, target=target, target_style='style1',
        #         coord='x1y1x2y2', isnorm=False, imgsize='new',
        #         waitKey=-200, window='imgshow', realtodo=1)
            # print(target)
        #############################


        teacher_feats = teacher_feats if self.bbox_head.feats_distill else None
        teacher_outs = teacher_outs if 'soft' in self.bbox_head.cates_distill else None
        teacher_keepid = teacher_keepid if 'soft' in self.bbox_head.cates_distill else None
        teacher_logits = teacher_logits or None

        teacher_info = {
            'neck_feats':  teacher_feats,
            'head_outs':   teacher_outs,
            'pred_keepid': teacher_keepid,
            'pred_logits': teacher_logits,
            'pred_scores': teacher_scores,
            'pred_labels': teacher_labels,
            'pred_bboxes': teacher_bboxes,
        }

        x = self.extract_feat(img)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore,
                                              proposal_cfg=None,
                                              teacher_info=teacher_info)
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.
        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def train_step(self, data, optimizer):
        """The iteration step during training.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
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

    def draw_boxes_on_img_v2(self, img_id=None, img_mat=None, img_path=None, target=Any,
                             isxywh=True, isnorm=False, waitKey=200, window='imgshow', realtodo=1):
        if not realtodo: return

        print(f'\n加载 Image........')
        if img_id:
            image = self._load_image(img_id)
            target = self._load_target(img_id)
            image_path = self._image_path(img_id)
        elif (isinstance(img_mat, Tensor) or img_mat) and target:
            if isinstance(img_mat, Tensor):
                img_mat = ToPILImage()(img_mat)
            image = img_mat
        elif img_path and target:
            image = Image.open(img_path)
        else:
            raise ValueError('无法加载图片')
        image = image.convert('RGB')
        w_now, h_now = image.size
        print('image.size: [w_now, h_now]', image.size)
        # image.show()

        print(f'加载 Target........')
        if img_id is not None:
            target = [(t['category_id'], t['bbox']) for t in target]
        elif isinstance(target, list):
            # [{ann1}, {ann2}, ...]
            target = [(t['category_id'], t['bbox']) for t in target]
        elif isinstance(target, dict):
            # image_id, orig_size = target['image_id'], target['orig_size']
            labels, boxes = list(target['labels'].numpy()), list(target['boxes'].numpy().tolist())
            h_org, w_org = target['orig_size'].numpy()
            h_new, w_new = target['size'].numpy()
            print(f'h_org, w_org: {h_org}, {w_org}',
                  f'h_new, w_new = > {h_new}, {w_new}',
                  f'h_now, w_now = > {h_now}, {w_now}', )
            if isnorm:
                boxes = [[box[0] * w_new, box[1] * h_new, box[2] * w_new, box[3] * h_new] for box in boxes]
            target = list(zip(labels, boxes))

        print(f'绘制 BBOX........')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf", 18)
        for idx, (label, bbox) in enumerate(target):
            if isxywh:
                x1, y1, w, h = (int(v) for v in bbox)
                x_min, y_min, x_max, y_max = x1, y1, x1 + w, y1 + h
            else:
                x_min, y_min, x_max, y_max = (int(v) for v in bbox)
            draw.line([(x_min, y_min), (x_min, y_max), (x_max, y_max),
                       (x_max, y_min), (x_min, y_min)], width=1, fill=(0, 0, 255))
            text = COCO_LABEL_MAP[label] + f'|{label}'
            draw.text((x_min, y_min), text, (255, 255, 0), font=font)
        # image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow(f'{window}', image)
        print(f'绘制完成........')
        cv2.waitKey(waitKey)
        return
