# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmdet.core.utils import filter_scores_and_topk
from mmcv.ops import batched_nms

INF = 1e8

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import HEADS, build_loss
from .detr_head import DETRHead
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean,anchor_inside_flags, bbox_overlaps,
                        images_to_levels, unmap,build_bbox_coder)



class Integral_average(nn.Module): # v2
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral_average, self).__init__()
        self.reg_max = reg_max
        # self.register_buffer('project',
        #                      torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x): # x: left, right, top, bottom
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = x.reshape(-1, self.reg_max + 1)#.softmax(1)
        x = x / x.sum(1).unsqueeze(1).repeat(1,self.reg_max+1)
        space = torch.linspace(0, self.reg_max,self.reg_max+1).to(x.device)
        space = space / self.reg_max / 2
        x = x * space
        x = x.sum(1).reshape(-1, 2,2).sum(2)
        return x




@HEADS.register_module()
class GFLDeformableDETRHead_il(DETRHead):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 reg_max = 16,
                 temp = 0.5,
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 # bbox_coder=dict(type='DistancePointBBoxCoder'),
                 transformer=None,
                 cates_distill='',
                 locat_distill='',
                 feats_distill='',
                 loss_kd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 loss_ld_bbox=dict(type='SmoothL1Loss', loss_weight=10, reduction='mean'),
                 loss_ld_logit=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_fd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 loss_memory=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=1,
                     T=2),
                 loss_fg_feature = dict(type='MSELoss', loss_weight=1, reduction='sum'),
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.reg_max = reg_max
        self.temp = temp
        self.has_teacher = kwargs.pop('has_teacher', False)
        super(GFLDeformableDETRHead_il, self).__init__(
            *args, transformer=transformer, **kwargs)
        # self.integral = Integral(self.reg_max)
        self.integral_average = Integral_average(self.reg_max)
        self.loss_dfl = build_loss(loss_dfl)
        # self.bbox_coder = build_bbox_coder(bbox_coder)
        self.cates_distill = cates_distill
        self.locat_distill = locat_distill
        self.feats_distill = feats_distill
        self.loss_kd = build_loss(loss_kd) if cates_distill else None
        self.loss_ld_bbox = build_loss(loss_ld_bbox) if 'bbox' in locat_distill else None
        self.loss_ld_logit = build_loss(loss_ld_logit) if 'logit' in locat_distill else None #and self.reg_val['usedfl'] else None
        self.loss_fd = build_loss(loss_fd) if 'kldv' in feats_distill else None
        self.loss_memory = build_loss(loss_memory) if 'memory' in feats_distill else None
        self.loss_fg_feature = build_loss(loss_fg_feature) if 'fg_info' in feats_distill else None

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 2+4*(self.reg_max + 1)))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:

            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            memory, enc_outputs_class, enc_outputs_coord = self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                enc_outputs_class, \
                enc_outputs_coord.sigmoid()
        else:
            return outputs_classes, outputs_coords, \
                memory, None

    # def forward_single(self, x, img_metas):
    #     """"Forward function for a single feature level.
    #
    #     Args:
    #         x (Tensor): Input feature from backbone's single stage, shape
    #             [bs, c, h, w].
    #         img_metas (list[dict]): List of image information.
    #
    #     Returns:
    #         all_cls_scores (Tensor): Outputs from the classification head,
    #             shape [nb_dec, bs, num_query, cls_out_channels]. Note
    #             cls_out_channels should includes background.
    #         all_bbox_preds (Tensor): Sigmoid outputs from the regression
    #             head with normalized coordinate format (cx, cy, w, h).
    #             Shape [nb_dec, bs, num_query, 4].
    #     """
    #     # construct binary masks which used for the transformer.
    #     # NOTE following the official DETR repo, non-zero values representing
    #     # ignored positions, while zero values means valid positions.
    #     batch_size = x.size(0)
    #     input_img_h, input_img_w = img_metas[0]['batch_input_shape']
    #     masks = x.new_ones((batch_size, input_img_h, input_img_w))
    #     for img_id in range(batch_size):
    #         img_h, img_w, _ = img_metas[img_id]['img_shape']
    #         masks[img_id, :img_h, :img_w] = 0
    #
    #     x = self.input_proj(x)
    #     # interpolate masks to have the same spatial shape with x
    #     masks = F.interpolate(
    #         masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
    #     # position encoding
    #     pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
    #     # outs_dec: [nb_dec, bs, num_query, embed_dim]
    #     outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
    #                                    pos_embed)
    #
    #     all_cls_scores = self.fc_cls(outs_dec)
    #     all_bbox_preds = self.fc_reg(self.activate(
    #         self.reg_ffn(outs_dec))).sigmoid()
    #     return all_cls_scores, all_bbox_preds

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        teacher_info = kwargs.pop('teacher_info', {})
        student_feat = x if self.has_teacher and self.feats_distill else []
        # assert proposal_cfg is None, '"proposal_cfg" must be None'

        outs = self.forward(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,
                           student_feat=student_feat, teacher_info=teacher_info)
        ################################################################
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        #################################################################

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             info_all,
             enc_bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None,
             student_feat = [],
             teacher_info = {}):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            info_all (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        ########################################################################
        # 合并 GT-Label-Boxes & Teacher-Label-Boxes
        if self.has_teacher and 'hard' in self.cates_distill:
            for i in range(len(img_metas)):
                gt_labels_list[i] = torch.cat([teacher_info['pred_labels'][i], gt_labels_list[i]], dim=0)
                gt_bboxes_list[i] = torch.cat([teacher_info['pred_bboxes'][i], gt_bboxes_list[i]], dim=0)

        #######################################################################

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, losses_dfl = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_dfl'] = losses_dfl[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_dfl_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1],
                                                       losses_dfl[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_dfl'] = loss_dfl_i
            num_dec_layer += 1


        if self.has_teacher:
            # TODO 背景类别在pred_keepid中被去除 ？ 只有高置信度正样本做Loss！
            if 'soft' in self.cates_distill:
                soft_label = teacher_info['head_outs'][0][-1].reshape(-1, self.cls_out_channels) # [600,80]
                soft_weight = soft_label.new_zeros(size=(soft_label.shape[0], 1))
                soft_weight[teacher_info['pred_keepid']] = 1
                loss_kd = self.loss_kd(all_cls_scores[-1].reshape(-1,self.cls_out_channels), soft_label, weight=soft_weight,
                                       avg_factor=len(teacher_info['pred_keepid']))
                loss_dict.update({'loss_kd': loss_kd})
            if 'bbox' in self.locat_distill:
                batch_pred_bbox = all_bbox_preds[-1]#.reshape(-1, 4*self.reg_val['num'])
                batch_soft_bbox = teacher_info['head_outs'][1][-1]#.reshape(-1, 4*self.reg_val['num'])
                wh_pred_bbox = self.integral_average(batch_pred_bbox[:,:,2:])
                wh_soft_bbox = self.integral_average(batch_soft_bbox[:,:,2:])
                soft_weight = wh_soft_bbox.new_zeros(size=(wh_soft_bbox.shape[0], 1))
                soft_weight[teacher_info['pred_keepid']] = 1
                cxcywh_pred_bbox = torch.cat((batch_pred_bbox[:,:,:2].reshape(-1,2),wh_pred_bbox),dim=1)
                cxcywh_soft_bbox = torch.cat((batch_soft_bbox[:,:,:2].reshape(-1,2),wh_soft_bbox),dim=1)
                loss_ld_bbox = self.loss_ld_bbox(cxcywh_pred_bbox, cxcywh_soft_bbox, weight=soft_weight,
                                                 avg_factor=len(teacher_info['pred_keepid']))
                loss_dict.update({'loss_ld_bbox': loss_ld_bbox})
            if 'logit' in self.locat_distill:
                # forked from ld_head.py Line99-Line122
                batch_pred_bbox = all_bbox_preds[-1].reshape(-1, 4* (self.reg_max+1)+2)
                batch_soft_bbox = teacher_info['head_outs'][1][-1].reshape(-1, 4*(self.reg_max+1)+2)
                soft_weight = batch_soft_bbox.new_zeros(size=(batch_soft_bbox.shape[0], 1))
                soft_weight[teacher_info['pred_keepid']] = 1
                loss_ld_logit = self.loss_ld_logit(batch_pred_bbox, batch_soft_bbox, weight=soft_weight,
                                                   avg_factor=len(teacher_info['pred_keepid']))
                loss_dict.update({'loss_ld_logit': loss_ld_logit})
            if 'kldv' in self.feats_distill:
                # assert len(student_feat) == len(teacher_info['neck_feats'])
                loss_fd = [self.loss_fd(sf, tf, weight=None, avg_factor=None)
                           for sf, tf in zip(student_feat, teacher_info['neck_feats'])]
                avg_factor = [1, len(loss_fd), len(img_metas), len(teacher_info['pred_keepid'])][2]
                loss_fd = sum(loss_fd)/avg_factor
                loss_dict.update({'loss_fd': loss_fd})
            if 'memory' in self.feats_distill:
                memory,_ = info_all
                batch_pred_memory = memory.permute(1,2,0)
                batch_soft_memory = teacher_info['head_outs'][2][0].permute(1,2,0)
                loss_memory = [self.loss_memory(s_memory, t_memory, weight=None, avg_factor=None)
                           for s_memory, t_memory in zip(batch_pred_memory, batch_soft_memory)]
                avg_factor = [1, len(loss_memory), len(img_metas), len(teacher_info['pred_keepid']), memory.shape[2]*len(img_metas)][2]
                loss_memory = sum(loss_memory)/avg_factor
                loss_dict.update({'loss_memory': loss_memory})
            if 'fg_info' in self.feats_distill:
                memory, spatial_shapes = info_all
                batch_pred_memory = memory.permute(1, 2, 0)
                batch_soft_memory = teacher_info['head_outs'][2][0].permute(1,2,0)
                N, C, HWall = batch_pred_memory.shape
                # S_attention_t = self.get_attention(batch_pred_memory, spatial_shapes, self.temp)
                # Mask_fg = torch.zeros_like(S_attention_t)
                # Mask_bg = torch.ones_like(S_attention_t)
                Mask_fg = [torch.zeros((N, hw[0],hw[1])).to(batch_pred_memory.device) for hw in spatial_shapes]
                Mask_bg = [torch.ones((N, hw[0],hw[1])).to(batch_pred_memory.device) for hw in spatial_shapes]
                wmin, wmax, hmin, hmax = [], [], [], []

                sp_shape = spatial_shapes.shape[0]
                for i in range(N):
                    for sp in range(sp_shape):
                        new_boxxes = torch.ones_like(teacher_info['pred_bboxes'][i])
                        new_boxxes[:, 0] = teacher_info['pred_bboxes'][i][:, 0] / img_metas[i]['img_shape'][1] * spatial_shapes[sp][0]
                        new_boxxes[:, 2] = teacher_info['pred_bboxes'][i][:, 2] / img_metas[i]['img_shape'][1] * spatial_shapes[sp][0]
                        new_boxxes[:, 1] = teacher_info['pred_bboxes'][i][:, 1] / img_metas[i]['img_shape'][0] * spatial_shapes[sp][1]
                        new_boxxes[:, 3] = teacher_info['pred_bboxes'][i][:, 3] / img_metas[i]['img_shape'][0] * spatial_shapes[sp][1]

                        wmin.append(torch.floor(new_boxxes[:, 0]).int())
                        wmax.append(torch.ceil(new_boxxes[:, 2]).int())
                        hmin.append(torch.floor(new_boxxes[:, 1]).int())
                        hmax.append(torch.ceil(new_boxxes[:, 3]).int())

                        area = 1.0 / (hmax[i*sp_shape+sp].view(1, -1) + 1 - hmin[i*sp_shape+sp].view(1, -1)) / (
                                    wmax[i*sp_shape+sp].view(1, -1) + 1 - wmin[i*sp_shape+sp].view(1, -1))

                        for j in range(len(teacher_info['pred_bboxes'][i])):
                            Mask_fg[sp][i][hmin[i*sp_shape+sp][j]:hmax[i*sp_shape+sp][j] + 1, wmin[i*sp_shape+sp][j]:wmax[i*sp_shape+sp][j] + 1] = \
                                torch.maximum(Mask_fg[sp][i][hmin[i*sp_shape+sp][j]:hmax[i*sp_shape+sp][j] + 1, wmin[i*sp_shape+sp][j]:wmax[i*sp_shape+sp][j] + 1], area[0][j])

                        Mask_bg[sp][i] = torch.where(Mask_fg[sp][i] > 0, 0, 1)
                        if torch.sum(Mask_bg[sp][i]):
                            Mask_bg[sp][i] /= torch.sum(Mask_bg[sp][i])

                Mask_fg_all = torch.cat([Mask_fg[m].reshape(N,-1) for m in range(len(Mask_fg))],dim=1)
                Mask_bg_all = torch.cat([Mask_bg[m].reshape(N, -1) for m in range(len(Mask_bg))], dim=1)

                fg_loss = [self.fg_feature_calculation(s_memory, t_memory, m_fg, m_bg, weight=None, avg_factor=None)
                                    for s_memory, t_memory, m_fg, m_bg in zip(batch_pred_memory, batch_soft_memory,Mask_fg_all, Mask_bg_all)]
                avg_factor = [1, len(fg_loss), len(img_metas), len(teacher_info['pred_keepid']), memory.shape[2]*len(img_metas)][2]
                loss_fg_feature = sum(fg_loss)/avg_factor
                loss_dict.update({'loss_fg_feature': loss_fg_feature})
        return loss_dict


    # def get_attention(self, preds,  spatial_shapes, temp):
    #     """ preds: Bs*C*W*H """
    #     N, C, HWall= preds.shape
    #
    #     value = torch.abs(preds)
    #     # Bs*W*H
    #     fea_map = value.mean(axis=1, keepdim=True)
    #     S_attention = (HWall * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, HWall)
    #     # print(S_attention.size())
    #     return S_attention

    # def fg_feature_calculation(self, preds_S, preds_T, Mask_fg, Mask_bg, weight=None, avg_factor=None):
    #     loss_mse = nn.MSELoss(reduction='sum')
    #     # self.loss_fg_feature
    #     Mask_fg = Mask_fg.unsqueeze(dim=0).repeat(preds_S.shape[0],1)
    #     Mask_bg = Mask_bg.unsqueeze(dim=0).repeat(preds_S.shape[0],1)
    #
    #     fg_fea_t = torch.mul(preds_T, torch.sqrt(Mask_fg))
    #     bg_fea_t = torch.mul(preds_T, torch.sqrt(Mask_bg))
    #
    #     fg_fea_s = torch.mul(preds_S, torch.sqrt(Mask_fg))
    #     bg_fea_s = torch.mul(preds_S, torch.sqrt(Mask_bg))
    #
    #     fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
    #     bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)
    #
    #     return fg_loss, bg_loss


    def fg_feature_calculation(self, preds_S, preds_T, Mask_fg, Mask_bg, weight=None, avg_factor=None):

        Mask_fg = Mask_fg.unsqueeze(dim=0).repeat(preds_S.shape[0],1)
        Mask_bg = Mask_bg.unsqueeze(dim=0).repeat(preds_S.shape[0],1)

        fg_fea_t = torch.mul(preds_T, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(preds_T, torch.sqrt(Mask_bg))

        fg_fea_s = torch.mul(preds_S, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(preds_S, torch.sqrt(Mask_bg))

        fg_loss = self.loss_fg_feature(fg_fea_s, fg_fea_t,weight=None, avg_factor=None) / len(Mask_fg)
        bg_loss = self.loss_fg_feature(bg_fea_s, bg_fea_t,weight=None, avg_factor=None) / len(Mask_bg)

        return fg_loss#, bg_loss


    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        ##########################################
        bbox_centers = bbox_preds[:,:,:2]
        bbox_lrtb = bbox_preds[:,:,2:]
        bbox_wh = self.integral_average(bbox_lrtb)
        bbox_cxcywh = torch.cat((bbox_centers,bbox_wh.reshape(num_imgs,-1,2)),dim=2)
        bbox_preds_list = [bbox_cxcywh[i] for i in range(num_imgs)] # 变成[center + integral(lrtb)--> normalized xyxy --> normalized cxcywh]
        bbox_lrtb_list = [bbox_lrtb[i] for i in range(num_imgs)]
        ##################################################################

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           bbox_lrtb_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        #####################################################
        bbox_preds_n = torch.cat(bbox_preds_list,0)
        score = label_weights.new_zeros(labels.shape)
        bg_class_ind = self.num_classes
        pos_inds = torch.nonzero((labels >= 0) & (labels < bg_class_ind)).squeeze(1)
        score[pos_inds] = bbox_overlaps(
            bbox_cxcywh_to_xyxy(bbox_preds_n[pos_inds]),
            bbox_cxcywh_to_xyxy(bbox_targets[pos_inds]),
            is_aligned=True)

        #############################################################################


        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = score.new_tensor([num_total_pos]) # loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        #####################################################
        # num_total_samples = reduce_mean(
        #     torch.tensor(num_total_pos, dtype=torch.float,
        #                  device=cls_scores[0].device)).item()
        # # num_total_samples = max(num_total_samples, 1.0)

        loss_cls = self.loss_cls(
            cls_scores, (labels, score), label_weights, avg_factor=num_total_pos)
        # loss_cls = self.loss_cls(
        #     cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        #############################################################################



        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_cxcywh
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        pred_corners = bbox_lrtb.reshape(-1, self.reg_max+1)
        target_corners = bbox_targets[:,2:].unsqueeze(2).repeat(1,1,2).reshape(-1) / 2
        loss_dfl = self.loss_dfl(pred_corners, # [128,17]
                target_corners, # [128]
                weight=bbox_weights.reshape(-1), # weight_target:[32] , weight:[128]
                avg_factor=num_total_pos*4)

        return loss_cls, loss_bbox, loss_iou, loss_dfl




    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   enc_cls_scores,
                   enc_bbox_preds,
                   img_metas,
                   rescale=False,
                   cfg=None,
                   **kwargs):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale, cfg, **kwargs)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False,
                           cfg=None,
                           **kwargs):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        cfg = self.test_cfg if cfg is None else cfg

        max_per_img = cfg.get('max_per_img', self.num_query)
        score_thr = cfg.get('score_thr',0)

        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            # scores, indexes = cls_score.view(-1).topk(max_per_img)
            scores, labels, indexes, filtered_result = filter_scores_and_topk(cls_score, score_thr, max_per_img, results=None)
            # det_labels = indexes % self.num_classes
            # bbox_index = indexes // self.num_classes
            det_labels = labels
            bbox_index = indexes
            bbox_pred = bbox_pred[bbox_index]
            det_logits = cls_score[bbox_index]
            det_keepid = bbox_index
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            # scores, bbox_index = scores.topk(max_per_img)
            scores_, labels_, bbox_index, filtered_result = filter_scores_and_topk(scores, score_thr, max_per_img,
                                                                              results=None)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]
            det_logits = cls_score[bbox_index]
            det_keepid = bbox_index
        ################################################
        bbox_centers = bbox_pred[:,:2]
        bbox_lrtb = bbox_pred[:,2:]
        bbox_wh = self.integral_average(bbox_lrtb)
        bbox_cxcywh = torch.cat((bbox_centers,bbox_wh.reshape(-1,2)),dim=1)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_cxcywh)
        #######################################################
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        if kwargs.get('need_logits', False):
            return det_bboxes, det_labels, det_logits, det_keepid #### TODO: det_scores 和 det_logits区别 ！
        else:
            return det_bboxes, det_labels

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    bbox_lrtb_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, bbox_lrtb_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           bbox_lrtb,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, bbox_lrtb, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

        # if score_factor_list[0] is None:
        #     # e.g. Retina, FreeAnchor, etc.
        #     with_score_factors = False
        # else:
        #     # e.g. FCOS, PAA, ATSS, etc.
        #     with_score_factors = True
        #
        # cfg = self.test_cfg if cfg is None else cfg
        # img_shape = img_meta['img_shape']
        # nms_pre = cfg.get('nms_pre', -1)
        #
        # mlvl_keepid = []
        # mlvl_logits = []
        # mlvl_scores = []
        # mlvl_labels = []
        # mlvl_bboxes = []
        # if with_score_factors:
        #     mlvl_score_factors = []
        # else:
        #     mlvl_score_factors = None
        # for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
        #         enumerate(zip(cls_score_list, bbox_pred_list,
        #                       score_factor_list, mlvl_priors)):
        #
        #     assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        #
        #     bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        #     if with_score_factors:
        #         score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
        #     cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
        #     if self.use_sigmoid_cls:
        #         scores = cls_score.sigmoid()
        #     else:
        #         # remind that we set FG labels to [0, num_class-1]
        #         # since mmdet v2.0
        #         # BG cat_id: num_class
        #         scores = cls_score.softmax(-1)[:, :-1]
        #
        #     # ① PreNMS 预先对每一个Level进行独立NMS，预先过滤
        #     # After https://github.com/open-mmlab/mmdetection/pull/6268/,
        #     # this operation keeps fewer bboxes under the same `nms_pre`.
        #     # There is no difference in performance for most models. If you
        #     # find a slight drop in performance, you can set a larger
        #     # `nms_pre` than before.
        #     results = filter_scores_and_topk(
        #         scores, cfg.score_thr, nms_pre,
        #         dict(bbox_pred=bbox_pred, priors=priors))
        #     scores, labels, keep_idxs, bbox_pred_prior = results
        #     # keep_idxs_list = keep_idxs.tolist()
        #     # print('keep_idxs1=>', len(keep_idxs_list)==len(set(keep_idxs_list)), len(keep_idxs_list), len(set(keep_idxs_list)))
        #     # print(keep_idxs_list)
        #
        #     bbox_pred = bbox_pred_prior['bbox_pred']
        #     priors = bbox_pred_prior['priors']
        #     bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
        #
        #     if with_score_factors:
        #         score_factor = score_factor[keep_idxs]
        #
        #     mlvl_scores.append(scores)  # Nx1
        #     mlvl_labels.append(labels)  # Nx1
        #     mlvl_bboxes.append(bboxes)  # Nx4
        #     if with_score_factors:
        #         mlvl_score_factors.append(score_factor)
        #     if kwargs.get('need_logits', False):
        #         mlvl_keepid.append(keep_idxs)
        #         logits = cls_score[keep_idxs]       # H*W*PxClass
        #         mlvl_logits.append(logits)          # NxClass
        #
        # if kwargs.get('need_logits', False):
        #     kwargs['mlvl_logits'] = mlvl_logits
        #     kwargs['mlvl_keepid'] = mlvl_keepid
        #
        # results = self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
        #                                img_meta['scale_factor'], cfg, rescale,
        #                                with_nms, mlvl_score_factors, **kwargs)
        # return results

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_logits (list[Tensor]): Box logits from all scale
                levels of a single image, each item has shape
                (num_bboxes, num_classes).
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        if kwargs.get('need_logits', False):
            mlvl_logits = kwargs['mlvl_logits']
            mlvl_keepid = kwargs['mlvl_keepid']
            assert len(mlvl_logits) == len(mlvl_keepid) == len(mlvl_labels)
            mlvl_logits = torch.cat(mlvl_logits)
            mlvl_keepid = torch.cat(mlvl_keepid)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        # ② PostNMS 后续对全部Level进行综合NMS，综合过滤
        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                if kwargs.get('need_logits', False):
                    return det_bboxes, mlvl_labels, mlvl_logits, mlvl_keepid
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores, mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            if kwargs.get('need_logits', False):
                # keep_idxs_list = keep_idxs.tolist()
                # mlvl_keepid_list = mlvl_keepid.tolist()
                # print('keep_idxs2=>', len(keep_idxs_list)==len(set(keep_idxs_list)), len(keep_idxs_list), len(set(keep_idxs_list)))
                # print('mlvl_keepid=>', len(mlvl_keepid_list)==len(set(mlvl_keepid_list)), len(mlvl_keepid_list), len(set(mlvl_keepid_list)))
                det_logits = mlvl_logits[keep_idxs][:cfg.max_per_img]
                det_keepid = mlvl_keepid[keep_idxs][:cfg.max_per_img]
                return det_bboxes, det_labels, det_logits, det_keepid
            return det_bboxes, det_labels
        else:
            if kwargs.get('need_logits', False):
                return mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_logits, mlvl_keepid
            else:
                return mlvl_bboxes, mlvl_scores, mlvl_labels