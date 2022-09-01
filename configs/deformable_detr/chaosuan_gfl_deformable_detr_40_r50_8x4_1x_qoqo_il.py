_base_ = [
    '../_base_/datasets/qoqo_detection.py',
    #'../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# task settings
task = dict(
    # teacher_config='../configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py'
    # teacher_ckpt='/home/softlink/Pretrained/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'
    resume_by_task=[False, 1, 2, 3][1],
    resume_by_epoch='',
    Task1={
        'load_teacher': 0,
        'load_student': 1,
        'teacher_config': '/home/zhangjp/project/incremental_mmdet/configs/deformable_detr/gfl_deformable_detr_40_r50_8x4_1x_qoqo_il.py',
       # 'teacher_ckpt': '/home/softlink/experiments/il_learning/gfl_deformable_detr_20_1/epoch_11_20-1.pth',
       # 'student_ckpt': '/home/softlink/experiments/il_learning/gfl_deformable_detr_20_1/epoch_11_20-1.pth',
#         'teacher_ckpt': '/wlsys/kangmx/experiments/il_learning/gfl_deformable_detr_20_r50_8x4_1x_qoqo_hard+fg_v0/task_1_epoch_12.pth',
        'student_ckpt': '/home/softlink/kmxexp/il_learning/gfl_deformable_detr_40_r50_8x4_1x_qoqo_hard_v0/task_1_epoch_12.pth',
    },
)

model = dict(
    type='DeformableDETR_il',
    # pretrained='torchvision://resnet50',
    teacher_config=None,  # bu ke chuan ru, digui!! use yolof.set_teacher()
    teacher_ckpt=None,
    # backbone=dict(
    #     type='ResNet',
    #     depth=18,
    #     num_stages=4,
    #     out_indices=(1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet18-5c106cde.pth')),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet50-19c8e357.pth')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='GFLDeformableDETRHead_il',
        num_query=300,
        num_classes=80,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        # loss_cls=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=2.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.5),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        # for increment il
        cates_distill = 'hard + teacher-first',  # hard + hardsoft + normsoft + soft
        locat_distill = '',
        memory_distill = '',  # memory
        feats_distill = 'corr + fg_info + decode_v1', # kldv # fg_info + bg_info # 'fg_info + bg_info + sg_out' # 'fg_info + bg_info + sg_both' # 'fg_info + bg_info + meaning'
        loss_kd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1, T=2, reduction='mean'),
        loss_ld_bbox=dict(type='SmoothL1Loss', loss_weight=10, reduction='mean'),
        # loss_ld_bbox=dict(type='L1Loss', beta=0.11, loss_weight=1.0, reduction='mean'),
        # loss_ld_bbox=dict(type='GIoULoss', loss_weight=1, reduction='mean'),
        loss_ld_logit=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1, T=2, reduction='mean'),
        loss_fd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1, T=2, reduction='sum'),
        loss_memory=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=2, T=2, reduction='sum'),
        loss_fg_feature=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1, T=2, reduction='sum'),
        # loss_bg_feature=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.01, T=2, reduction='sum'),
        # loss_fg_feature=dict(type='MSELoss', loss_weight=1, reduction='sum'),
        # loss_bg_feature=dict(type='MSELoss', loss_weight=1, reduction='sum'),
        loss_corr=dict(type='MSELoss', loss_weight=1, reduction='mean'),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GFLHungarianAssigner',
            cls_cost=dict(type='QualityFocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        # assigner=dict(
        #     type='HungarianAssigner',
        #     cls_cost=dict(type='FocalLossCost', weight=2.0),
        #     reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
        #     iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100,
                  score_thr=0.0),
    # 配置Teacher的测试输出
    teacher_test_cfg=dict(
        min_bbox_size=0,
        score_thr=0.3,  # used in teacher self.get_bboxes() # 0.3
        max_per_img=100),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),    # (640, 640),
    # dict(type='Resize',
    #      img_scale=[(600, 600), (640, 640), (700, 700)],
    #      multiscale_mode='value',
    #      keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# data
# catsplit, catload = (80, ), (1, )
catsplit, catload = (40, 40), (1, 0)
# catsplit, catload = (20, 20, 20, 20), (1, 0, 0, 0)
# catsplit, catload = (10, 10, 10, 10), (1, 0, 0, 0)
# catsplit, catload = (5, 5, 5, 5, 5), (1, 0, 0, 0, 0)
cat_split_load = ['auto', 'manual', 'auto:任务增量训练 & manual:任务单独训练'][0]
data = dict(
    samples_per_gpu=16, workers_per_gpu=4,cat_split_load=cat_split_load,
    train=dict(test_mode=False, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1),
    val=dict(test_mode=True, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1), # only-cur
    test=dict(test_mode=True, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1),
)
task_nums = len(data['train']['catsplit'])

# runtime
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]

# # optimizer op1
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# optimizer op2
optimizer = dict(
    type='AdamW',
    lr=4e-4, # 2e-4
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
optimizer = [optimizer] * task_nums


# learning policy  lr1
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.01,
    step=[8, 11])
lr_config = [lr_config] * task_nums

runner = dict(type='TaskEpochBasedRunner', max_epochs=12, max_tasks=task_nums, save_teacher=False)
runner = [runner] * task_nums

# # learning policy  lr2
# lr_config = dict(policy='step', step=[40])
# runner = dict(type='EpochBasedRunner', max_epochs=50)
