_base_ = [
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/wrxt_detection.py',
    '../_base_/default_runtime.py']

# task settings
task = dict(
    # teacher_config='../configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py'
    # teacher_ckpt='/home/softlink/Pretrained/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'
    resume=[False, 1, 2, 3][2],
    Task2={
        'load_teacher': 0,
        'load_student': 1,
        # 'teacher_config': '/home/zhangjp/projects/now-projects/xmdet220/configs/yoloy/yoloy_resnet_qoqo_il.py',
        # 'teacher_ckpt': '/home/softlink/zhjpexp/yoloy-r18-stst-qoqo-il20-v0/epoch_12.pth',
        # 'student_ckpt': '/home/softlink/zhjpexp/yoloy-r18-stst-qoqo-il20-v0/epoch_12.pth',
        'student_ckpt': '/home/xdata/zhangjp/experiments/common_exp_il/task_2_epoch_4.pth',
    },
)

# model settings
model = dict(
    type='YOLOY',
    # backbone=dict(
    #     type='CSPDarknet',
    #     deepen_factor=0.33,
    #     widen_factor=0.5),           # [128, 256, 512]
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(1, 2, 3),      # [256, 512, 1024, 2048]
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet50-19c8e357.pth')),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),  # [64, 128, 256, 512]
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet18-5c106cde.pth')),
    # t:[96, 192, 384]->96     s: [128, 256, 512]->128,
    # m:[192, 384, 768]->192,  l:[256, 512, 1024]->256
    # neck=dict(
    #     type='YOLOYPAFPN',
    #     in_channels=[128, 256, 512],
    #     out_channels=128,
    #     num_csp_blocks=1),
    neck=dict(
        type='YOLOYPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        conv_cfg=None,
        act_cfg=dict(type='Swish'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), ),
    # bbox_head=dict(
    #     type='YOLOYHead',
    #     num_classes=9,
    #     in_channels=128,
    #     feat_channels=128),
    bbox_head=dict(
        type='YOLOYHead',
        num_classes=9,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=[8, 16, 32],  # org 8, 16, 32，与前面fmap的levels相等
        use_depthwise=False,
        dcn_on_last_conv=False,
        conv_bias='auto',
        conv_cfg=None,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1.0),
        loss_obj=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1.0),
        # loss_bbox=dict(type='IoULoss', mode='square', eps=1e-16, reduction='mean', loss_weight=5.0),
        loss_bbox=dict(type='DIoULoss', eps=1e-16, reduction='mean', loss_weight=5.0),
        # loss_bbox=dict(type='CIoULoss', eps=1e-16, reduction='mean', loss_weight=5.0),
        loss_l1=dict(type='L1Loss', reduction='mean', loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        # for increment il
        reg_val={'min': 0, 'max': 16, 'num': 17, 'usedfl': False},
        cates_distill='hard+soft',    # hard + hardsoft + normsoft + soft
        locat_distill='',             # bbox + logit + (#decode & #encode)
        feats_distill='kldv',             # kldv
        loss_kd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1, T=2, reduction='mean'),
        loss_ld_bbox=dict(type='SmoothL1Loss', loss_weight=5, reduction='mean'),            # wo #decode
        # loss_ld_bbox=dict(type='L1Loss', beta=0.11, loss_weight=1.0, reduction='mean'),   # w #decode
        # loss_ld_bbox=dict(type='DIoULoss', loss_weight=1, reduction='mean'),              # w #decode
        loss_ld_logit=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1, T=2, reduction='mean'),
        loss_fd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1, T=2, reduction='sum'),
    ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
    # 配置Teacher的测试输出
    teacher_test_cfg=dict(score_thr=0.3, nms=dict(type='nms', iou_threshold=0.65)),
)

# catsplit, catload = (9, ), (1, )
# catsplit, catload = (5, 4), (1, 0)
catsplit, catload = (3, 3, 3), (1, 1, 1)
# catsplit, catload = (2, 2, 2, 2, 1), (1, 0, 1, 0, 1)
data = dict(
    samples_per_gpu=4, workers_per_gpu=4,
    train=dict(test_mode=False, catsplit=catsplit, catload=catload, catwise=True, imgpercent=1),
    val=dict(test_mode=True, catsplit=catsplit, catload=catload, catwise=True, imgpercent=1),
    test=dict(test_mode=True, catsplit=catsplit, catload=catload, catwise=True, imgpercent=1),
)
task_nums = len(data['train']['catsplit'])

# runtime
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]

################## MMDet 配置 8x4_1x ############################
# # optimizer op1 ==> schedule_1x.py
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# optimizer op2
# optimizer = dict(
#     type='SGD', lr=0.02,
#     momentum=0.9, weight_decay=0.0001, nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)
# # optimizer op3
# optimizer = dict(type='AdamW', lr=0.000025, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
optimizer = [optimizer] * task_nums

# # learning policy  lr1 ==> schedule_1x.py
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[8, 11])
# # learning policy  lr2
# lr_config = dict(
#     policy='YOLOY',
#     warmup='exp',
#     by_epoch=False,
#     warmup_ratio=0.01,       # warmup_start_lr = warmup_ratio * initial_lr, warmup_end_lr=initial_lr
#     warmup_iters=800,        # 1 epoch
#     warmup_by_epoch=False,   # warmup_iters指向iter//epoch
#     num_last_epochs=1,       # 最后阶段的稳定学习率
#     min_lr_ratio=0.01)       # 最后阶段的最终最小学习率，ended_lr = min_lr_ratio * initial_lr
lr_config = [lr_config] * task_nums

runner = dict(type='TaskEpochBasedRunner', max_epochs=4, max_tasks=task_nums, save_teacher=False)
runner = [runner] * task_nums
