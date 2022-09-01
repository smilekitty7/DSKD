_base_ = [
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/qoqo_detection.py',
    # '../_base_/datasets/mini_detection.py',
    '../_base_/default_runtime.py']

# task settings
# task = dict(
#     resume=[False, 1, 2, 3][0],
#     Task1={'teacher_config': None,
#            'teacher_ckpt': None,
#            'student_ckpt': None},
#     Task2={'teacher_config': None,
#            'teacher_ckpt': None,
#            'student_ckpt': None},
#     Task3={'teacher_config': None,
#            'teacher_ckpt': None,
#            'student_ckpt': None},
# )

# model settings
model = dict(
    type='YOLOX',
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
    #     type='YOLOXPAFPN',
    #     in_channels=[128, 256, 512],
    #     out_channels=128,
    #     num_csp_blocks=1),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        conv_cfg=None,
        act_cfg=dict(type='Swish'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), ),
    # bbox_head=dict(
    #     type='YOLOXHead',
    #     num_classes=9,
    #     in_channels=128,
    #     feat_channels=128),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=80,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=[8, 16, 32],  # org 8, 16, 32，与前面fmap的levels相等
        use_depthwise=False,
        dcn_on_last_conv=False,
        conv_bias='auto',
        conv_cfg=None,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'), ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.3, nms=dict(type='nms', iou_threshold=0.65)))

# data
catsplit, catload = (80, ), (1, )
# catsplit, catload = (40, 40), (1, 0)
# catsplit, catload = (20, 20, 20, 20), (1, 0, 0, 0)
# catsplit, catload = (5, 5, 5, 5, 5), (1, 0, 0, 0, 0)
data = dict(
    samples_per_gpu=2, workers_per_gpu=4,
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
    warmup_iters=300,       # 1500
    warmup_ratio=0.001,
    step=[8, 11])
# # learning policy  lr2
# lr_config = dict(
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_ratio=0.01,       # warmup_start_lr = warmup_ratio * initial_lr, warmup_end_lr=initial_lr
#     warmup_iters=800,        # 1 epoch
#     warmup_by_epoch=False,   # warmup_iters指向iter//epoch
#     num_last_epochs=1,       # 最后阶段的稳定学习率
#     min_lr_ratio=0.01)       # 最后阶段的最终最小学习率，ended_lr = min_lr_ratio * initial_lr
lr_config = [lr_config] * task_nums

runner = dict(type='EpochBasedRunner', max_epochs=12)
runner = [runner] * task_nums