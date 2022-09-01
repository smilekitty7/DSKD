_base_ = [
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/qoqo_detection.py',
    # '../_base_/datasets/mini_detection.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='YOLOF',
    teacher_config=None,
    teacher_ckpt=None,
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(3,),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',      # caffe
    #     init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet50-19c8e357.pth')),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # [64, 128, 256, 512]
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet18-5c106cde.pth')),
    # backbone=dict(
        ## _delete_=True,
        # type='PyramidVisionTransformerV2',
        # embed_dims=64,
        # num_layers=[3, 4, 6, 3],
        # init_cfg=dict(checkpoint='/home/zhangjp/softlink/Pretrained/pvt_v2_b2.pth')),
    neck=dict(
        type='DilatedEncoder',
        in_channels=512,       # 2048  512
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4),
    bbox_head=dict(
        type='YOLOFHead',
        old_classes=20,
        new_classes=20,
        num_classes=80,
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2, 4, 8, 16],
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7, match_times=4),
        allowed_border=-1,
        pos_weight=-1,          # label_weights[pos_inds] = self.train_cfg.pos_weight
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# data
data = dict(
    samples_per_gpu=16, workers_per_gpu=4,
    train=dict(test_mode=False, catsplit=(40, 40), catload=(0, 1), catwise=True, imgpercent=1),
    val=dict(test_mode=True, catsplit=(40, 40), catload=(1, 1), catwise=True, imgpercent=1),
    test=dict(test_mode=True, catsplit=(40, 40), catload=(0, 1), catwise=True, imgpercent=1),
)

# runtime
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]

# # optimizer
# sgd1:0.02 N=8; sgd2:0.04 N=16; sgd3:0.04 N=8; sgd4:0.06 N=8; sgd5:0.12 N=8;
optimizer = dict(type='SGD',
                 lr=0.04,
                 momentum=0.9,
                 weight_decay=0.0001,
                 paramwise_cfg=dict(norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)}))
# optimizer = [optimizer] * len(task['task_cats'])
optimizer_config = dict(grad_clip=None)
# optimizer = dict(type='AdamW', lr=0.000025, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

# # learning policy
# # lr0 policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
# lr_config = [lr_config] * len(task['task_cats'])

runner = dict(type='EpochBasedRunner', max_epochs=12)
# runner = [runner] * len(task['task_cats'])