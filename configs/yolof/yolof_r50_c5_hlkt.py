_base_ = [
    '../_base_/datasets/hlkt_detection_v7.py',
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='YOLOF',
    # pretrained='open-mmlab://detectron/resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',      # caffe
        init_cfg=dict(type='Pretrained', checkpoint='/home/zhangjp/softlink/Pretrained/resnet50-19c8e357.pth')),
    # backbone=dict(
    #     # _delete_=True,
    #     type='PyramidVisionTransformerV2',
    #     embed_dims=64,
    #     num_layers=[3, 4, 6, 3],
    #     init_cfg=dict(checkpoint='/home/zhangjp/softlink/Pretrained/pvt_v2_b2.pth')),
    neck=dict(
        type='DynamicEncoder',
        in_channels=2048,       # 2048，512
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4),
    # neck=dict(
    #     type='DilatedEncoder',
    #     in_channels=2048,  # 2048  512
    #     out_channels=512,
    #     block_mid_channels=128,
    #     num_residual_blocks=4),
    bbox_head=dict(
        type='YOLOFHead',
        num_classes=10,
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
        assigner=dict(type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# data
data = dict(samples_per_gpu=2, workers_per_gpu=2)

# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
optimizer = dict(type='AdamW', lr=0.000025, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

# # learning policy
# # lr0 policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
# # lr1 policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-5)
# runner = dict(type='EpochBasedRunner', max_epochs=12)

# # optimizer
# optimizer = dict(
#     type='SGD',
#     lr=0.12,
#     momentum=0.9,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)}))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=0.00066667,
#     step=[8, 11])

