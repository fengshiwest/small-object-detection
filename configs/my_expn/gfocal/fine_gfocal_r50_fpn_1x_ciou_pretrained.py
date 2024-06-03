_base_ = [
    '../../_base_/datasets/visdrone_detection_UFP.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='GFL',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        loss_bbox=dict(type='CIoULoss', loss_weight=2.0)),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=500))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='VisDroneDataset',
            ann_file=
            './data/VisDrone/annotations/instance_UFP_gfocal_r50_fpn_1x_ciou_pretrained_UAVtrain.json',
            img_prefix=
            './data/VisDrone/VisDrone2019-DET-train-UFP_gfocal_r50_fpn_1x_ciou_pretrained/images/',
    )),
    val=dict(
        type='VisDroneDataset',
        ann_file=
        './data/VisDrone/annotations/instance_UFP_gfocal_r50_fpn_1x_ciou_pretrained_UAVval.json',
        img_prefix=
        './data/VisDrone/VisDrone2019-DET-val-UFP_gfocal_r50_fpn_1x_ciou_pretrained/images/',
    ),
    test=dict(
        type='VisDroneDataset',
        ann_file=
        './data/VisDrone/annotations/instance_UFP_gfocal_r50_fpn_1x_ciou_pretrained_UAVval.json',
        img_prefix=
        './data/VisDrone/VisDrone2019-DET-train-UFP_gfocal_r50_fpn_1x_ciou_pretrained/images/',
    )
)

load_from = 'checkpoint/gfocal_r50_fpn_1x.pth'

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

