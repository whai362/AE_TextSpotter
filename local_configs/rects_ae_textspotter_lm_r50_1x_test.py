# model settings
data_root = 'data/ReCTS/'
char_dict_file = 'char_dict.json'
model = dict(
    type='AE_TextSpotter',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='AETSRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_strides=[4, 8, 16, 32, 64],
        text_anchor_ratios=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        char_anchor_ratios=[0.5, 1.0, 2.0],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    # text detection module
    text_bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    text_bbox_head=dict(
        type='AETSBBoxHead',
        num_shared_fcs=0,
        num_cls_convs=2,
        num_reg_convs=2,
        in_channels=256,
        conv_out_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=True,
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    text_mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    text_mask_head=dict(
        type='AETSMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=2,
        loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
    # character-based recognition module
    char_bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    char_bbox_head=dict(
        type='AETSBBoxHead',
        num_shared_fcs=0,
        num_cls_convs=4,
        num_reg_convs=2,
        in_channels=256,
        conv_out_channels=256,
        fc_out_channels=1024,
        roi_feat_size=14,
        num_classes=3614,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=True,
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    crm_cfg=dict(
        char_dict_file=data_root + char_dict_file,
        char_assign_iou=0.3),
    # language module
    lm_cfg=dict(
        dictmap_file=data_root + 'dictmap_to_lower.json',
        bert_vocab_file='bert-base-chinese/bert-base-chinese-vocab.txt',
        bert_cfg_file='bert-base-chinese/bert-base-chinese-config.json',
        bert_model_file='bert-base-chinese/bert-base-chinese-pytorch_model.bin',
        sample_num=32,
        pos_iou=0.8,
        lang_score_weight=0.3,
        lang_model=dict(
            input_dim=768,
            output_dim=2,
            gru_num=2,
            with_bi=True)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    text_rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    char_rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    text_rpn=dict(
        nms_across_levels=False,
        nms_pre=900,
        nms_post=900,
        max_num=900,
        nms_thr=0.7,
        min_bbox_size=0),
    char_rpn=dict(
        nms_across_levels=False,
        nms_pre=900,
        nms_post=900,
        max_num=900,
        nms_thr=0.5,  # 0.7
        min_bbox_size=0),
    text_rcnn=dict(
        score_thr=0.01,
        nms=dict(type='nms', iou_thr=0.9),
        max_per_img=500,
        mask_thr_binary=0.5),
    char_rcnn=dict(
        score_thr=0.1,
        nms=dict(type='nms', iou_thr=0.1),
        max_per_img=200,
        mask_thr_binary=0.5),
    recognizer=dict(
        char_dict_file=data_root + char_dict_file,
        char_assign_iou=0.5),
    poly_iou=0.1,
    ignore_thr=0.3)
# dataset settings
dataset_type = 'ReCTSDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=[(1664, 672), (1664, 928)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0),  # 0.5
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
         meta_keys=['filename', 'annpath', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'img_norm_cfg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=8,
    # imgs_per_gpu=1,
    # workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/gt/',
        img_prefix='train/img/',
        cache_file='tda_rects_train_cache_file.json',
        char_dict_file=char_dict_file,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/gt/',
        img_prefix='train/img/',
        cache_file='tda_rects_val_cache_file.json',
        char_dict_file=char_dict_file,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=None,
        img_prefix='test/img/',
        cache_file='tda_rects_test_cache_file.json',
        char_dict_file=char_dict_file,
        pipeline=test_pipeline)
)
# optimizer
optimizer = dict(type='SGD', lr=0.20, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/rects_ae_textspotter_lm_r50_1x/'
load_from = 'work_dirs/rects_ae_textspotter_r50_1x/epoch_12.pth'
resume_from = None
workflow = [('train', 1)]
