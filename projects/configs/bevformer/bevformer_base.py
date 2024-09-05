_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.mmdet3d_plugin'], allow_failed_imports=False)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 2 # each sequence contains `queue_length` frames.

model = dict(
    type='BEVFormer',
    data_preprocessor=dict(
        type='CustomDet3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        queue_length=queue_length),
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        # "BN2d and DCNv2 in mmcv, fallback_on_stride used in resnet.py in mmdet"
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            rotate_center=[bev_h_//2, bev_w_//2],
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    embed_dims=_dim_,
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    self_attn_cfg=dict(
                        type='MultiheadAttention',
                        embed_dims=_dim_,
                        num_heads=8,
                        dropout=0.1
                    ),
                    cross_attn_cfg=dict(
                        type='CustomMSDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=1
                    ),
                    ffn_cfg=dict(
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    )))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding3D',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='mmdet.IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
backend_args = None

ida_aug_conf = {
    'resize_lim': (0.47, 0.625),
    'final_dim': (320, 800),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}

train_pipeline = [
     dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=6,
        backend_args=backend_args),
    dict(type='MultiViewPhotoMetricDistortion3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    # dict(
    #     type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=False,
        training=True),
    dict(
        type='CustomPack3DDetInputs',
        keys=[
            'img', 'gt_bboxes_3d', 'gt_labels_3d'
        ])
]

test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        backend_args=backend_args),
    # dict(
    #     type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf,
    #     training=False),
    dict(type='CustomPack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_temporal_train.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        pipeline=train_pipeline,
        load_type="mv_image_based",
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=False,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_temporal_val.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        pipeline=test_pipeline,
        load_type="mv_image_based",
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_temporal_val.pkl',
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        pipeline=test_pipeline,
        load_type="mv_image_based",
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args,
        bev_size=(bev_h_, bev_w_)))

test_evaluator = dict(
    ann_file=data_root+'/nuscenes_infos_temporal_val.pkl',
    backend_args=None,
    data_root=data_root,
    metric='bbox',
    type='NuScenesMetric')
val_evaluator = test_evaluator

optim_wrapper = dict(
    # TODO Add Amp
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2))

num_epochs = 24

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        # TODO Figure out what T_max
        T_max=num_epochs,
        by_epoch=True,
    )
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=num_epochs,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
find_unused_parameters = False

runner = dict(type='EpochBasedRunner', max_epochs=num_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
resume = False
