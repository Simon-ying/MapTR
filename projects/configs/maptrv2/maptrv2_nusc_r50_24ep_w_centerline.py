_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.mmdet3d_plugin'], allow_failed_imports=False)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-15.0, -30.0,-10.0, 15.0, 30.0, 10.0]
voxel_size = [0.15, 0.15, 20.0]
dbound=[1.0, 35.0, 0.5]

grid_config = {
    'x': [-30.0, -30.0, 0.15], # useless
    'y': [-15.0, -15.0, 0.15], # useless
    'z': [-10, 10, 20],        # useless
    'depth': [1.0, 35.0, 0.5], # useful
}


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing','boundary','centerline']

num_vec=70
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 100
queue_length = 1 # each sequence contains `queue_length` frames.
metainfo = dict(classes=class_names, map_classes=map_classes)

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=True,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)

model = dict(
    type='MapTRv2',
    data_preprocessor=dict(
        type='CustomDet3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        queue_length=queue_length),
    use_grid_mask=True,
    video_test_mode=False,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='MapTRv2Head',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec_one2one=num_vec,
        num_vec_one2many=300,
        k_one2many=6,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=aux_seg_cfg,
        # z_cfg=z_cfg,
        transformer=dict(
            type='MapTRPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='LSSTransform',
                in_channels=_dim_,
                out_channels=_dim_,
                feat_down_sample=32,
                pc_range=point_cloud_range,
                voxel_size=voxel_size,
                dbound=dbound,
                downsample=2,
                loss_depth_weight=3.0,
                depthnet_cfg=dict(use_dcn=False, with_cp=False, aspp_mid_channels=96),
                grid_config=grid_config,),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DecoupledDetrTransformerDecoderLayer',
                    num_vec=num_vec,
                    num_pts_per_vec=fixed_ptsnum_per_pred_line,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'self_attn', 'norm','cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
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
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.0),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', 
                      loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_seg=dict(type='SimpleLoss', 
            pos_weight=4.0,
            loss_weight=1.0),
        loss_pv_seg=dict(type='SimpleLoss', 
                    pos_weight=1.0,
                    loss_weight=2.0),),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='mmdet.MapTRAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            reg_cost=dict(type='mmdet.CustomBBoxL1Cost', weight=0.0, box_format='xywh'),
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='mmdet.CustomIoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='mmdet.OrderedPtsL1Cost', 
                      weight=5),
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesOfflineLocalMapDataset'
data_root = 'data/nuscenes/'
backend_args = None

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='CustomLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(type='CustomPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='CustomPack3DDetInputs',
        keys=[
            'img', 'points', 'gt_depth'
        ],
        pad_size_divisor=32)
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(
        type='CustomLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(type='CustomPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='CustomPack3DDetInputs',
        keys=[
            'img', 'points', 'gt_depth'
        ],
        pad_size_divisor=32)
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_map_infos_temporal_train.pkl',
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
        aux_seg=aux_seg_cfg,
        use_valid_flag=True,
        backend_args=backend_args,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        queue_length=queue_length,))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_map_infos_temporal_val.pkl',
        map_ann_file=data_root+'/nuscenes_map_anns_val.pkl',
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
        test_mode=False,
        modality=input_modality,
        aux_seg=aux_seg_cfg,
        use_valid_flag=True,
        backend_args=backend_args,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        queue_length=queue_length,))
val_dataloader = test_dataloader

test_evaluator = dict(
    ann_file=data_root+'/nuscenes_map_infos_temporal_val.pkl',
    # map_ann_file=data_root+'/nuscenes_map_anns_val.pkl',
    backend_args=None,
    data_root=data_root,
    metric='bbox',
    type='NuScenesMetric')
val_evaluator = test_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=6e-4, weight_decay=0.01),
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
    val_interval=24)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
find_unused_parameters = False

runner = dict(type='EpochBasedRunner', max_epochs=num_epochs)
load_from = 'ckpts/resnet50-19c8e357.pth'
resume = False