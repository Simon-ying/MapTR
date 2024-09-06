import sys
from projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset import CustomNuScenesOfflineLocalMapDataset
from projects.mmdet3d_plugin.datasets.transforms import *
''' Configs
'''
point_cloud_range = [-15.0, -30.0,-10.0, 15.0, 30.0, 10.0]
grid_config = {
    'x': [-30.0, -30.0, 0.15], # useless
    'y': [-15.0, -15.0, 0.15], # useless
    'z': [-10, 10, 20],        # useless
    'depth': [1.0, 35.0, 0.5], # useful
}

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

dataset_type = 'CustomNuScenesOfflineLocalMapDataset'
data_root = 'data/nuscenes/'
backend_args = None

train_pipeline = [
    dict(type='mmdet3d.LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='mmdet3d.RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='mmdet3d.PhotoMetricDistortionMultiViewImage'),
    dict(
        type='mmdet3d.LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(type='mmdet3d.CustomPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='mmdet3d.CustomPack3DDetInputs',
        keys=[
            'img', 'depths', 'gt_labels_3d'
        ])
]

''' Create dataset
'''
dataset = CustomNuScenesOfflineLocalMapDataset(
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
    queue_length=queue_length
)

item = dataset(0)