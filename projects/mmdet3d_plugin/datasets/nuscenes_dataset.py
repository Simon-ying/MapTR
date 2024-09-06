import copy

import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmengine
from os import path as osp
import torch
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmengine.structures import BaseDataElement
from mmdet3d.structures import Det3DDataSample
import random
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.structures import InstanceData

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        
    def prepare_data(self, index):
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            example = self.pipeline(input_dict)
            if not self.test_mode and self.filter_empty_gt and \
                (example is None or ~(example["data_samples"].gt_instances_3d.labels_3d != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)


    def union2one(self, queue):
        imgs_list = [each['inputs']['img'] for each in queue]

        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['data_samples']
            metainfo_i = copy.deepcopy(metas_map[i].metainfo)
            if not self.test_mode:
                if metainfo_i['scene_token'] != prev_scene_token:
                    metainfo_i['prev_bev_exists'] = False
                    prev_scene_token = metainfo_i['scene_token']
                    prev_pos = copy.deepcopy(metainfo_i['can_bus'][:3])
                    prev_angle = copy.deepcopy(metainfo_i['can_bus'][-1])
                    metainfo_i['can_bus'][:3] = 0
                    metainfo_i['can_bus'][-1] = 0
                else:
                    metainfo_i['prev_bev_exists'] = True
                    tmp_pos = copy.deepcopy(metainfo_i['can_bus'][:3])
                    tmp_angle = copy.deepcopy(metainfo_i['can_bus'][-1])
                    metainfo_i['can_bus'][:3] -= prev_pos
                    metainfo_i['can_bus'][-1] -= prev_angle
                    prev_pos = copy.deepcopy(tmp_pos)
                    prev_angle = copy.deepcopy(tmp_angle)
            metas_map[i].set_metainfo(metainfo_i)
        queue[-1]['inputs']['img'] = torch.stack(imgs_list)
        queue[-1]['data_samples'] = metas_map
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = super().get_data_info(index)
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            input_dict.update(
                dict(
                    images = dict()
                )
            )
            for cam_type, cam_info in info['cams'].items():
                cam_dict = dict()
                cam_dict["img_path"] = cam_info['data_path']
                cam_dict["cam2img"] = cam_info['cam_intrinsic']

                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                cam_dict["lidar2cam"] = lidar2cam_rt.T
                input_dict["images"][cam_type] = cam_dict

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            input_dict['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            input_dict['eval_ann_info'] = self.parse_ann_info(info)

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def _filter_with_mask(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos.

        Returns:
            dict: Annotations after filtering.
        """
        filtered_annotations = {}
        if self.use_valid_flag:
            filter_mask = ann_info['valid_flag']
        else:
            filter_mask = ann_info['num_lidar_pts'] > 0
            # filter_mask = ann_info['']
        for key in ann_info.keys():
            if key != 'instances':
                filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                filtered_annotations[key] = ann_info[key]
        return filtered_annotations
    
    def parse_ann_info(self, info: dict):
        """Process the `instances` in data info to `ann_info`.

        In `Custom3DDataset`, we simply concatenate all the field
        in `instances` to `np.ndarray`, you can do the specific
        process in subclass. You have to convert `gt_bboxes_3d`
        to different coordinates according to the task.

        Args:
            info (dict): Info dict.

        Returns:
            dict or None: Processed `ann_info`.
        """
        # add s or gt prefix for most keys after concat
        # we only process 3d annotations here, the corresponding
        # 2d annotation process is in the `LoadAnnotations3D`
        # in `transforms`
        ann_list = ["gt_boxes", "gt_names", "gt_velocity", "num_lidar_pts", "num_radar_pts", "valid_flag"]
        name_mapping = {
            'gt_boxes': 'gt_bboxes_3d',
            'gt_names': 'gt_labels_3d',
            'gt_velocity': 'velocities',
        }
        # empty gt
        if len(info["gt_boxes"]) == 0:
            return None
        else:
            ann_info = dict()
            for ann_name in ann_list:
                temp_anns = copy.deepcopy(info[ann_name])
                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping[ann_name]
                else:
                    mapped_ann_name = ann_name
                if ann_name == "gt_names":
                    for ind, name in enumerate(temp_anns):
                        if name in self.metainfo['classes']:
                            temp_anns[ind] = self.metainfo['classes'].index(name)
                        else:
                            temp_anns[ind] = -1
                    temp_anns = np.array(temp_anns).astype(np.int64)
                elif ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32) 
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns

            for label in ann_info['gt_labels_3d']:
                if label != -1:
                    self.num_ins_per_cat[label] += 1
            ann_info = self._filter_with_mask(ann_info)
            new_ann_info = dict()
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.hstack(
                    (ann_info["gt_bboxes_3d"], 
                     ann_info["velocities"]))
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
            ann_info['gt_bboxes_3d'] = gt_bboxes_3d

            return ann_info
    
    def parse_data_info(self, info: dict):
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)
        return info
