# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmengine
import numpy as np
import torch
from mmcv import BaseTransform
from mmengine.structures import InstanceData, PixelData
from numpy import dtype

from mmengine.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.points import BasePoints
from projects.mmdet3d_plugin.structures import MapTRDataSample, MultiViewPixelData

def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype('float64'):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class CustomPack3DDetInputs(BaseTransform):
    INPUTS_KEYS = ['points', 'img']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels',
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]
    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg',
    ]
    DEPTH_KEYS = [
        'gt_depth',
    ]
    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = ('sample_idx', 'curr_idx', 'prev_idx', 'next_idx', 'scene_token', 
                            'can_bus', 'timestamp', 'map_location', 'lidar2ego', 'ego2global',
                            'lidar2global', 'camera2ego', 'camego2global', 'lidar2cam', 
                            'lidar2img', 'cam_intrinsic', 'annotation'),
        pad_size_divisor = 1
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys
        self.pad_size_divisor = pad_size_divisor

    def _remove_prefix(self, key: str) -> str:
        if key.startswith('gt_'):
            key = key[3:]
        return key

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`MapTRDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`MapTRDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                img = np.stack(results['img'], axis=0)
                if img.flags.c_contiguous:
                    # (num_cam, C, H, W)
                    img = to_tensor(img).permute(0, 3, 1, 2).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(0, 3, 1, 2)))
                results['img'] = img
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                # (C, H, W)
                results['img'] = img

        # TODO: transform corresponding key to tensor
        for key in ['can_bus', 'lidar2ego', 'ego2global',
                    'lidar2global', 'camera2ego', 'camego2global',
                    'lidar2cam', 'lidar2img', 'cam_intrinsic']:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])

        data_sample = MapTRDataSample()
        gt_instances_3d = InstanceData()
        gt_depth = MultiViewPixelData()

        # create metainfo
        data_metas = {}
        pad_h = int(
                    np.ceil(results['img'].shape[-2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
        pad_w = int(
            np.ceil(results['img'].shape[-1] /
                    self.pad_size_divisor)) * self.pad_size_divisor
        data_metas["pad_shape"] = [(pad_h, pad_w) for _ in range(results['img'].shape[0])]

        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]

        data_sample.set_metainfo(data_metas)
        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.DEPTH_KEYS:
                    gt_depth.set_field(results[key], self._remove_prefix(key))
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_depth = gt_depth

        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
