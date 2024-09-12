# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmengine.structures import PixelData
from mmdet3d.structures import Det3DDataSample

class MapTRDataSample(Det3DDataSample):

    @property
    def gt_depth(self) -> PixelData:
        return self._gt_depth

    @gt_depth.setter
    def gt_depth(self, value: PixelData) -> None:
        self.set_field(value, '_gt_depth', dtype=PixelData)

    @gt_depth.deleter
    def gt_depth(self) -> None:
        del self._gt_depth

    @property
    def pred_depth(self) -> PixelData:
        return self._pred_depth

    @pred_depth.setter
    def pred_depth(self, value: PixelData) -> None:
        self.set_field(value, '_pred_depth', dtype=PixelData)

    @pred_depth.deleter
    def pred_depth(self) -> None:
        del self._pred_depth

    @property
    def gt_bev_seg(self) -> PixelData:
        return self._gt_bev_seg

    @gt_bev_seg.setter
    def gt_bev_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_bev_seg', dtype=PixelData)

    @gt_bev_seg.deleter
    def gt_bev_seg(self) -> None:
        del self._gt_bev_seg

    @property
    def pred_bev_seg(self) -> PixelData:
        return self._pred_bev_seg

    @pred_bev_seg.setter
    def pred_bev_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_bev_seg', dtype=PixelData)

    @pred_bev_seg.deleter
    def pred_bev_seg(self) -> None:
        del self._pred_bev_seg



SampleList = List[MapTRDataSample]
OptSampleList = Optional[SampleList]
ForwardResults = Union[Dict[str, torch.Tensor], List[MapTRDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
