import numpy as np
from numpy import random
import mmcv
from mmdet3d.registry import TRANSFORMS
from mmdet.datasets.transforms import (PhotoMetricDistortion, RandomCrop,
                                       RandomFlip)
from mmcv.transforms import BaseTransform, Compose, RandomResize, Resize

@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage(PhotoMetricDistortion):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    PhotoMetricDistortion3D further support using predefined randomness
    variable to do the augmentation.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Required Keys:

    - img (np.uint8)

    Modified Keys:

    - img (np.float32)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        assert 'img' in results, '`img` is not found in results'
        img = results['img']
        img = img.astype(np.float32)
        if 'photometric_param' not in results:
            photometric_param = self._random_flags()
            results['photometric_param'] = photometric_param
        else:
            photometric_param = results['photometric_param']

        (mode, brightness_flag, contrast_flag, saturation_flag, hue_flag,
         swap_flag, delta_value, alpha_value, saturation_value, hue_value,
         swap_value) = photometric_param

        # random brightness
        if brightness_flag:
            img += delta_value

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        if mode == 1:
            if contrast_flag:
                img *= alpha_value

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if saturation_flag:
            img[..., 1] *= saturation_value

        # random hue
        if hue_flag:
            img[..., 0] += hue_value
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if contrast_flag:
                img *= alpha_value

        # randomly swap channels
        if swap_flag:
            img = img[..., swap_value]

        results['img'] = img
        return results


@TRANSFORMS.register_module()
class RandomScaleImageMultiViewImage(BaseTransform):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['imgs']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['imgs']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['imgs'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['imgs'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        img_aug_matrix = [scale_factor for _ in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_aug_matrix'] = img_aug_matrix
        results['img_shape'] = [img.shape for img in results['imgs']]
        results['ori_shape'] = [img.shape for img in results['imgs']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@TRANSFORMS.register_module()
class CustomPointsRangeFilter(BaseTransform):
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict):
        """Transform function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str