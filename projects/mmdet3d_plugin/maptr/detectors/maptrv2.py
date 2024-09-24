import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.ops import DynamicScatter, Voxelization
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.structures import InstanceData

@MODELS.register_module()
class MapTRv2(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 video_test_mode=False,
                 modality='vision',
                 lidar_encoder=None,
                 **kwargs):

        super(MapTRv2,
              self).__init__(pts_voxel_encoder, pts_middle_encoder,
                             pts_fusion_layer, img_backbone, pts_backbone,
                             img_neck, pts_neck, pts_bbox_head, img_roi_head,
                             img_rpn_head, train_cfg, test_cfg, init_cfg,
                             data_preprocessor, **kwargs)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        # TODO: test with fusion modality
        self.modality = modality
        if self.modality == 'fusion' and lidar_encoder is not None :
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": MODELS.build(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(
                    B // len_queue, len_queue, BN // B, C, H, W))
            else:
                img_feats_reshaped.append(
                    img_feat.view(B, BN // B, C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          lidar_feat,
                          batch_data_samples,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        img_metas = [item.metainfo for item in batch_data_samples]
        outs = self.pts_bbox_head(
            pts_feats, lidar_feat, img_metas, prev_bev)

        depth = outs.pop('depth')
        losses = dict()
        # calculate depth loss
        gt_depth = [item.gt_depth.get("depth", None) for item in batch_data_samples]
        if not any(x is None for x in gt_depth):
            loss_depth = self.pts_bbox_head.transformer.encoder.get_depth_loss(gt_depth, depth)
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_depth = torch.nan_to_num(loss_depth)
            losses.update(loss_depth=loss_depth)

        gt_bboxes_3d = [item.gt_instances_3d["bboxes_3d"] for item in batch_data_samples]
        gt_labels_3d = [item.gt_instances_3d["labels_3d"] for item in batch_data_samples]
        gt_seg_mask = [item.gt_bev_seg.seg_mask for item in batch_data_samples]
        gt_pv_seg_mask = [item.gt_pv_seg.pv_seg_mask for item in batch_data_samples]
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_seg_mask, gt_pv_seg_mask, outs]
        losses_pts = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        losses.update(losses_pts)

        k_one2many = self.pts_bbox_head.k_one2many
        multi_gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d)
        multi_gt_labels_3d = copy.deepcopy(gt_labels_3d)
        for i, (each_gt_bboxes_3d, each_gt_labels_3d) in enumerate(zip(multi_gt_bboxes_3d, multi_gt_labels_3d)):
            each_gt_bboxes_3d.instance_list = each_gt_bboxes_3d.instance_list * k_one2many
            each_gt_bboxes_3d.instance_labels = each_gt_bboxes_3d.instance_labels * k_one2many
            multi_gt_labels_3d[i] = each_gt_labels_3d.repeat(k_one2many)

        one2many_outs = outs['one2many_outs']
        loss_one2many_inputs = [multi_gt_bboxes_3d, multi_gt_labels_3d, gt_seg_mask, gt_pv_seg_mask, one2many_outs]
        loss_dict_one2many = self.pts_bbox_head.loss(*loss_one2many_inputs, img_metas=img_metas)

        lambda_one2many = self.pts_bbox_head.lambda_one2many
        for key, value in loss_dict_one2many.items():
            if key + "_one2many" in losses.keys():
                losses[key + "_one2many"] += value * lambda_one2many
            else:
                losses[key + "_one2many"] = value * lambda_one2many
        return losses
    
    def obtain_history_bev(self, imgs_queue, img_metas_dict):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = img_metas_dict[i]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    def extract_lidar_feat(self, points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)
        
        return lidar_feat

    def loss(self,
             batch_inputs_dict,
             batch_data_samples):
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (bs, len_queue, num_cams, C, H, W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """
        lidar_feat = None
        if self.modality == 'fusion' and 'points' in batch_inputs_dict:
            points = batch_inputs_dict['points']
            lidar_feat = self.extract_lidar_feat(points)
        img = batch_inputs_dict.get('imgs', None)
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        img_metas = {}
        for queue_id in range(len_queue):
            img_metas[queue_id] = [item.metainfo for item in batch_data_samples[queue_id]]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None
        img_metas = img_metas[len_queue-1]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, lidar_feat, 
                                            batch_data_samples[len_queue-1], prev_bev)

        losses.update(losses_pts)
        return losses

    def aug_test(self, batch_inputs_dict, batch_data_samples,  **kwargs):
        return self.predict(batch_inputs_dict, batch_data_samples,  **kwargs)
    
    def predict(self, batch_inputs_dict,
                batch_data_samples,  **kwargs):
        img = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        assert img.size(0) == 1, "only support batch size = 0 now."
        len_queue = img.size(1)
        img_metas = {}
        for queue_id in range(len_queue):
            img_metas[queue_id] = [copy.deepcopy(data_samples.metainfo) for data_samples in batch_data_samples[queue_id]]
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, dict):
                raise TypeError('{} must be a dict, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        if self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[len_queue-1][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[len_queue-1][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[len_queue-1][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[len_queue-1][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[len_queue-1][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[len_queue-1][0]['can_bus'][-1] = 0
            img_metas[len_queue-1][0]['can_bus'][:3] = 0

        img = img[:, len_queue-1, ...]
        img_metas = img_metas[len_queue-1]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        lidar_feat = None
        if self.modality =='fusion':
            lidar_feat = self.extract_lidar_feat(points)
        outs = self.pts_bbox_head(img_feats, lidar_feat, img_metas, prev_bev=self.prev_frame_info['prev_bev'])
        results_list_3d = self.pts_bbox_head.predict_by_feat(outs, img_metas, **kwargs)
        batch_data_samples = batch_data_samples[len_queue-1]
        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d)

        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = outs['bev_embed']
        return detsamples

    def add_pred_to_datasample(
        self,
        data_samples,
        data_instances_3d = None,
        data_instances_2d = None,
    ):
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples

