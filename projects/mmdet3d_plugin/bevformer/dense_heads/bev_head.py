import copy
from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils import digit_version

from mmdet.registry import MODELS

from traitlets import import_item
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmengine.model import BaseModule
import numpy as np
from projects.mmdet3d_plugin.bevformer.modules import PerceptionTransformerBEVEncoder
from mmdet3d.models.dense_heads.free_anchor3d_head import FreeAnchor3DHead

@MODELS.register_module()
class BEVHead(BaseModule):
    def __init__(self, 
                 bev_h,
                 bev_w,
                 pc_range,
                 embed_dims,
                 transformer, 
                 positional_encoding: dict,
                 pts_bbox_head_3d: dict, 
                 init_cfg=None,
                 **kwargs,
                 ):
        super(BEVHead, self).__init__(init_cfg=init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.transformer :PerceptionTransformerBEVEncoder = MODELS.build(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        pts_bbox_head_3d.update(kwargs)
        self.pts_bbox_head_3d = MODELS.build(pts_bbox_head_3d)
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        
        self._init_layers()
    def init_weights(self):
        """Initialize weights of the Multi View BEV Encoder"""
        self.transformer.init_weights()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_embed = self.transformer(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        if only_bev:
            return bev_embed
        
        bev_feature = bev_embed.permute(0, 2, 1).reshape(bs, self.embed_dims, self.bev_h, self.bev_w)
        ret = {}
        ret['pred'] = self.pts_bbox_head_3d([bev_feature,])
        if not self.training:
            ret['bev_embed'] = bev_embed
        return ret 
    
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             ret,
             gt_bboxes_ignore=None,
             img_metas=None):
        assert gt_bboxes_ignore is None
        return self.pts_bbox_head_3d.loss(gt_bboxes_list, gt_labels_list, ret['pred'], gt_bboxes_ignore=gt_bboxes_ignore, img_metas=img_metas)
    
    def get_bboxes(self, ret, img_metas, rescale=False):
        return self.pts_bbox_head_3d.get_bboxes(ret['pred'], img_metas)

@MODELS.register_module()
class FreeAnchor3DHeadV2(FreeAnchor3DHead):
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             pred,
             gt_bboxes_ignore=None,
             img_metas=None):
            cls_scores, bbox_preds, dir_cls_preds = pred
            
            return super().loss(cls_scores, bbox_preds, dir_cls_preds, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore)
    def get_bboxes(self, pred, img_metas, rescale=False):
        cls_scores, bbox_preds, dir_cls_preds = pred
        return super().get_bboxes(
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   img_metas,
                   cfg=None,
                   rescale=rescale)