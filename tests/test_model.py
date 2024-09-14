from mmdet3d.registry import MODELS, DATASETS
from mmengine import Config
import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes
import numpy as np
import copy
import sys


cfg = Config.fromfile("tests/maptrv2_nusc_r50_24ep_w_centerline.py")
dataset = DATASETS.build(cfg.train_dataloader['dataset'])
item = dataset[0]
model = MODELS.build(cfg.model)
import pdb;pdb.set_trace()