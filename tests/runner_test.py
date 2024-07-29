from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import MODELS

config = Config.fromfile('mask-rcnn_r50_fpn.py')
# print(config)
model = MODELS.build(config.model)
# runner = Runner.from_cfg(config)
# runner.train()