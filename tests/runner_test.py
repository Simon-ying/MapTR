from mmengine.config import Config
from mmengine.runner import Runner
config = Config.fromfile('example_config.py')
runner = Runner.from_cfg(config)
runner.train()