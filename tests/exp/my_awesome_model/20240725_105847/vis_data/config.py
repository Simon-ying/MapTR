custom_imports = dict(
    allow_failed_imports=False, imports=[
        'runner_utils',
    ])
default_hooks = dict(checkpoint=dict(interval=1, type='CheckpointHook'))
env_cfg = dict(
    backend='nccl', cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork'))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(activation='relu', layers=2, type='MyAwesomeModel')
optim_wrapper = dict(optimizer=dict(lr=0.001, type='Adam'))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        4,
        8,
    ], type='MultiStepLR')
resume = False
train_cfg = dict(by_epoch=True, max_epochs=10, val_begin=2, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(is_train=True, size=10000, type='MyDataset'),
    num_workers=2,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict()
val_dataloader = dict(
    batch_size=1000,
    collate_fn=dict(type='default_collate'),
    dataset=dict(is_train=False, size=1000, type='MyDataset'),
    num_workers=2,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='Accuracy')
work_dir = 'exp/my_awesome_model'
