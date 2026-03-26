from opengaze.engine.transform import BaseTransform
from opengaze.registry import TRANSFORMS
from opengaze.runtime.scripts import ScriptEnv, ScriptOptions

from mmengine.config import Config
from mmengine.runner import Runner

import argparse
import numpy as np
import os.path as osp
import torchvision.transforms.functional as F


# Data Transformations
@TRANSFORMS.register_module()
class FaceGazeRandomHFlip(BaseTransform):
  def transform(self, data_dict: dict):
    if np.random.random() < 0.5:
      data_dict['face'] = F.hflip(data_dict['face'])
      data_dict['gaze'][1] = -data_dict['gaze'][1]
    return data_dict


# Script Configuration
def build_data_config_dict(opts: argparse.Namespace):
  # Dataset config
  dataset_kwargs = dict(
    type='FaceGazeDataset',
    data_name=opts.data_name,
    root=osp.join('data', opts.data_name),
    transform=[dict(type='ToTensor')],
  )
  train_dataset = dict(
    train=True, **dataset_kwargs,
    pipeline=[
      dict(type='FaceGazeRandomHFlip'),
      dict(
        type='RandomImageAugmentation',
        image_data_key='face',
        p=0.6, n_max_effective=4, drop_batch_dim=True,
        normalize_kwargs=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        color_jiggle_kwargs=dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        random_gray_scale_kwargs=dict(p=0.4),
        random_gaussian_blur_kwargs=dict(kernel_size=3, sigma=(0.1, 2.0), p=0.4),
        random_motion_blur_kwargs=dict(kernel_size=3, angle=80.0, direction=0, p=0.4),
        random_sharpness_kwargs=dict(sharpness=(0.1, 0.5), p=0.4),
        random_gamma_kwargs=dict(gamma=(0.5, 2.0), gain=(0.9, 1.1), p=0.4),
        random_posterize_kwargs=dict(bits=6, p=0.4),
        random_jpeg_kwargs=dict(jpeg_quality=(20, 80), p=0.4),
        random_planckian_jitter_kwargs=dict(p=0.4),
      ),
    ],
    subset=opts.train_subset,
  )
  test_dataset = dict(
    train=False, **dataset_kwargs,
    pipeline=[
      dict(
        type='RandomImageAugmentation',
        image_data_key='face',
        normalize_kwargs=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ),
    ],
    subset=opts.test_subset,
  )

  # Metric config
  metric_cfgs = ScriptEnv.load_config_dict('configs/metric/gaze-3d.py')
  evaluator_metrics = metric_cfgs['AngularError']

  # Dataloader config
  dataloader_kwargs = dict(
    num_workers=opts.num_workers,
    batch_size=opts.batch_size,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
  )
  if opts.mode == 'train':
    config_dict = dict(
      train_dataloader=dict(dataset=train_dataset, **dataloader_kwargs),
      train_cfg=dict(by_epoch=True, max_epochs=opts.max_epochs),
    )
    if not opts.skip_test:
      config_dict.update(
        val_dataloader=dict(dataset=test_dataset, **dataloader_kwargs),
        val_cfg=dict(type='ValLoop'),
        val_evaluator=evaluator_metrics,
      )

  if opts.mode == 'test':
    config_dict = dict(
      test_dataloader=dict(dataset=test_dataset, **dataloader_kwargs),
      test_cfg=dict(type='TestLoop'),
      test_evaluator=evaluator_metrics,
    )

  return config_dict

def build_optim_dict(opts: argparse.Namespace):
  optimizer = dict(type='AdamW', lr=opts.base_lr, betas=(0.90, 0.95), weight_decay=5e-3)
  optim_wrapper_cls = 'AmpOptimWrapper' if opts.mixed_precision else 'OptimWrapper'
  optim_wrapper = dict(type=optim_wrapper_cls, optimizer=optimizer)

  param_scheduler = [
    dict(
      type='LinearLR', by_epoch=True,
      start_factor=opts.warm_up_ratio,
      end_factor=1.0,
      begin=0, end=opts.warn_up_epoch,
      convert_to_iter_based=True,
    ),
    dict(
      type='LinearLR', by_epoch=True,
      start_factor=1.0,
      end_factor=opts.cool_down_ratio,
      begin=opts.cool_down_epoch,
      end=opts.max_epochs,
      convert_to_iter_based=True,
    ),
  ]

  return dict(optim_wrapper=optim_wrapper, param_scheduler=param_scheduler)

def build_config(opts: argparse.Namespace):
  # Default runtime config
  config = ScriptEnv.load_config_dict('configs/default-runtime.py')

  # Model config
  model_cfgs = ScriptEnv.load_config_dict('configs/model/gaze-3d.py')
  if not opts.model_name in model_cfgs:
    raise RuntimeError(f'Model "{opts.model_name}" not found in model configs.')
  config['model'] = model_cfgs[opts.model_name]

  # Dataset, Evaluator and Loop config
  data_config_dict = build_data_config_dict(opts)
  config.update(data_config_dict)

  # Optimizer, Scheduler config
  if opts.mode == 'train':
    optim_dict = build_optim_dict(opts)
    config.update(optim_dict)

    # Hook config
    config['custom_hooks'] = [
      dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        rule='less',
        save_last=False,
      ),
    ]
    if opts.ema_epoch in range(opts.max_epochs):
      ema_hook = dict(type='EMAHook', begin_epoch=opts.ema_epoch)
      config['custom_hooks'].append(ema_hook)

    # Auto config
    config['auto_scale_lr'] = dict(enable=True, base_batch_size=opts.batch_size)

  return Config(config)


# Entrypoint and Arguments
def numeric_type(value):
  try:  # Parse a string as int or float
    if '.' in value or 'e' in value.lower():
      return float(value)
    else:
      return int(value)
  except ValueError:
      raise argparse.ArgumentTypeError(f'Invalid numeric value: "{value}".')

def main_procedure(opts: argparse.Namespace):
  ScriptEnv.unified_runtime_environment(opts.post_mortem)

  config = build_config(opts)
  ScriptEnv.merge_config(config, opts)

  runner = Runner.from_cfg(config)
  if opts.mode == 'train':
    runner.train()
  if opts.mode == 'test':
    runner.test()



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='run script for FaceGaze experiments.')

  parser.add_argument(
    '--mode', choices=['train', 'test'], default='train',
    help='select mode for script, train or test.',
  )
  parser.add_argument(
    '--post-mortem', action='store_true', default=False,
    help='enable post-mortem debugging.',
  )

  model_group = parser.add_argument_group(
    title='model options',
    description='model options for script.',
  )

  model_group.add_argument(
    '--model-name', required=True, choices=[
      'XGaze224', 'GazeTR', 'UniGaze',
    ],
    help='select model name for current run.',
  )

  data_group = parser.add_argument_group(
    title='data options',
    description='data options for script.',
  )

  data_group.add_argument(
    '--data-name', required=True, choices=[
      'buaa-gazegene', 'eth-xgaze-224', 'idiap-eyediap',
      'mit-gaze-360', 'mpii-facegaze', 'ucas-synthgaze',
    ],
    help='select data name for current run.',
  )
  data_group.add_argument(
    '--skip-test', action='store_true', default=False,
    help='skip validation data or test data for training process.',
  )
  data_group.add_argument(
    '--train-subset', type=numeric_type,
    help='use a subset (int: size, float: ratio) of training data.',
  )
  data_group.add_argument(
    '--test-subset', type=numeric_type,
    help='use a subset (int: size, float: ratio) of test data.',
  )
  data_group.add_argument(
    '--num-workers', type=int, default=4,
    help='number of workers for pytorch dataloader.',
  )
  data_group.add_argument(
    '--batch-size', type=int, default=50,
    help='batch size for pytorch dataloader.',
  )

  config_group = parser.add_argument_group(
    title='config options',
    description='config options for script.',
  )

  config_group.add_argument(
    '--base-lr', type=float, default=1e-4,
    help='base learning rate for training.',
  )
  config_group.add_argument(
    '--max-epochs', type=int, default=50,
    help='max number of epochs for training.',
  )
  config_group.add_argument(
    '--warm-up-ratio', type=float, default=0.01,
    help='warm up ratio for learning rate scheduler.',
  )
  config_group.add_argument(
    '--warn-up-epoch', type=float, default=1.00,
    help='warn up epoch for learning rate scheduler.',
  )
  config_group.add_argument(
    '--cool-down-ratio', type=float, default=0.1,
    help='cool down ratio for learning rate scheduler.',
  )
  config_group.add_argument(
    '--cool-down-epoch', type=float, default=1.00,
    help='cool down epoch for learning rate scheduler.',
  )
  config_group.add_argument(
    '--mixed-precision', action='store_true', default=False,
    help='enable mixed precision training.',
  )
  config_group.add_argument(
    '--ema-epoch', type=int, default=-1,
    help='begin epoch of exponential moving average.',
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
