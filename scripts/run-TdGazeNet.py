from opengaze.engine.transform import BaseTransform
from opengaze.registry import TRANSFORMS
from opengaze.runtime.scripts import ScriptEnv, ScriptOptions
from opengaze.utils.image import scaled_crop

from mmengine.config import Config
from mmengine.runner import Runner
from PIL import Image
from typing import Dict, Tuple

import argparse
import cv2
import math
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF


# Data Transformations
@TRANSFORMS.register_module()
class GetFaceAndBBox(BaseTransform):
  def __init__(self, p_noisy: float = 0.5,
               face_bbox_shift: float = 0.0,
               face_bbox_scale: tuple = (1.0, 1.0),
               face_crop_size: int = 224):

    self.p_noisy = p_noisy

    self.face_bbox_shift = face_bbox_shift
    self.face_bbox_scale = face_bbox_scale

    self.face_crop_size = face_crop_size

  def _bbox_from_ldmk(self, landmarks_2d: np.ndarray):
    x_min, y_min = np.min(landmarks_2d, axis=0)
    x_max, y_max = np.max(landmarks_2d, axis=0)

    bbox_cx = (x_min + x_max) / 2.0
    bbox_cy = (y_min + y_max) / 2.0
    bbox_ca = math.sqrt((x_max - x_min) * (y_max - y_min))

    return bbox_cx, bbox_cy, bbox_ca

  def _crop_face(self, image: np.ndarray, bbox: np.ndarray):
    face_crop = scaled_crop(image, bbox, (self.face_crop_size, self.face_crop_size))

    image_h, image_w, _ = image.shape

    center = np.array([image_w / 2, image_h / 2])
    metric = max(image_h, image_w) / 2

    x_min, y_min, x_max, y_max = bbox

    face_bbox = np.concatenate([
      np.array([(x_min + x_max) / 2, (y_min + y_max) / 2]) - center,
      np.array([x_max - x_min, y_max - y_min]),
    ], axis=0) / metric

    return face_crop, face_bbox

  def transform(self, results: dict):
    bbox_cx, bbox_cy, bbox_ca = self._bbox_from_ldmk(
      landmarks_2d=results['face_mesh_light_2d'],
    )

    if random.random() < self.p_noisy:
      shift_x = (2 * random.random() - 1) * bbox_ca * self.face_bbox_shift
      shift_y = (2 * random.random() - 1) * bbox_ca * self.face_bbox_shift
      scale = random.uniform(*self.face_bbox_scale)

      bbox_cx = bbox_cx + shift_x
      bbox_cy = bbox_cy + shift_y
      bbox_ca = bbox_ca * scale

    x_min = bbox_cx - bbox_ca / 2.0
    y_min = bbox_cy - bbox_ca / 2.0
    x_max = bbox_cx + bbox_ca / 2.0
    y_max = bbox_cy + bbox_ca / 2.0

    bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    face_crop, face_bbox = self._crop_face(results['image'], bbox)
    results.update(face_crop=face_crop, face_bbox=face_bbox)

    return results


@TRANSFORMS.register_module()
class PrepareDataDict(BaseTransform):
  def __init__(self, max_gaze_angle: float = 80.0, pupil_vis_thres: float = 0.5):
    self.max_gaze_angle = max_gaze_angle
    self.pupil_vis_thres = pupil_vis_thres

  def _create_eyes_mask(self, gaze: np.ndarray, pupil_vis: np.ndarray):
    neg_z = np.array([0.0, 0.0, -1.0])
    dot = np.dot(gaze, neg_z)
    m_g = np.linalg.norm(gaze)
    sim = np.clip(dot / m_g, -1.0, 1.0)
    deg = np.rad2deg(np.acos(sim))

    vis = np.sum(pupil_vis) / len(pupil_vis)

    mask_1 = 0.0 <= deg and deg <= self.max_gaze_angle
    mask_2 = vis >= self.pupil_vis_thres
    mask = 1.0 if mask_1 and mask_2 else 0.0

    return torch.tensor([mask], dtype=torch.float32)

  def image_fn(self, image: Image):
    return TF.to_tensor(image)

  def torch_fn(self, x: np.ndarray):
    return torch.tensor(x, dtype=torch.float32)

  def transform(self, results: dict):
    data_dict = {k:v for k, v in results.items() if k.endswith('idx')}

    # Shape: (3, 224, 224), ImageNet Normalization
    face = cv2.cvtColor(results['face_crop'], cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face, mode='RGB')
    data_dict.update(face=self.image_fn(face))

    # Shape: (4, ), Standardized BBox: (p - center) / metric
    data_dict.update(bbox=self.torch_fn(results['face_bbox']))

    # Shape: (151, 3), Standard Unit (10cm)
    face_kpts = np.array(results['face_mask_3d']) / 1e2
    data_dict.update(face_kpts=self.torch_fn(face_kpts))

    # Shape: (2, 110, 3), Standard Unit (10cm)
    eyes_kpts = np.stack([results['reye_globe_3d'], results['leye_globe_3d']]) / 1e2
    data_dict.update(eyes_kpts=self.torch_fn(eyes_kpts))

    # Shape: (2, 2, 3), Standard Unit (10cm)
    eyes_gaze = np.stack([
      np.stack([results['reye_origin'] / 1e2, results['reye_vector']]),
      np.stack([results['leye_origin'] / 1e2, results['leye_vector']]),
    ])
    data_dict.update(eyes_gaze=self.torch_fn(eyes_gaze))

    # Shape: (1, ), Mask for High Quality Eye Patches
    data_dict['reye_mask'] = self._create_eyes_mask(
      gaze=results['reye_vector'],
      pupil_vis=results['reye_pupil_vis'],
    )
    data_dict['leye_mask'] = self._create_eyes_mask(
      gaze=results['leye_vector'],
      pupil_vis=results['leye_pupil_vis'],
    )

    return data_dict


# Script Configuration
def _build_train_data_names(opts: argparse.Namespace):
  return [n for n in opts.data_name.split('+')]

def _build_test_data_names(opts: argparse.Namespace):
  if opts.test_data_name == 'none':
    opts.test_data_name = opts.data_name
  return [n for n in opts.test_data_name.split('+')]

def _build_train_loop_config(
  opts: argparse.Namespace,
  loop_cls: str,
  train_dataset_cfgs: dict,
  dataloader_kwargs: dict,
):
  assert loop_cls in ['EpochBasedTrainLoop']

  if opts.data_name == 'default+extended':
    dataset_cfg = dict(
      type='ConcatDataset',
      datasets=[
        train_dataset_cfgs.get('default'),
        train_dataset_cfgs.get('extended'),
      ],
    )
  else:
    dataset_cfg = train_dataset_cfgs.get(opts.data_name)

  dataloader = dict(dataset=dataset_cfg, **dataloader_kwargs)

  dataloader_config = dict(dataset=dataset_cfg, **dataloader_kwargs)
  loop_config = dict(type=loop_cls, dataloader=dataloader, max_epochs=opts.max_epochs)

  return dataloader_config, loop_config

def _build_single_dataset_loop_config(
  opts: argparse.Namespace,
  loop_cls: str,
  test_dataset_cfgs: dict,
  dataloader_kwargs: dict,
  evaluator_metrics: list,
) -> Tuple[Dict, Dict, Dict]:
  assert loop_cls in ['ValLoop', 'TestLoop']

  dataset_cfg = test_dataset_cfgs.get(opts.test_data_name)

  dataloader = dict(dataset=dataset_cfg, **dataloader_kwargs)
  evaluator = [
    dict(prefix=opts.test_data_name, **evaluator_metric)
    for evaluator_metric in evaluator_metrics
  ]

  dataloader_config = dict(dataset=dataset_cfg, **dataloader_kwargs)
  evaluator_config = [
    dict(prefix=opts.test_data_name, **evaluator_metric)
    for evaluator_metric in evaluator_metrics
  ]
  loop_config = dict(type=loop_cls, dataloader=dataloader, evaluator=evaluator)

  return dataloader_config, evaluator_config, loop_config

def _build_multi_dataset_loop_config(
  opts: argparse.Namespace,
  loop_cls: str,
  test_dataset_cfgs: dict,
  dataloader_kwargs: dict,
  evaluator_metrics: list,
) -> Tuple[Dict, Dict, Dict]:
  assert loop_cls in ['MultiSetValLoop', 'MultiSetTestLoop']

  dataloaders, evaluators = [], []
  for name, dataset_cfg in test_dataset_cfgs.items():
    dataloaders.append(dict(dataset=dataset_cfg, **dataloader_kwargs))
    evaluators.append([
      dict(prefix=name, **evaluator_metric)
      for evaluator_metric in evaluator_metrics
    ])

  dataloader_config = dict(dataset=test_dataset_cfgs, **dataloader_kwargs)
  evaluator_config = [
    dict(prefix=opts.test_data_name, **evaluator_metric)
    for evaluator_metric in evaluator_metrics
  ]
  loop_config = dict(type=loop_cls, dataloaders=dataloaders, evaluators=evaluators)

  return dataloader_config, evaluator_config, loop_config

def build_data_config_dict(opts: argparse.Namespace):
  assert opts.data_name in ['default', 'extended', 'default+extended'], f'Invalid data name: {opts.data_name}'
  assert 0.0 <= opts.max_gaze_angle <= 180.0, f'Invalid max gaze angle: {opts.max_gaze_angle}'
  assert 0.0 <= opts.pupil_vis_thres <= 1.0, f'Invalid pupil visibility threshold: {opts.pupil_vis_thres}'

  # Dataset config
  dataset_cfgs = ScriptEnv.load_config_dict('configs/dataset/ucas-synthgaze.py')
  dataset_cfgs['train'] = dict()  # Train data
  for name in _build_train_data_names(opts):
    dataset_cfgs['train'][name] = dataset_cfgs[f'train_{name}']

  dataset_cfgs['test'] = dict() # Test data
  for name in _build_test_data_names(opts):
    dataset_cfgs['test'][name] = dataset_cfgs[f'test_{name}']

  swap_file = ScriptEnv.resource_path('synthgaze/vertices-swap-pairs.json')
  pipeline = [
    dict(type='RandomCameraRotate3D', camera_roll=60, safe_margin=5),
    dict(type='RandomHFlip2D', swap_file=swap_file, p_hflip=0.5),
    dict(
      type='GetFaceAndBBox',
      p_noisy=0.8, face_bbox_shift=0.15, face_bbox_scale=(1.0, 1.5),
    ),
    dict(
      type='PrepareDataDict',
      max_gaze_angle=opts.max_gaze_angle,
      pupil_vis_thres=opts.pupil_vis_thres,
    ),
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
  ]
  for name in dataset_cfgs['train']:
    dataset_cfgs['train'][name].update(subset=opts.train_subset, pipeline=pipeline)

  pipeline = [
    dict(type='RandomCameraRotate3D', camera_roll=60, safe_margin=5),
    dict(type='GetFaceAndBBox', p_noisy=0.0),
    dict(
      type='PrepareDataDict',
      max_gaze_angle=80.0,
      pupil_vis_thres=0.5,
    ),
    dict(
      type='RandomImageAugmentation',
      image_data_key='face',
      normalize_kwargs=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ),
  ]
  for name in dataset_cfgs['test']:
    dataset_cfgs['test'][name].update(subset=opts.test_subset, pipeline=pipeline)

  # Metric config
  metric_cfgs = ScriptEnv.load_config_dict('configs/metric/gaze-dp.py')
  evaluator_metrics = [
    metric_cfgs['TdGazeNetGazeMetrics'],
    metric_cfgs['TdGazeNetMeshMetrics'],
  ]
  if opts.model_name == 'TdGazeNetPlus':
    evaluator_metrics.append(metric_cfgs['TdGazeNetPlusMetrics'])

  # Dataloader config
  train_dataloader_kwargs = dict(
    num_workers=opts.num_workers,
    batch_size=opts.batch_size,
    sampler=dataset_cfgs['train_data_sampler'],
    collate_fn=dict(type='default_collate'),
  )
  test_dataloader_kwargs = dict(
    num_workers=opts.num_workers,
    batch_size=opts.batch_size,
    sampler=dataset_cfgs['test_data_sampler'],
    collate_fn=dict(type='default_collate'),
  )

  # Loop config
  train_dataloader_cfg, train_loop_cfg = _build_train_loop_config(
    opts=opts, loop_cls='EpochBasedTrainLoop',
    train_dataset_cfgs=dataset_cfgs['train'],
    dataloader_kwargs=train_dataloader_kwargs,
  )

  if opts.test_data_name == 'default+extended':
    val_dataloader_cfg, val_evaluator_cfg, val_loop_cfg = (
      _build_multi_dataset_loop_config(
        opts=opts, loop_cls='MultiSetValLoop',
        test_dataset_cfgs=dataset_cfgs['test'],
        dataloader_kwargs=test_dataloader_kwargs,
        evaluator_metrics=evaluator_metrics,
      )
    )
    test_dataloader_cfg, test_evaluator_cfg, test_loop_cfg = (
      _build_multi_dataset_loop_config(
        opts=opts, loop_cls='MultiSetTestLoop',
        test_dataset_cfgs=dataset_cfgs['test'],
        dataloader_kwargs=test_dataloader_kwargs,
        evaluator_metrics=evaluator_metrics,
      )
    )
  else:
    val_dataloader_cfg, val_evaluator_cfg, val_loop_cfg = (
      _build_single_dataset_loop_config(
        opts=opts, loop_cls='ValLoop',
        test_dataset_cfgs=dataset_cfgs['test'],
        dataloader_kwargs=test_dataloader_kwargs,
        evaluator_metrics=evaluator_metrics,
      )
    )
    test_dataloader_cfg, test_evaluator_cfg, test_loop_cfg = (
      _build_single_dataset_loop_config(
        opts=opts, loop_cls='TestLoop',
        test_dataset_cfgs=dataset_cfgs['test'],
        dataloader_kwargs=test_dataloader_kwargs,
        evaluator_metrics=evaluator_metrics,
      )
    )

  # Data config
  if opts.mode == 'train':
    config_dict = dict(
      train_dataloader=train_dataloader_cfg,
      train_cfg=train_loop_cfg,
    )
    if not opts.skip_test:
      config_dict.update(
        val_dataloader=val_dataloader_cfg,
        val_cfg=val_loop_cfg,
        val_evaluator=val_evaluator_cfg,
      )

  if opts.mode == 'test':
    config_dict = dict(
      test_dataloader=test_dataloader_cfg,
      test_cfg=test_loop_cfg,
      test_evaluator=test_evaluator_cfg,
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
  model_cfgs = ScriptEnv.load_config_dict('configs/model/gaze-dp.py')
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
  parser = argparse.ArgumentParser(description='run script for TdGazeNet experiments.')

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
      'TdGazeNet', 'TdGazeNetPlus',
    ],
    help='select model name for current run.',
  )

  data_group = parser.add_argument_group(
    title='data options',
    description='data options for script.',
  )

  data_group.add_argument(
    '--data-name', required=True, choices=[
      'default', 'extended', 'default+extended',
    ],
    help='select data name for current run.',
  )
  data_group.add_argument(
    '--test-data-name', default='none', choices=[
      'none', 'default+extended',
      'default', 'extended',
    ],
    help='select test data name for current run.',
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
  data_group.add_argument(
    '--max-gaze-angle', type=float, default=80.0,
    help='maximum gaze angle, used for data validation (train).',
  )
  data_group.add_argument(
    '--pupil-vis-thres', type=float, default=0.5,
    help='pupil visibility threshold, used for data validation (train).',
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
