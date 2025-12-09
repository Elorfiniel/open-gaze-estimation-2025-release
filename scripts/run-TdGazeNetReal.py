from opengaze.utils.gc import FaceAlignment
from opengaze.utils.image import scaled_crop
from opengaze.engine.transform import BaseTransform
from opengaze.registry import TRANSFORMS
from opengaze.runtime.scripts import ScriptEnv, ScriptOptions

from mmengine.config import Config
from mmengine.runner import Runner
from PIL import Image

import argparse
import cv2
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF


# Data Transformations
@TRANSFORMS.register_module()
class GetFaceAndBBox(FaceAlignment, BaseTransform):
  def __init__(self, p_noisy: float = 0.5,
               face_bbox_shift: float = 0.0,
               face_bbox_scale: tuple = (1.0, 1.0),
               face_bbox_rotate: float = 0.0,
               face_crop_size: int = 224, **kwargs):

    super(GetFaceAndBBox, self).__init__(**kwargs)

    self.p_noisy = p_noisy

    self.face_bbox_shift = face_bbox_shift
    self.face_bbox_scale = face_bbox_scale
    self.face_bbox_rotate = face_bbox_rotate

    self.face_crop_size = face_crop_size

  def _apply_noise(self, theta: float, image: np.ndarray, ldmks: np.ndarray, bbox: np.ndarray,
                   shift_x: float, shift_y: float, scale: float, rotate: float):
    image_h, image_w, _ = image.shape

    M = cv2.getRotationMatrix2D(
      center=(image_w / 2, image_h / 2),
      angle=rotate, scale=1.0,
    )
    image_rot = cv2.warpAffine(
      image, M, (image_w, image_h),
      flags=cv2.INTER_CUBIC,
      borderMode=cv2.BORDER_CONSTANT,
    )
    ldmks_rot = np.dot(
      np.concatenate([
        ldmks,
        np.ones(shape=(len(ldmks), 1)),
      ], axis=1),
      M.T,
    )

    noisy_theta = theta + rotate

    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    ca = (bbox[2] - bbox[0]) * scale

    cx, cy = np.dot(M, np.array([cx, cy, 1.0])) + np.array([shift_x, shift_y])

    noisy_bbox = np.array([
      cx - ca / 2, cy - ca / 2,
      cx + ca / 2, cy + ca / 2,
    ], dtype=np.float32)

    return noisy_theta, image_rot, ldmks_rot, noisy_bbox

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

  def _rotate_vector(self, vector: np.ndarray, theta: float):
    radian = np.deg2rad(theta, dtype=np.float32)

    cos, sin = np.cos(radian), np.sin(radian)
    mat = np.array([[cos, -sin], [sin, cos]])

    return np.dot(mat, vector)

  def transform(self, results: dict):
    theta = self._align_angle(results['ldmk'])
    image_rot, ldmks_rot = self._align_rotate(
      results['image'], results['ldmk'], theta,
    )

    x_min, y_min, x_max, y_max = self._bbox_from_ldmk(ldmks_rot)
    y_max = y_min + x_max - x_min # Initial face bbox to square
    bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    # Apply face bbox noise
    if random.random() < self.p_noisy:
      shift_x = (2 * random.random() - 1) * (y_max - y_min) * self.face_bbox_shift
      shift_y = (2 * random.random() - 1) * (y_max - y_min) * self.face_bbox_shift
      scale = random.uniform(*self.face_bbox_scale)
      rotate = (2 * random.random() - 1) * self.face_bbox_rotate

      theta, image_rot, ldmks_rot, (
        x_min, y_min, x_max, y_max,
      ) = self._apply_noise(
        theta, image_rot, ldmks_rot, bbox,
        shift_x, shift_y, scale, rotate,
      )

    results['gaze'] = self._rotate_vector(results['gaze'], -theta)

    # Prepare face crop and norm bbox
    face_crop, face_bbox = self._crop_face(image_rot, bbox)
    results.update(face_crop=face_crop, face_bbox=face_bbox)

    return results


@TRANSFORMS.register_module()
class PrepareDataDict(BaseTransform):
  def __init__(self, device_mapping: dict):
    self.device_mapping = device_mapping

  def image_fn(self, image: Image):
    return TF.to_tensor(image)

  def torch_fn(self, x: np.ndarray):
    return torch.tensor(x, dtype=torch.float32)

  def transform(self, results: dict):
    if self.device_mapping and 'device' in results:
      data_dict = dict(device=self.device_mapping[results['device']])
    elif not self.device_mapping:
      data_dict = dict(device=0)
    else:
      raise RuntimeError(f'No device in results, or empty device mapping.')

    # Shape: (3, 224, 224), ImageNet Normalization
    face = cv2.cvtColor(results['face_crop'], cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face, mode='RGB')
    data_dict.update(face=self.image_fn(face))

    # Shape: (4, ), Standardized BBox: (p - center) / metric
    data_dict.update(bbox=self.torch_fn(results['face_bbox']))

    # Convert PoG annotations in GazeCapture dataset to the prediction
    # space of TdGazeNet (Coordinate System + Unit Length)
    gaze = self.torch_fn(-results['gaze'] / 10)
    # Shape: (2, ), Point of Gaze, Standard Unit (10cm)
    data_dict.update(reye_gaze=gaze, leye_gaze=gaze)

    return data_dict


# Script Configuration
def build_model_config_dict(opts: argparse.Namespace):
  # Model config
  model_cfgs = ScriptEnv.load_config_dict('configs/model/gaze-dp.py')
  model_config_dict = model_cfgs['TdGazeNetReal']

  if opts.data_name == 'mit-gaze-capture':
    model_config_dict['model_cfg']['n_adapters'] = 15

  return dict(model=model_config_dict)

def build_data_config_dict(opts: argparse.Namespace):
  # Dataset config
  if opts.data_name == 'mit-gaze-capture':
    dataset_cfgs = ScriptEnv.load_config_dict('configs/dataset/point-of-gaze/mit-gaze-capture.py')
    train_dataset = dict(**dataset_cfgs['train'], return_meta_keys=['device'])
    test_dataset = dict(**dataset_cfgs['test'], return_meta_keys=['device'])

    # Skip subjects with insufficient samples or with glasses on
    train_dataset['meta_filter_configs'].update(drop=[False], glasses=[False])
    test_dataset['meta_filter_configs'].update(drop=[False], glasses=[False])

    DEVICES = [
      'iPad 2', 'iPad 3', 'iPad 4', 'iPad Air', 'iPad Air 2', 'iPad Mini', 'iPad Pro',
      'iPhone 4S', 'iPhone 5', 'iPhone 5C', 'iPhone 5S',
      'iPhone 6', 'iPhone 6 Plus', 'iPhone 6s', 'iPhone 6s Plus',
    ]

    # Metric config
    metric_cfgs = ScriptEnv.load_config_dict('configs/metric/gaze-2d.py')
    evaluator_metrics = metric_cfgs['DistanceError']

  if opts.data_name == 'single-device-generic':
    dataset_cfgs = ScriptEnv.load_config_dict('configs/dataset/point-of-gaze/single-device-generic.py')
    train_dataset, test_dataset = dataset_cfgs['train'], dataset_cfgs['test']

    # Skip subjects with insufficient samples or with glasses on
    train_dataset['meta_filter_configs'].update(drop=[False], glasses=[False])
    test_dataset['meta_filter_configs'].update(drop=[False], glasses=[False])

    DEVICES = []  # Leave empty to use "Unknown Device" as default

    # Metric config
    metric_cfgs = ScriptEnv.load_config_dict('configs/metric/gaze-2d.py')
    evaluator_metrics = metric_cfgs['AFFNetDMetrics']

  device_mapping = {x:it for it, x in enumerate(DEVICES)}

  pipeline = [
    dict(
      type='GetFaceAndBBox',
      p_noisy=0.8,
      face_bbox_shift=0.15,
      face_bbox_scale=(1.0, 1.2),
      face_bbox_rotate=30.0,
    ),
    dict(type='PrepareDataDict', device_mapping=device_mapping),
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
  train_dataset.update(subset=opts.train_subset, pipeline=pipeline)

  pipeline = [
    dict(type='GetFaceAndBBox', p_noisy=0.0),
    dict(type='PrepareDataDict', device_mapping=device_mapping),
    dict(
      type='RandomImageAugmentation',
      image_data_key='face',
      normalize_kwargs=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ),
  ]
  test_dataset.update(subset=opts.test_subset, pipeline=pipeline)

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
      type='StepLR', by_epoch=True,
      begin=opts.step_begin, end=opts.step_end,
      step_size=opts.step_size, gamma=opts.step_gamma,
    ),
  ]

  return dict(optim_wrapper=optim_wrapper, param_scheduler=param_scheduler)

def build_config(opts: argparse.Namespace):
  # Default runtime config
  config = ScriptEnv.load_config_dict('configs/default-runtime.py')

  # Model config
  model_config_dict = build_model_config_dict(opts)
  config.update(model_config_dict)

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
  parser = argparse.ArgumentParser(description='run script for TdGazeNetReal experiments.')

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

  data_group = parser.add_argument_group(
    title='data options',
    description='data options for script.',
  )

  data_group.add_argument(
    '--data-name', required=True, choices=[
      'mit-gaze-capture', 'single-device-generic',
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
    '--base-lr', type=float, default=1e-2,
    help='constant learning rate for adaptation.',
  )
  config_group.add_argument(
    '--max-epochs', type=int, default=2,
    help='max number of epochs for adaptation.',
  )
  config_group.add_argument(
    '--step-begin', type=int, default=0,
    help='begin epoch for learning rate scheduler.',
  )
  config_group.add_argument(
    '--step-end', type=int, default=2,
    help='end epoch for learning rate scheduler.',
  )
  config_group.add_argument(
    '--step-size', type=int, default=1,
    help='step size for learning rate scheduler.',
  )
  config_group.add_argument(
    '--step-gamma', type=float, default=0.5,
    help='multiplicative factor for learning rate decay.',
  )
  config_group.add_argument(
    '--mixed-precision', action='store_true', default=False,
    help='enable mixed precision training.',
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
