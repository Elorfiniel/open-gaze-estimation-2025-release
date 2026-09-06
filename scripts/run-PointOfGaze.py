from opengaze.engine.transform import BaseTransform
from opengaze.registry import TRANSFORMS
from opengaze.runtime.scripts import ScriptEnv, ScriptOptions
from opengaze.utils.image import scaled_crop

from mmengine.config import Config
from mmengine.runner import Runner
from PIL import Image

import argparse
import copy
import cv2
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF


def align_rotate(image: np.ndarray, landmarks: np.ndarray, theta: float):
  image_h, image_w, _ = image.shape

  image_l = 2 * max(image_h, image_w)
  image_a = int((image_l - image_w) / 2)
  image_b = int((image_l - image_h) / 2)

  image_pad = cv2.copyMakeBorder(
    image, image_b, image_b, image_a, image_a,
    cv2.BORDER_CONSTANT, None, value=(0, 0, 0),
  )
  M = cv2.getRotationMatrix2D(
    center=(image_w / 2 + image_a, image_h / 2 + image_b),
    angle=theta, scale=1.0,
  )
  image_rot = cv2.warpAffine(
    image_pad, M, (image_w + 2 * image_a, image_h + 2 * image_b),
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
  )
  ldmks_rot = np.dot(
    np.concatenate([
      landmarks + np.array([image_a, image_b]),
      np.ones(shape=(len(landmarks), 1)),
    ], axis=1),
    M.T,
  )

  return image_rot, ldmks_rot

def apply_bbox_noise(theta: float, image: np.ndarray, ldmks: np.ndarray, bbox: np.ndarray,
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

def rotate_vector(vector: np.ndarray, theta: float):
  radian = np.deg2rad(theta, dtype=np.float32)

  cos, sin = np.cos(radian), np.sin(radian)
  mat = np.array([[cos, -sin], [sin, cos]])

  return np.dot(mat, vector)


# Data Transformations
@TRANSFORMS.register_module()
class GetFaceAndBBoxes(BaseTransform):
  def __init__(self, p_face_noisy: float = 0.5,
               face_bbox_shift: float = 0.0,
               face_bbox_scale: tuple = (1.0, 1.0),
               face_bbox_rotate: float = 0.0,
               p_eyes_noisy: float = 0.5,
               eyes_bbox_shift: float = 0.0,
               eyes_bbox_scale: tuple = (1.0, 1.0),
               face_crop_size: int = 224,
               eyes_crop_expand: float = 1.6):

    self.p_face_noisy = p_face_noisy

    self.face_bbox_shift = face_bbox_shift
    self.face_bbox_scale = face_bbox_scale
    self.face_bbox_rotate = face_bbox_rotate

    self.p_eyes_noisy = p_eyes_noisy

    self.eyes_bbox_shift = eyes_bbox_shift
    self.eyes_bbox_scale = eyes_bbox_scale

    self.face_crop_size = face_crop_size
    self.eyes_crop_expand = eyes_crop_expand

  def _align_angle(self, landmarks: np.ndarray):
    ldmk_reye, ldmk_leye = landmarks[133], landmarks[362]

    norm = np.linalg.norm(ldmk_reye - ldmk_leye, ord=2)
    sin = (ldmk_reye[1] - ldmk_leye[1]) / norm
    theta = -np.rad2deg(np.arcsin(sin))

    return theta

  def _bbox_from_ldmk(self, landmarks: np.ndarray):
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    return np.array([x_min, y_min, x_max, y_max], dtype=int)

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

  def _eyes_bboxes(self, image: np.ndarray, bbox: np.ndarray,
                   ldmks: np.ndarray, eye_center: np.ndarray):
    x_min, y_min, x_max, y_max = self._bbox_from_ldmk(ldmks)
    eye_center_x, eye_center_y = eye_center

    crop_a = (x_max - x_min) * self.eyes_crop_expand

    x_min = eye_center_x - crop_a / 2
    x_max = eye_center_x + crop_a / 2
    y_min = eye_center_y - crop_a / 2
    y_max = eye_center_y + crop_a / 2

    # Apply eyes bbox noise
    if random.random() < self.p_eyes_noisy:
      shift_x = (2 * random.random() - 1) * (y_max - y_min) * self.eyes_bbox_shift
      shift_y = (2 * random.random() - 1) * (y_max - y_min) * self.eyes_bbox_shift
      scale = random.uniform(*self.eyes_bbox_scale)

      cx = (x_min + x_max) / 2 + shift_x
      cy = (y_min + y_max) / 2 + shift_y
      ca = (x_max - x_min) * scale

      x_min = cx - ca / 2
      x_max = cx + ca / 2
      y_min = cy - ca / 2
      y_max = cy + ca / 2

    eyes_bbox_crop = np.array([
      (x_min - bbox[0]) / (bbox[2] - bbox[0]),
      (y_min - bbox[1]) / (bbox[3] - bbox[1]),
      (x_max - bbox[0]) / (bbox[2] - bbox[0]),
      (y_max - bbox[1]) / (bbox[3] - bbox[1]),
    ], dtype=np.float32)

    image_h, image_w, _ = image.shape

    center = np.array([image_w / 2, image_h / 2])
    metric = max(image_h, image_w) / 2

    eyes_bbox = np.concatenate([
      np.array([(x_min + x_max) / 2, (y_min + y_max) / 2]) - center,
      np.array([x_max - x_min, y_max - y_min]),
    ], axis=0) / metric

    return eyes_bbox, eyes_bbox_crop

  def _reye_bboxes(self, image: np.ndarray, ldmks: np.ndarray, bbox: np.ndarray):
    reye_ldmks = ldmks[[
      160, 33, 161, 163, 133, 7, 173, 144,
      145, 246, 153, 154, 155, 157, 158, 159,
    ]]
    reye_center = ldmks[468]
    return self._eyes_bboxes(image, bbox, reye_ldmks, reye_center)

  def _leye_bboxes(self, image: np.ndarray, ldmks: np.ndarray, bbox: np.ndarray):
    leye_ldmks = ldmks[[
      384, 385, 386, 387, 388, 390, 263, 362,
      398, 466, 373, 374, 249, 380, 381, 382,
    ]]
    leye_center = ldmks[473]
    return self._eyes_bboxes(image, bbox, leye_ldmks, leye_center)

  def transform(self, results: dict):
    theta = self._align_angle(results['ldmk'])
    image_rot, ldmks_rot = align_rotate(results['image'], results['ldmk'], theta)

    x_min, y_min, x_max, y_max = self._bbox_from_ldmk(ldmks_rot)
    y_max = y_min + x_max - x_min # Initial face bbox to square
    bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    # Apply face bbox noise
    if random.random() < self.p_face_noisy:
      shift_x = (2 * random.random() - 1) * (y_max - y_min) * self.face_bbox_shift
      shift_y = (2 * random.random() - 1) * (y_max - y_min) * self.face_bbox_shift
      scale = random.uniform(*self.face_bbox_scale)
      rotate = (2 * random.random() - 1) * self.face_bbox_rotate

      theta, image_rot, ldmks_rot, bbox = apply_bbox_noise(
        theta, image_rot, ldmks_rot, bbox,
        shift_x, shift_y, scale, rotate,
      )

    data_dict = dict(gaze=rotate_vector(results['gaze'], -theta))

    # Prepare face crop and norm bbox
    face_crop, face_bbox = self._crop_face(image_rot, bbox)
    data_dict.update(face_crop=face_crop, face_bbox=face_bbox)

    # Prepare eyes bbox and crop bbox wrt face crop
    reye_bbox, reye_bbox_crop = self._reye_bboxes(image_rot, ldmks_rot, bbox)
    leye_bbox, leye_bbox_crop = self._leye_bboxes(image_rot, ldmks_rot, bbox)
    data_dict.update(
      reye_bbox=reye_bbox, reye_bbox_crop=reye_bbox_crop,
      leye_bbox=leye_bbox, leye_bbox_crop=leye_bbox_crop,
    )

    return data_dict


@TRANSFORMS.register_module()
class PrepareDataDictA(BaseTransform):
  def image_fn(self, image: Image):
    return TF.to_tensor(image)

  def torch_fn(self, x: np.ndarray):
    return torch.tensor(x, dtype=torch.float32)

  def transform(self, results: dict):
    face = cv2.cvtColor(results['face_crop'], cv2.COLOR_BGR2RGB)
    face = self.image_fn(Image.fromarray(face, mode='RGB'))
    gaze = self.torch_fn(results['gaze'])
    rect = self.torch_fn(
      np.concatenate([
        results['face_bbox'],
        results['reye_bbox'],
        results['leye_bbox'],
      ], axis=0)
    )
    data_dict = dict(face=face, gaze=gaze, rect=rect)

    data_dict.update(reye_bbox_crop=results['reye_bbox_crop'])
    data_dict.update(leye_bbox_crop=results['leye_bbox_crop'])

    return data_dict


@TRANSFORMS.register_module()
class PrepareDataDictB(BaseTransform):
  def transform(self, results: dict):
    data_dict = dict(face=results['face'], gaze=results['gaze'], rect=results['rect'])

    x_min, y_min, x_max, y_max = np.astype(results['reye_bbox_crop'] * 224, np.int32)
    reye_crop = TF.resize(
      TF.crop(results['face'], y_min, x_min, y_max - y_min, x_max - x_min),
      size=(112, 112),
      interpolation=TF.InterpolationMode.BICUBIC,
    )
    reye_crop = TF.hflip(reye_crop)

    x_min, y_min, x_max, y_max = np.astype(results['leye_bbox_crop'] * 224, np.int32)
    leye_crop = TF.resize(
      TF.crop(results['face'], y_min, x_min, y_max - y_min, x_max - x_min),
      size=(112, 112),
      interpolation=TF.InterpolationMode.BICUBIC,
    )

    data_dict.update(reye=reye_crop, leye=leye_crop)

    return data_dict


# Script Configuration
def build_model_config_dict(opts: argparse.Namespace):
  # Model config
  if opts.data_name in ['mit-gaze-capture', 'mit-gaze-capture-abalation-glasses']:
    model_cfgs = ScriptEnv.load_config_dict('configs/model/gaze-2d.py')
    model_config_dict = model_cfgs['AFFNet']

  return dict(model=model_config_dict)

def build_data_pipeline(data_name: str):
  # Data pipeline config
  if data_name == 'mit-gaze-capture':
    train_pipeline = [
      dict(
        type='GetFaceAndBBoxes',
        p_face_noisy=0.8,
        face_bbox_shift=0.1,
        face_bbox_scale=(1.0, 1.2),
        face_bbox_rotate=4.0,
        p_eyes_noisy=0.8,
        eyes_bbox_shift=0.1,
        eyes_bbox_scale=(1.0, 1.2),
      ),
      dict(type='PrepareDataDictA'),
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
      dict(type='PrepareDataDictB'),
    ]
    test_pipeline = [
      dict(type='GetFaceAndBBoxes', p_face_noisy=0.0, p_eyes_noisy=0.0),
      dict(type='PrepareDataDictA'),
      dict(
        type='RandomImageAugmentation',
        image_data_key='face',
        normalize_kwargs=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ),
      dict(type='PrepareDataDictB'),
    ]

  return train_pipeline, test_pipeline

def build_data_config_dict(opts: argparse.Namespace):
  # Dataset config
  if opts.data_name == 'mit-gaze-capture':
    dataset_cfgs = ScriptEnv.load_config_dict('configs/dataset/mit-gaze-capture.py')
    train_pipeline, test_pipeline = build_data_pipeline('mit-gaze-capture')

    dataset_cfgs['train'].update(subset=opts.train_subset, pipeline=train_pipeline)
    dataset_cfgs['train']['meta_filter_configs'].update(drop=[False], glasses=[False])

    dataset_cfgs['test'].update(subset=opts.test_subset, pipeline=test_pipeline)
    dataset_cfgs['test']['meta_filter_configs'].update(drop=[False], glasses=[False])

    # Metric config
    metric_cfgs = ScriptEnv.load_config_dict('configs/metric/gaze-2d.py')
    evaluator_metrics = metric_cfgs['DistanceError']

  if opts.data_name == 'mit-gaze-capture-abalation-glasses':
    assert opts.abalation_glasses_device in ['iPhone 6', 'iPhone 5S']

    dataset_cfgs = ScriptEnv.load_config_dict('configs/dataset/mit-gaze-capture.py')
    train_pipeline, test_pipeline = build_data_pipeline('mit-gaze-capture')

    train_a_size = opts.abalation_glasses_n_total_samples - opts.abalation_glasses_n_glasses_samples
    train_b_size = opts.abalation_glasses_n_glasses_samples
    test_size = opts.abalation_glasses_test_size

    train_dataset_a_cfg = copy.deepcopy(dataset_cfgs['train'])
    train_dataset_a_cfg.update(subset=train_a_size, pipeline=train_pipeline)
    train_dataset_a_cfg['meta_filter_configs'].update(
      split=['train'], drop=[False], glasses=[False],
      device=[opts.abalation_glasses_device],
    )

    train_dataset_b_cfg = copy.deepcopy(dataset_cfgs['train'])
    train_dataset_b_cfg.update(subset=train_b_size, pipeline=train_pipeline)
    train_dataset_b_cfg['meta_filter_configs'].update(
      split=['train', 'val', 'test'], drop=[False], glasses=[True],
      device=[opts.abalation_glasses_device],
    )

    test_dataset_cfg = copy.deepcopy(dataset_cfgs['test'])
    test_dataset_cfg.update(subset=test_size, pipeline=test_pipeline)
    test_dataset_cfg['meta_filter_configs'].update(
      split=['val', 'test'], drop=[False], glasses=[False],
      device=[opts.abalation_glasses_device],
    )

    dataset_cfgs['train'] = dict(
      type='ConcatDataset',
      datasets=[train_dataset_a_cfg, train_dataset_b_cfg],
    )
    dataset_cfgs['test'] = test_dataset_cfg

    # Metric config
    metric_cfgs = ScriptEnv.load_config_dict('configs/metric/gaze-2d.py')
    evaluator_metrics = metric_cfgs['DistanceError']

  # Dataloader config
  dataloader_kwargs = dict(
    num_workers=opts.num_workers,
    batch_size=opts.batch_size,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
  )
  if opts.mode == 'train':
    config_dict = dict(
      train_dataloader=dict(dataset=dataset_cfgs['train'], **dataloader_kwargs),
      train_cfg=dict(by_epoch=True, max_epochs=opts.max_epochs),
    )
    if not opts.skip_test:
      config_dict.update(
        val_dataloader=dict(dataset=dataset_cfgs['test'], **dataloader_kwargs),
        val_cfg=dict(type='ValLoop'),
        val_evaluator=evaluator_metrics,
      )

  if opts.mode == 'test':
    config_dict = dict(
      test_dataloader=dict(dataset=dataset_cfgs['test'], **dataloader_kwargs),
      test_cfg=dict(type='TestLoop'),
      test_evaluator=evaluator_metrics,
    )

  return config_dict

def build_optim_dict(opts: argparse.Namespace):
  optimizer = dict(type='AdamW', lr=opts.base_lr, betas=(0.90, 0.95), weight_decay=5e-3)
  optim_wrapper_cls = 'AmpOptimWrapper' if opts.mixed_precision else 'OptimWrapper'
  optim_wrapper = dict(type=optim_wrapper_cls, optimizer=optimizer)

  param_scheduler = []

  if 0 <= opts.warm_up_epoch <= opts.max_epochs:
    param_scheduler.append(
      dict(
        type='LinearLR', by_epoch=True,
        start_factor=opts.warm_up_ratio,
        end_factor=1.0,
        begin=0, end=opts.warm_up_epoch,
        convert_to_iter_based=True,
      ),
    )

  if 0 <= opts.cool_down_epoch <= opts.max_epochs:
    param_scheduler.append(
      dict(
        type='LinearLR', by_epoch=True,
        start_factor=1.0,
        end_factor=opts.cool_down_ratio,
        begin=opts.cool_down_epoch,
        end=opts.max_epochs,
        convert_to_iter_based=True,
      ),
    )

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
  parser = argparse.ArgumentParser(description='run script for PointOfGaze experiments.')

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
      'mit-gaze-capture',
      'mit-gaze-capture-abalation-glasses',
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
    '--warm-up-epoch', type=float, default=1.00,
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

  abalation_group = parser.add_argument_group(
    title='abalation options',
    description='options for abalation study.',
  )

  abalation_group.add_argument(
    '--abalation-glasses-device', type=str, default='iPhone 6',
    choices=['iPhone 6', 'iPhone 5S'],
    help='select device type for glasses abalation study.',
  )
  abalation_group.add_argument(
    '--abalation-glasses-n-total-samples', type=int, default=200000,
    help='number of total samples for glasses abalation study.',
  )
  abalation_group.add_argument(
    '--abalation-glasses-n-glasses-samples', type=int, default=0,
    help='ratio of glasses samples for glasses abalation study.',
  )
  abalation_group.add_argument(
    '--abalation-glasses-test-size', type=int, default=20000,
    help='number of test samples for glasses abalation study.',
  )

  opts, _ = ScriptOptions(parser).parse_args()

  main_procedure(opts)
