from opengaze.runtime.camera import VideoCaptureBuilder, CaptureHandler
from opengaze.runtime.camera import use_state
from opengaze.runtime.time import TimeViaEMA
from opengaze.utils.gc import FaceLandmarks
from opengaze.utils.image import scaled_crop
from opengaze.model.gaze_dp import TdGazeNet

from PIL import Image
from typing import Callable, List

import argparse
import cv2
import collections
import json
import math
import numpy as np
import os.path as osp
import time
import torch
import torchvision.transforms.functional as TF


# Utils for this demo script, not intended for other scripts
def _load_json_file(json_file: str, **kwargs):
  with open(json_file, 'r', encoding='utf-8') as file:
    json_data = json.load(file, **kwargs)
  return json_data

def _load_demo_data(data_path: str = ''):
  '''Load configuration from JSON file, defaults to same-named .json file.'''

  if not data_path:
    data_name = osp.basename(__file__).replace('.py', '.json')
    data_path = osp.join(osp.dirname(__file__), data_name)

  data_path = osp.abspath(data_path)
  if not osp.isfile(data_path):
    raise FileNotFoundError(f"No demo data found: '{data_path}'.")

  demo_data = _load_json_file(data_path)

  return demo_data

def _capture_builder_kwargs(demo_data: dict):
  '''Extract camera settings from demo data for VideoCaptureBuilder.'''

  return dict(
    capture_id=demo_data['camera']['capture_id'],
    resolution=[
      demo_data['camera']['frame_h'],
      demo_data['camera']['frame_w'],
    ],
  )


# Utility Functions for Animation
def _lerp_color(color1: np.ndarray, color2: np.ndarray, t: float) -> tuple:
  '''Linear interpolation between two BGR colors.'''

  t = np.clip(t, 0.0, 1.0)
  result = (1.0 - t) * color1 + t * color2
  return tuple(result.astype(np.int32).tolist())


def _ease_in_out_quad(t: float) -> float:
  '''Quadratic easing in-out for smooth acceleration/deceleration.'''

  t = np.clip(t, 0.0, 1.0)
  if t < 0.5:
    return 2 * t * t
  else:
    return -1 + (4 - 2 * t) * t


def _random_point_in_ellipse(
  center: np.ndarray,
  axes: np.ndarray,
  angle: float,
  rng: np.random.Generator,
) -> np.ndarray:
  '''Generate a random point inside a rotated ellipse.'''

  # Random point in unit circle
  r = rng.uniform(0, 1)
  theta = rng.uniform(0, 2 * math.pi)
  x = r * math.cos(theta)
  y = r * math.sin(theta)

  # Scale by ellipse axes
  a, b = axes[0], axes[1]
  x *= a
  y *= b

  # Rotate
  rad = math.radians(angle)
  cos_a, sin_a = math.cos(rad), math.sin(rad)
  x_rot = x * cos_a - y * sin_a
  y_rot = x * sin_a + y * cos_a

  # Translate to center
  return center + np.array([x_rot, y_rot], dtype=np.float32)


# Particle System
class Particle:
  '''Single explosion particle with physics-based motion.'''

  def __init__(
    self,
    center: np.ndarray,
    velocity: np.ndarray,
    color: tuple,
    lifetime: float,
    gravity: float,
  ):
    self.pos = center.copy()
    self.vel = velocity  # [vx, vy] in px/s
    self.color = color
    self.lifetime = lifetime
    self.max_lifetime = lifetime
    self.gravity = gravity

  def update(self, dt: float) -> bool:
    '''Update particle position and lifetime. Returns False when particle dies.'''

    self.pos += self.vel * dt
    self.vel[1] += self.gravity * dt  # Apply gravity
    self.lifetime -= dt
    return self.lifetime > 0

  def draw(self, canvas: np.ndarray):
    '''Draw particle with sharp, clean fading effect.'''

    alpha = self.lifetime / self.max_lifetime
    if alpha <= 0:
      return

    # Scale color by alpha for sharp fading effect
    b, g, r = self.color
    color = (int(b * alpha), int(g * alpha), int(r * alpha))

    # Smaller, sharper particles
    radius = max(1, int(3 * alpha))
    cv2.circle(canvas, self.pos.astype(np.int32), radius, color, -1, cv2.LINE_AA)


# Target Class
class Target:
  '''Animated target ellipse with proximity-based visual effects.'''

  # Color constants (BGR format) - Modern purple-pink-red gradient
  # Far: Purple (#8B5CF6) -> Near: Pink (#EC4899) -> Close: Red (#DA0037)
  COOL_COLOR = np.array([246, 92, 199], dtype=np.float32)  # Purple-ish pink
  WARM_COLOR = np.array([55, 0, 218], dtype=np.float32)  # Modern red

  def __init__(
    self,
    screen_size: tuple,
    config: dict,
    rng: np.random.Generator,
  ):
    self.screen_h, self.screen_w = screen_size
    self.config = config
    self.rng = rng

    # State
    self.center = np.zeros(2, dtype=np.float32)
    self.angle = 0.0
    self.distance_to_pog = float('inf')
    self.proximity = 0.0
    self.is_exploded = False
    self.respawn_timer = 0.0
    self.triggered = False  # Track if this target instance has been triggered
    self.dwell_timer = 0.0  # Accumulate fixation time
    self.must_exit = False  # Require PoG to leave exit radius after trigger
    self.animation_time = 0.0  # For easing animation

    # Spawn initial target
    self.respawn()

  def respawn(self):
    '''Move target to a new random position.'''

    margin = self.config['target']['min_size'] * 2
    self.center = self.rng.uniform(
      low=[margin, margin],
      high=[self.screen_w - margin, self.screen_h - margin],
    ).astype(np.float32)
    self.angle = 0.0
    self.is_exploded = False
    self.respawn_timer = 0.0
    self.triggered = False
    self.dwell_timer = 0.0
    self.must_exit = False
    self.animation_time = 0.0

  def update(self, pog_pos: np.ndarray, dt: float):
    '''Update target state based on point of gaze position.'''

    # Calculate distance to PoG
    diff = self.center - pog_pos
    self.distance_to_pog = float(np.linalg.norm(diff))

    # Calculate proximity (1.0 = at target, 0.0 = far)
    threshold = self.config['target']['proximity_threshold']
    self.proximity = 1.0 - min(self.distance_to_pog / threshold, 1.0)

    # Update animation time for easing
    self.animation_time += dt

    # Calculate rotation speed with quadratic easing for smooth acceleration/deceleration
    # Create a smooth oscillating effect using sine wave
    oscillation = (math.sin(self.animation_time * 2.0) + 1.0) / 2.0  # 0 to 1
    eased_speed = _ease_in_out_quad(oscillation)

    # Base speed increases with proximity, modulated by easing
    base_speed = self.config['animation']['base_rotation_speed']
    speed_multiplier = self.config['animation']['max_rotation_speed_add']
    rotation_speed = base_speed + eased_speed * speed_multiplier * (0.5 + 0.5 * self.proximity)

    self.angle += rotation_speed * dt
    self.angle %= 360.0

    # State machine: handle trigger/exit logic
    if not self.triggered:
      # Not yet triggered - accumulate dwell time when fixating
      if self.is_fixating:
        self.dwell_timer += dt
        if self.dwell_timer >= self.config['target']['dwell_time']:
          self.triggered = True
          self.must_exit = True
      else:
        # Reset dwell timer if not fixating
        self.dwell_timer = max(0.0, self.dwell_timer - dt)
    elif self.must_exit:
      # Already triggered - check if PoG has exited the area
      exit_radius = self.config['target']['exit_radius']
      if self.distance_to_pog > exit_radius:
        self.must_exit = False  # PoG has exited, can now show new target

    # Update respawn timer if exploded
    if self.is_exploded:
      self.respawn_timer -= dt
      if self.respawn_timer <= 0:
        self.respawn()

  def explode(self) -> List[Particle]:
    '''Create explosion particles with modern color palette.'''

    self.is_exploded = True
    self.respawn_timer = self.config['target']['respawn_delay']

    particles = []
    particle_config = self.config['particles']

    # Modern particle colors (purple-pink-red gradient)
    particle_colors = [
      (246, 92, 199),  # Purple-pink
      (236, 72, 153),  # Pink
      (55, 0, 218),  # Modern red
      (219, 39, 119),  # Red-pink
      (255, 255, 255),  # White accent
    ]

    for _ in range(particle_config['count']):
      # Random velocity direction
      angle = self.rng.uniform(0, 2 * math.pi)
      speed = self.rng.uniform(
        particle_config['speed'] * 0.5,
        particle_config['speed'] * 1.5,
      )
      velocity = np.array(
        [
          math.cos(angle) * speed,
          math.sin(angle) * speed,
        ],
        dtype=np.float32,
      )

      # Start position at random point within target ellipse
      axes = self.current_axes
      start_pos = _random_point_in_ellipse(
        self.center,
        axes,
        self.angle,
        self.rng,
      )

      # Random color from modern palette
      color = self.rng.choice(particle_colors)

      p = Particle(
        center=start_pos,
        velocity=velocity,
        color=color,
        lifetime=particle_config['lifetime'],
        gravity=particle_config['gravity'],
      )
      particles.append(p)

    return particles

  @property
  def current_size(self) -> float:
    '''Current size based on proximity - shrinks when approaching.'''

    min_size = self.config['target']['min_size']
    max_size = self.config['target']['max_size']
    # Invert: larger when far, smaller when close
    return max_size - self.proximity * (max_size - min_size)

  @property
  def current_axes(self) -> np.ndarray:
    '''Current ellipse axes (width, height radii).'''

    size = self.current_size
    return np.array([size * 1.5, size], dtype=np.float32)

  @property
  def current_color(self) -> tuple:
    '''Current color based on proximity.'''

    return _lerp_color(self.COOL_COLOR, self.WARM_COLOR, self.proximity)

  @property
  def is_fixating(self) -> bool:
    '''Check if PoG is within fixation radius (larger than trigger zone).'''

    fixation_radius = self.config['target']['fixation_radius']
    return self.distance_to_pog < fixation_radius

  @property
  def is_fixated(self) -> bool:
    '''Check if target should be triggered (dwell time completed).'''

    return self.triggered and not self.is_exploded

  @property
  def dwell_progress(self) -> float:
    '''Get dwell timer progress (0.0 to 1.0).'''

    dwell_time = self.config['target']['dwell_time']
    return min(self.dwell_timer / dwell_time, 1.0)

  def draw(self, canvas: np.ndarray):
    '''Draw the target ellipse with sharp, modern styling.'''

    if self.is_exploded:
      return

    axes = self.current_axes
    cx, cy = self.center.astype(np.int32)

    # Draw very subtle glow effect (minimal for sharpness)
    glow_axes = axes * 1.2
    glow_overlay = canvas.copy()
    cv2.ellipse(
      glow_overlay,
      (cx, cy),
      glow_axes.astype(np.int32),
      self.angle,
      0,
      360,
      (246, 92, 199),  # Purple-pink glow
      -1,
      cv2.LINE_AA,
    )
    cv2.addWeighted(
      glow_overlay,
      0.08,
      canvas,
      0.92,
      0,
      canvas,
    )

    # Draw main white target ellipse (sharp, clean fill)
    cv2.ellipse(
      canvas,
      (cx, cy),
      axes.astype(np.int32),
      self.angle,
      0,
      360,
      (255, 255, 255),  # Pure white
      -1,  # Filled
      cv2.LINE_AA,
    )

    # Draw sharp, thin outline for better definition
    cv2.ellipse(
      canvas,
      (cx, cy),
      axes.astype(np.int32),
      self.angle,
      0,
      360,
      (200, 200, 200),  # Subtle gray outline
      1,  # Thin, sharp
      cv2.LINE_AA,
    )

    # Draw dwell progress indicator (sharp, thin lines)
    if self.is_fixating and not self.triggered:
      progress_radius = int(self.current_size * 2.5)
      progress_angle = int(360 * self.dwell_progress)

      # Draw background circle (sharp, thin)
      cv2.ellipse(
        canvas,
        (cx, cy),
        (progress_radius, progress_radius),
        0,
        0,
        360,
        (68, 68, 68),  # Gray
        1,
        cv2.LINE_AA,
      )

      # Draw progress arc (sharp, thin, purple-pink-red gradient)
      progress_color = _lerp_color(
        np.array([246, 92, 199]),  # Purple-pink (start)
        np.array([55, 0, 218]),  # Modern red (complete)
        self.dwell_progress,
      )
      cv2.ellipse(
        canvas,
        (cx, cy),
        (progress_radius, progress_radius),
        0,
        0,
        progress_angle,
        progress_color,
        2,
        cv2.LINE_AA,
      )


# Data Transformations, Model Inference and Result Display
class FrameConsumer:
  def __call__(self, src_image: np.ndarray, set_exit_cond: Callable, model: TdGazeNet):
    '''Process frame and update exit condition.'''

    result_dict = self.process(src_image, model)
    exit_cond = self.display(result_dict)
    set_exit_cond(exit_cond)
    return result_dict, exit_cond

  def __init__(self, demo_data: dict, landmarker: FaceLandmarks, device: torch.device):
    '''Initialize FrameConsumer with demo data, landmarker, and device.'''

    self.name = 'TdGazeNet Animation Demo'
    self.name = 'TdGazeNet Animation Demo'
    self.demo_data = demo_data
    self.landmarker = landmarker
    self.device = device

    self.time = TimeViaEMA(alpha=0.1)

    # Initialize target animation components
    self.screen_h = demo_data['screen']['h_px']
    self.screen_w = demo_data['screen']['w_px']
    self.rng = np.random.default_rng()
    self.target = Target((self.screen_h, self.screen_w), demo_data, self.rng)
    self.particles: List[Particle] = []
    self.last_time = time.time()
    # Initialize PoG to center
    self.pog_pos = np.array([self.screen_w / 2, self.screen_h / 2], dtype=np.float32)

  def __enter__(self):
    # Manage OpenCV resources
    cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Manage face landmarker
    self.landmarker.create()

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    cv2.destroyWindow(self.name)
    self.landmarker.destroy()

  @property
  def norm_intrinsic(self):
    return np.array(self.demo_data['intrinsic'], dtype=np.float32)

  @property
  def real_intrinsic(self):
    return np.array(self.demo_data['camera']['intrinsic'], dtype=np.float32)

  @property
  def real_extrinsic(self):
    return np.array(self.demo_data['camera']['extrinsic'], dtype=np.float32)

  @property
  def real_dist_coef(self):
    return np.array(self.demo_data['camera']['distortion'], dtype=np.float32)

  @property
  def M_real2norm(self):
    if not hasattr(self, '_M_real2norm'):
      self._M_real2norm = np.dot(self.norm_intrinsic, np.linalg.inv(self.real_intrinsic))
    return self._M_real2norm

  @property
  def M_norm2real(self):
    if not hasattr(self, '_M_norm2real'):
      self._M_norm2real = np.dot(self.real_intrinsic, np.linalg.inv(self.norm_intrinsic))
    return self._M_norm2real

  @property
  def norm_center(self):
    u_x = self.demo_data['intrinsic'][0][2]
    u_y = self.demo_data['intrinsic'][1][2]
    return np.array([u_x, u_y], dtype=np.float32)

  @property
  def norm_metric(self):
    u_x = self.demo_data['intrinsic'][0][2]
    u_y = self.demo_data['intrinsic'][1][2]
    return (u_x + u_y) / 2.0

  @property
  def screen_hw_px(self):
    return (self.demo_data['screen']['h_px'], self.demo_data['screen']['w_px'])

  @property
  def screen_hw_mm(self):
    return (self.demo_data['screen']['h_mm'], self.demo_data['screen']['w_mm'])

  @property
  def image_size(self):
    return self.demo_data['pre_process']['image_size']

  @property
  def bbox_scale(self):
    return self.demo_data['pre_process']['bbox_scale']

  @property
  def norm_params(self):
    return dict(
      mean=self.demo_data['pre_process']['norm_mean'],
      std=self.demo_data['pre_process']['norm_std'],
    )

  @property
  def gaze_source(self):
    return self.demo_data['post_process']['gaze_source']

  @property
  def epsilon(self):
    return self.demo_data['post_process']['epsilon']

  @property
  def disp_margin(self):
    return self.demo_data['post_process']['disp_margin']

  def _bbox_from_ldmk(self, landmarks: np.ndarray):
    '''Extract bounding box from facial landmarks.'''

    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)

    bbox_cx = (x_min + x_max) / 2.0
    bbox_cy = (y_min + y_max) / 2.0
    bbox_ca = math.sqrt((x_max - x_min) * (y_max - y_min))
    bbox_ca = self.bbox_scale * bbox_ca

    x_min = bbox_cx - bbox_ca / 2.0
    y_min = bbox_cy - bbox_ca / 2.0
    x_max = bbox_cx + bbox_ca / 2.0
    y_max = bbox_cy + bbox_ca / 2.0

    bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    return bbox

  def _norm_bbox(self, bbox: np.ndarray):
    '''Normalize bounding box from real camera to normalized camera space.'''

    x_min, y_min, x_max, y_max = bbox

    homo_coords = np.array([
      [x_min, y_min, 1.0],
      [x_max, y_max, 1.0],
    ], dtype=np.float32)
    pt_1, pt_2 = np.dot(homo_coords, self.M_real2norm.T)[:, :2]
    [x_min, y_min], [x_max, y_max] = pt_1, pt_2

    norm_bbox = np.concatenate([
      np.array([(x_min + x_max) / 2, (y_min + y_max) / 2]) - self.norm_center,
      np.array([x_max - x_min, y_max - y_min]),
    ], axis=0) / self.norm_metric

    return norm_bbox

  def _model_data_dict(self, face_crop: np.ndarray, face_bbox: np.ndarray):
    '''Prepare face crop and normalized bbox for model input.'''

    face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face, mode='RGB')
    face = TF.to_tensor(face).unsqueeze(0)
    face = TF.normalize(face, **self.norm_params)
    bbox = torch.tensor(face_bbox, dtype=torch.float32).unsqueeze(0)
    return dict(face=face, bbox=bbox)

  def _model_inference(self, model: TdGazeNet, data_dict: dict):
    '''Run model inference and extract outputs.'''

    data_dict = {k:v.to(self.device) for k, v in data_dict.items()}
    with torch.no_grad():
      model_outputs = model(**data_dict)
    face_kpts, eyes_kpts, eyes_gaze = [o.to('cpu') for o in model_outputs]

    output_dict = dict(
      face_kpts_3d=face_kpts[0],
      reye_kpts_3d=eyes_kpts[0, 0],
      leye_kpts_3d=eyes_kpts[0, 1],
      reye_origin_3d=eyes_gaze[0, 0, 0],
      reye_vector=eyes_gaze[0, 0, 1],
      leye_origin_3d=eyes_gaze[0, 1, 0],
      leye_vector=eyes_gaze[0, 1, 1],
    )

    return output_dict

  def _project_points_to_image(self, points_3d: np.ndarray):
    # Project 3D points from norm camera space to norm image space
    image_coords = np.dot(self.norm_intrinsic, points_3d.T)
    homo_coords = (image_coords / image_coords[2, :]).T

    # Convert 2D points from norm image space to real image space
    image_coords = np.dot(homo_coords, self.M_norm2real.T)[:, :2]

    return image_coords

  def _point_of_gaze(self, origin: np.ndarray, vector: np.ndarray):
    # Note: 1. convert unit length from 10cm (norm) to 1mm (real)
    #       2. the screen is in parallel to camera's XoY plane
    #
    # Point of Gaze: (xc, yc, zc), real camera space, 10cm as unit
    # Gaze Origin: (ox, oy, oz), real camera space, 10cm as unit
    # Gaze Vector: (vx, vy, vz), real camera space, 10cm as unit

    [ox, oy, oz], [vx, vy, vz] = origin, vector

    zc = self.real_extrinsic[2, 3] / 1e2
    t = (zc - oz) / (vz + self.epsilon)

    xc = ox + t * vx
    yc = oy + t * vy

    gaze_c = 1e2 * np.array([xc, yc, zc], dtype=np.float32)

    R, T = self.real_extrinsic[:, :3], self.real_extrinsic[:, 3]
    gaze_s_mm = np.dot(R.T, gaze_c - T)
    gaze_s_px = np.array([
      gaze_s_mm[0] / self.screen_hw_mm[1] * self.screen_hw_px[1],
      gaze_s_mm[1] / self.screen_hw_mm[0] * self.screen_hw_px[0],
    ], dtype=np.float32)

    return gaze_c, gaze_s_px

  def _model_post_proc(self, output_dict: dict):
    '''Post-process model outputs to 2D image coordinates and PoG.'''

    proc_dict = dict(
      face_kpts_2d=self._project_points_to_image(output_dict['face_kpts_3d']),
      reye_kpts_2d=self._project_points_to_image(output_dict['reye_kpts_3d']),
      leye_kpts_2d=self._project_points_to_image(output_dict['leye_kpts_3d']),
    )

    nasal_distance = 1e2 * np.linalg.norm(output_dict['face_kpts_3d'][2])
    proc_dict.update(nasal_distance=nasal_distance)

    reye_origin_2d, leye_origin_2d = self._project_points_to_image(
      np.stack([output_dict['reye_origin_3d'], output_dict['leye_origin_3d']])
    )
    proc_dict.update(reye_origin_2d=reye_origin_2d, leye_origin_2d=leye_origin_2d)

    reye_target_2d, leye_target_2d = self._project_points_to_image(
      np.stack([
        output_dict['reye_origin_3d'] + output_dict['reye_vector'],
        output_dict['leye_origin_3d'] + output_dict['leye_vector'],
      ])
    )
    proc_dict.update(reye_target_2d=reye_target_2d, leye_target_2d=leye_target_2d)

    # Initialize gaze coordinates with default values (will be overwritten by actual computation)
    reye_gaze_c = np.zeros(3, dtype=np.float32)
    reye_gaze_s = np.zeros(2, dtype=np.float32)
    leye_gaze_c = np.zeros(3, dtype=np.float32)
    leye_gaze_s = np.zeros(2, dtype=np.float32)

    if self.gaze_source == 'gaze':
      reye_gaze_c, reye_gaze_s = self._point_of_gaze(
        origin=output_dict['reye_origin_3d'],
        vector=output_dict['reye_vector'],
      )
      leye_gaze_c, leye_gaze_s = self._point_of_gaze(
        origin=output_dict['leye_origin_3d'],
        vector=output_dict['leye_vector'],
      )

    if self.gaze_source == 'mesh':
      reye_gaze_c, reye_gaze_s = self._point_of_gaze(
        origin=(output_dict['reye_kpts_3d'][35] + output_dict['reye_kpts_3d'][82]) / 2,
        vector=(output_dict['reye_kpts_3d'][35] - output_dict['reye_kpts_3d'][82]) / 2,
      )
      leye_gaze_c, leye_gaze_s = self._point_of_gaze(
        origin=(output_dict['leye_kpts_3d'][33] + output_dict['leye_kpts_3d'][80]) / 2,
        vector=(output_dict['leye_kpts_3d'][33] - output_dict['leye_kpts_3d'][80]) / 2,
      )

    proc_dict.update(reye_gaze_c=reye_gaze_c, reye_gaze_s=reye_gaze_s)
    proc_dict.update(leye_gaze_c=leye_gaze_c, leye_gaze_s=leye_gaze_s)

    return proc_dict

  def process(self, frame: np.ndarray, model: TdGazeNet):
    '''Process frame through face detection, model inference, and post-processing.'''

    frame = cv2.undistort(frame, self.real_intrinsic, self.real_dist_coef)

    self.time.tick(tag='mediapipe')
    landmarks = self.landmarker.process(frame, bgr2rgb=True)
    self.time.tock(tag='mediapipe')

    if landmarks is None:
      return dict(success=False, frame=frame, message='No face detected.')

    bbox = self._bbox_from_ldmk(landmarks)

    face_crop = scaled_crop(frame, bbox, self.image_size)
    face_bbox = self._norm_bbox(bbox)

    data_dict = self._model_data_dict(face_crop, face_bbox)

    self.time.tick(tag='inference')
    output_dict = self._model_inference(model, data_dict)
    self.time.tock(tag='inference')

    proc_dict = self._model_post_proc(output_dict)

    # Update PoG position from model output
    # Average of left and right eye PoGs
    # Defensive check: ensure gaze data exists
    if 'reye_gaze_s' not in proc_dict or 'leye_gaze_s' not in proc_dict:
      return dict(success=False, frame=frame, message='Gaze computation failed.')

    avg_gaze_s = (proc_dict['reye_gaze_s'] + proc_dict['leye_gaze_s']) / 2
    self.pog_pos = avg_gaze_s

    return dict(success=True, frame=frame, **output_dict, **proc_dict)

  def _update_animation(self, dt: float):
    '''Update target and particles based on PoG position.'''

    # Update target
    self.target.update(self.pog_pos, dt)

    # Check for fixation (explosion trigger)
    if not self.target.is_exploded and self.target.is_fixated:
      new_particles = self.target.explode()
      self.particles.extend(new_particles)

    # Update particles
    self.particles = [p for p in self.particles if p.update(dt)]

  def _draw_frame(self, canvas: np.ndarray, result_dict: dict):
    '''Draw camera frame with landmarks and gaze vectors on canvas.'''

    screen_h, screen_w = self.screen_hw_px

    if result_dict['success']:
      kpts_kwargs = dict(
        face_kpts_2d=dict(radius=2, color=(0, 255, 0), thickness=-1),
        reye_kpts_2d=dict(radius=2, color=(255, 255, 0), thickness=-1),
        leye_kpts_2d=dict(radius=2, color=(255, 255, 0), thickness=-1),
      )
      for name, kwargs in kpts_kwargs.items():
        for pt in result_dict[name]:
          cv2.circle(result_dict['frame'], pt.astype(np.int32), **kwargs)

      gaze_kwargs = dict(color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA)
      cv2.arrowedLine(
        result_dict['frame'],
        result_dict['reye_origin_2d'].astype(np.int32),
        result_dict['reye_target_2d'].astype(np.int32),
        **gaze_kwargs,
      )
      cv2.arrowedLine(
        result_dict['frame'],
        result_dict['leye_origin_2d'].astype(np.int32),
        result_dict['leye_target_2d'].astype(np.int32),
        **gaze_kwargs,
      )

    frame_h, frame_w, _ = result_dict['frame'].shape

    avail_h = screen_h // 1 - 2 * self.disp_margin
    avail_w = screen_w // 2 - 2 * self.disp_margin

    scale_h = avail_h / frame_h
    scale_w = avail_w / frame_w
    scale = min(scale_h, scale_w)

    final_h = int(frame_h * scale)
    final_w = int(frame_w * scale)

    y_offset = (avail_h - final_h) // 2 + self.disp_margin
    x_offset = (avail_w - final_w) // 2 + self.disp_margin

    frame = cv2.resize(result_dict['frame'], (final_w, final_h))
    canvas[y_offset:y_offset+final_h, x_offset:x_offset+final_w] = frame

    # Draw gaze on the main screen
    if result_dict['success']:
      gaze_kwargs = dict(radius=20, thickness=-1)
      cv2.circle(
        canvas,
        result_dict['reye_gaze_s'].astype(np.int32),
        color=(0, 0, 255),
        **gaze_kwargs,
      )
      cv2.circle(
        canvas,
        result_dict['leye_gaze_s'].astype(np.int32),
        color=(255, 0, 0),
        **gaze_kwargs,
      )

  def _draw_animation_elements(self, canvas: np.ndarray):
    '''Draw animated target and particles.'''

    # Draw particles
    for particle in self.particles:
      particle.draw(canvas)

    # Draw target
    self.target.draw(canvas)

    # Draw PoG cursor
    px, py = self.pog_pos.astype(np.int32)
    # Outer ring (purple-pink)
    cv2.circle(canvas, (px, py), 10, (246, 92, 199), 2, cv2.LINE_AA)
    # Inner dot (white)
    cv2.circle(canvas, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)

  def _draw_dashboard(self, canvas: np.ndarray):
    '''Draw dashboard with status information'''

    # Draw modern dashboard with bold text
    state_str = 'IDLE'
    state_color = (237, 237, 237)  # White-ish
    if self.target.is_fixating:
      state_str = f'FIXATING ({self.target.dwell_progress*100:.0f}%)'
      state_color = (246, 92, 199)  # Purple-pink
    if self.target.triggered:
      state_str = 'TRIGGERED'
      state_color = (55, 0, 218)  # Modern red
    if self.target.must_exit:
      state_str = 'EXIT REQUIRED'
      state_color = (55, 0, 218)
    if self.target.is_exploded:
      state_str = 'EXPLODED'
      state_color = (55, 0, 218)

    # Draw title section
    cv2.putText(
      canvas,
      'TdGazeNet Demo with Animation',
      (30, 35),
      cv2.FONT_HERSHEY_DUPLEX,
      0.8,
      (246, 92, 199),
      2,
      cv2.LINE_AA,
    )

    # Draw state indicator
    cv2.putText(
      canvas,
      f'STATE: {state_str}',
      (30, 75),
      cv2.FONT_HERSHEY_DUPLEX,
      0.7,
      state_color,
      2,
      cv2.LINE_AA,
    )

    # Draw stats (thicker, modern look)
    stats = [
      (f'Proximity: {self.target.proximity:.2f}', (237, 237, 237)),
      (f'Distance: {self.target.distance_to_pog:.1f} px', (237, 237, 237)),
      (f'Particles: {len(self.particles)}', (237, 237, 237)),
    ]

    y_offset = 115
    for text, color in stats:
      cv2.putText(
        canvas,
        text,
        (30, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
      )
      y_offset += 30

  def _draw_pipeline_metrics(self, canvas: np.ndarray, result_dict: dict):
    '''Draw pipeline performance metrics in bottom-left corner with hints.'''

    # Position metrics in bottom-left area, above hints
    x_start = 30
    y_start = self.screen_h - 130
    line_height = 25

    # Initialize y_offset before conditional block to prevent UnboundLocalError
    y_offset = y_start

    if result_dict.get('success', False):
      # Pipeline metrics
      metrics = [
        f'MediaPipe: {1e3 * self.time.report(tag="mediapipe"):.2f} ms',
        f'Inference: {1e3 * self.time.report(tag="inference"):.2f} ms',
        f'Nasal Dist: {result_dict.get("nasal_distance", 0):.2f} mm',
      ]

      for text in metrics:
        cv2.putText(
          canvas,
          text,
          (x_start, y_offset),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (150, 150, 150),
          1,
          cv2.LINE_AA,
        )
        y_offset += line_height

    # Draw hints below metrics
    hints = [
      'Move your head to interact with target',
      'Press \'X\' to exit',
    ]

    y_offset += 10
    for text in hints:
      cv2.putText(
        canvas,
        text,
        (x_start, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
      )
      y_offset += 25

  def display(self, result_dict: dict):
    '''Render frame, animation elements, and dashboard to fullscreen window.'''

    canvas = np.zeros(
      shape=(*self.screen_hw_px, 3),
      dtype=np.uint8,
    )

    # Draw background grid (subtle gray)
    grid_spacing = 100
    for x in range(0, self.screen_w, grid_spacing):
      cv2.line(canvas, (x, 0), (x, self.screen_h), (68, 68, 68), 1)
    for y in range(0, self.screen_h, grid_spacing):
      cv2.line(canvas, (0, y), (self.screen_w, y), (68, 68, 68), 1)

    # Draw the original frame with face landmarks and gaze overlay
    self._draw_frame(canvas, result_dict)

    # Update animation elements
    current_time = time.time()
    dt = current_time - self.last_time
    self.last_time = current_time

    # Cap dt to prevent physics explosions on lag
    dt = min(dt, 0.1)
    self._update_animation(dt)

    # Draw animation elements (target and particles)
    self._draw_animation_elements(canvas)

    # Draw dashboard info (animation state - left side)
    self._draw_dashboard(canvas)

    # Draw pipeline metrics (right side)
    self._draw_pipeline_metrics(canvas, result_dict)

    cv2.imshow(self.name, canvas)
    return cv2.waitKey(6) & 0xFF == ord('X')


# Adaptation to Inference on Image Sequence
class ImageCaptureHandler(CaptureHandler):
  def main_loop(self, **extra_kwargs):
    capture = self.capture_builder.build()
    exit_cond, set_exit_cond = use_state(False)

    while not exit_cond():
      success, src_image = capture.read()
      if not success: break
      self.frame_consumer(src_image, set_exit_cond, **extra_kwargs)

    capture.release()


class ImageFrameConsumer(FrameConsumer):
  def __call__(self, src_image: np.ndarray, set_exit_cond: Callable, model: TdGazeNet):
    '''Process image sequence frame and write results to file.'''

    result_dict = self.process(src_image, model)

    if not self.headless:
      exit_cond = self.display(result_dict)
      set_exit_cond(exit_cond)

    if result_dict['success']:
      gaze_dict = dict(
        success=True,
        reye_gaze_c=result_dict['reye_gaze_c'].tolist(),
        reye_gaze_s=result_dict['reye_gaze_s'].tolist(),
        leye_gaze_c=result_dict['leye_gaze_c'].tolist(),
        leye_gaze_s=result_dict['leye_gaze_s'].tolist(),
      )
    else:
      gaze_dict = dict(success=False, message=result_dict['message'])

    self.results_file.write(json.dumps(gaze_dict) + '\n')

  def __init__(self, image_sequence: str, headless: bool, **kwargs):
    '''Initialize ImageFrameConsumer for batch processing image sequences.'''

    super(ImageFrameConsumer, self).__init__(**kwargs)

    images_folder = osp.abspath(osp.dirname(image_sequence))
    self.results_path = osp.join(
      osp.dirname(images_folder),
      'TdGazeNet-results.jsonl',
    )
    self.headless = headless

  def __enter__(self):
    self.results_file = open(self.results_path, 'w', encoding='utf-8')

    if not self.headless:
      return super().__enter__()

    # Manage face landmarker manually in headless mode
    self.landmarker.create()

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.results_file.close()

    if not self.headless:
      super().__exit__(exc_type, exc_val, exc_tb)
    else:
      self.landmarker.destroy()


# Entrypoint, Arguments and Top-Level Utilities
def load_wrapped_model(
  demo_data: dict,
  state_dict_file: str,
  device: torch.device,
  optimize: str,
):
  '''Load TdGazeNet model from checkpoint and apply optimization if specified.'''

  model = TdGazeNet(**demo_data['model']).to(device=device)

  state_dict = torch.load(state_dict_file, map_location=device)
  params = collections.OrderedDict()
  for key in state_dict:
    param_key = key.removeprefix('model.')
    params[param_key] = state_dict[key]
  model.load_state_dict(params, strict=False)

  if optimize == 'reparameterize':
    for name, module in model.named_modules():
      if hasattr(module, 'reparameterize'):
        module.reparameterize()

  return model.eval()


def main_procedure(opts: argparse.Namespace):
  '''Main entry point for TdGazeNet animation demo.'''

  demo_data = _load_demo_data(opts.demo_data)
  model = load_wrapped_model(
    demo_data,
    state_dict_file=opts.state_dict_file,
    device=torch.device(opts.device),
    optimize=opts.optimize,
  )

  consumer_kwargs = dict(
    landmarker=FaceLandmarks(p_detection=0.8, p_presence=0.8),
    device=torch.device(opts.device),
  )
  if opts.image_sequence:
    demo_data['camera']['capture_id'] = opts.image_sequence
    consumer = ImageFrameConsumer(
      image_sequence=opts.image_sequence,
      headless=opts.image_headless_mode,
      demo_data=demo_data,
      **consumer_kwargs,
    )
    capture_handler_cls = ImageCaptureHandler
  else:
    consumer = FrameConsumer(demo_data=demo_data, **consumer_kwargs)
    capture_handler_cls = CaptureHandler

  capture_builder = VideoCaptureBuilder(**_capture_builder_kwargs(demo_data))
  capture_handler = capture_handler_cls(capture_builder, consumer)
  with consumer:
    capture_handler.main_loop(model=model)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='run demo for TdGazeNet with target animation.',
  )

  parser.add_argument(
    '--demo-data',
    type=str,
    default='',
    help='optional demo data file, use DEFAULT if empty.',
  )
  parser.add_argument(
    '--image-sequence',
    type=str,
    default='',
    help='optional input image sequence for `cv2.VideoCapture`.',
  )
  parser.add_argument(
    '--image-headless-mode',
    action='store_true',
    default=False,
    help='run headless mode for image sequence (no display).',
  )

  model_group = parser.add_argument_group(
    title='model options',
    description='model options for script.',
  )

  model_group.add_argument(
    '--state-dict-file',
    type=str,
    required=True,
    help='state dict file extracted from mmengine checkpoint.',
  )
  model_group.add_argument(
    '--device',
    type=str,
    default='cpu',
    help='inference device, see `torch.device` for more details.',
  )
  model_group.add_argument(
    '--optimize',
    type=str,
    default='none',
    choices=['none', 'reparameterize'],
    help='inference optimization method for TdGazeNet.',
  )

  main_procedure(parser.parse_args())
