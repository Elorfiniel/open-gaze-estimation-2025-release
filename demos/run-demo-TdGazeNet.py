from opengaze.runtime.camera import VideoCaptureBuilder, CaptureHandler
from opengaze.runtime.camera import use_state
from opengaze.utils.gc import FaceLandmarks
from opengaze.utils.image import scaled_crop
from opengaze.model.gaze_dp import TdGazeNet

from PIL import Image
from typing import Callable

import argparse
import cv2
import collections
import json
import math
import numpy as np
import os.path as osp
import torch
import torchvision.transforms.functional as TF


# Utils for this demo script, not intended for other scripts
def _load_json_file(json_file: str, **kwargs):
  with open(json_file, 'r', encoding='utf-8') as file:
    json_data = json.load(file, **kwargs)
  return json_data

def _load_demo_data(data_path: str = ''):
  if not data_path:
    data_name = osp.basename(__file__).replace('.py', '.json')
    data_path = osp.join(osp.dirname(__file__), data_name)

  data_path = osp.abspath(data_path)
  if not osp.isfile(data_path):
    raise FileNotFoundError(f'No demo data found: "{data_path}".')

  demo_data = _load_json_file(data_path)

  return demo_data

def _capture_builder_kwargs(demo_data: dict):
  return dict(
    capture_id=demo_data['camera']['capture_id'],
    resolution=[
      demo_data['camera']['frame_h'],
      demo_data['camera']['frame_w'],
    ],
  )


# Data Transformations, Model Inference and Result Display
class FrameConsumer:
  def __call__(self, src_image: np.ndarray, set_exit_cond: Callable, model: TdGazeNet):
    result_dict = self.process(src_image, model)
    exit_cond = self.display(result_dict)
    set_exit_cond(exit_cond)
    return result_dict, exit_cond

  def __init__(self, demo_data: dict, landmarker: FaceLandmarks, device: torch.device):
    self.name = 'TdGazeNet Demo'
    self.demo_data = demo_data
    self.landmarker = landmarker
    self.device = device

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
    face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face, mode='RGB')
    face = TF.to_tensor(face).unsqueeze(0)
    face = TF.normalize(face, **self.norm_params)
    bbox = torch.tensor(face_bbox, dtype=torch.float32).unsqueeze(0)
    return dict(face=face, bbox=bbox)

  def _model_inference(self, model: TdGazeNet, data_dict: dict):
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
    proc_dict = dict(
      face_kpts_2d=self._project_points_to_image(output_dict['face_kpts_3d']),
      reye_kpts_2d=self._project_points_to_image(output_dict['reye_kpts_3d']),
      leye_kpts_2d=self._project_points_to_image(output_dict['leye_kpts_3d']),
    )

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
    frame = cv2.undistort(frame, self.real_intrinsic, self.real_dist_coef)

    landmarks = self.landmarker.process(frame, bgr2rgb=True)
    if landmarks is None:
      return dict(success=False, frame=frame, message='No face detected.')

    bbox = self._bbox_from_ldmk(landmarks)

    face_crop = scaled_crop(frame, bbox, self.image_size)
    face_bbox = self._norm_bbox(bbox)

    data_dict = self._model_data_dict(face_crop, face_bbox)
    output_dict = self._model_inference(model, data_dict)
    proc_dict = self._model_post_proc(output_dict)

    return dict(success=True, frame=frame, **output_dict, **proc_dict)

  def _draw_frame(self, canvas: np.ndarray, result_dict: dict):
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

  def display(self, result_dict: dict):
    canvas = np.zeros(shape=(*self.screen_hw_px, 3), dtype=np.uint8)
    self._draw_frame(canvas, result_dict)
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
    result_dict, _ = super().__call__(src_image, set_exit_cond, model)

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

  def __init__(self, image_sequence: str, **kwargs):
    super(ImageFrameConsumer, self).__init__(**kwargs)

    images_folder = osp.abspath(osp.dirname(image_sequence))
    self.results_path = osp.join(
      osp.dirname(images_folder),
      'TdGazeNet-results.jsonl',
    )

  def __enter__(self):
    self.results_file = open(self.results_path, 'w', encoding='utf-8')
    return super().__enter__()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.results_file.close()
    super().__exit__(exc_type, exc_val, exc_tb)


# Entrypoint, Arguments and Top-Level Utilities
def load_wrapped_model(demo_data: dict, state_dict_file: str, device: torch.device):
  model = TdGazeNet(**demo_data['model']).to(device=device)

  state_dict = torch.load(state_dict_file, map_location=device)
  params = collections.OrderedDict()
  for key in state_dict:
    param_key = key.removeprefix('model.')
    params[param_key] = state_dict[key]
  model.load_state_dict(params, strict=False)

  return model.eval()


def main_procedure(opts: argparse.Namespace):
  demo_data = _load_demo_data(opts.demo_data)
  model = load_wrapped_model(
    demo_data,
    state_dict_file=opts.state_dict_file,
    device=torch.device(opts.device),
  )

  consumer_kwargs = dict(
    landmarker=FaceLandmarks(p_detection=0.8, p_presence=0.8),
    device=torch.device(opts.device),
  )
  if opts.image_sequence:
    demo_data['camera']['capture_id'] = opts.image_sequence
    consumer = ImageFrameConsumer(
      image_sequence=opts.image_sequence,
      demo_data=demo_data, **consumer_kwargs,
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
  parser = argparse.ArgumentParser(description='run demo for TdGazeNet.')

  parser.add_argument(
    '--demo-data', type=str, default='',
    help='optional demo data file, use DEFAULT if empty.',
  )
  parser.add_argument(
    '--image-sequence', type=str, default='',
    help='optional input image sequence for `cv2.VideoCapture`.',
  )

  model_group = parser.add_argument_group(
    title='model options',
    description='model options for script.',
  )

  model_group.add_argument(
    '--state-dict-file', type=str, required=True,
    help='state dict file extracted from mmengine checkpoint.',
  )
  model_group.add_argument(
    '--device', type=str, default='cpu',
    help='inference device, see `torch.device` for more details.',
  )

  main_procedure(parser.parse_args())
