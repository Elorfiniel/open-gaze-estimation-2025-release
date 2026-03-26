from opengaze.runtime.camera import VideoCaptureBuilder, CaptureHandler
from opengaze.runtime.camera import use_state
from opengaze.runtime.scripts import ScriptEnv
from opengaze.runtime.time import TimeViaEMA
from opengaze.utils import MpiiDataNormalizer, FaceBoundingBox, SparseFaceLandmarks
from opengaze.utils.euler import gaze_2d_3d_a
from opengaze.utils.geom import PoseEstimator
from opengaze.model.gaze_3d import XGaze224

from PIL import Image
from typing import Callable

import argparse
import cv2
import collections
import json
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
class HeadPoseEstimator(PoseEstimator):
  def __init__(self, cam_mat: np.ndarray, cam_dist: np.ndarray = None):
    super().__init__(cam_mat=cam_mat, cam_dist=cam_dist)

    xgaze_model_path = ScriptEnv.resource_path('face-models/ethxgaze-generic.txt')
    face_model = np.loadtxt(xgaze_model_path, dtype=np.float32)
    landmarks_indices = [
      20, # reye, outer
      23, # reye, inner
      26, # leye, inner
      29, # leye, outer
      15, # mouth, rc
      19, # mouth, lc
    ]
    self.face_model = face_model[landmarks_indices, :]

  def estimate(self, landmarks_2d: np.ndarray):
    return super().estimate(self.face_model, landmarks_2d)


class FrameConsumer:
  def __call__(self, src_image: np.ndarray, set_exit_cond: Callable, model: XGaze224):
    result_dict = self.process(src_image, model)
    exit_cond = self.display(result_dict)
    set_exit_cond(exit_cond)
    return result_dict, exit_cond

  def __init__(self, demo_data: dict, face_bbox: FaceBoundingBox, device: torch.device):
    self.name = 'XGaze224 Demo'

    self.demo_data = demo_data
    self.face_bbox = face_bbox
    self.device = device

    self.landmarker = SparseFaceLandmarks(width_expand=1.6, image_size=120)
    self.normalizer = MpiiDataNormalizer(960, (224, 224), distance=300)
    self.pose_estim = HeadPoseEstimator(self.real_intrinsic, self.real_dist_coef)

    self.time = TimeViaEMA(alpha=0.1)

  def __enter__(self):
    # Manage OpenCV resources
    cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Manage face bounding box
    self.face_bbox.create()

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    cv2.destroyWindow(self.name)
    self.face_bbox.destroy()

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
  def screen_hw_px(self):
    return (self.demo_data['screen']['h_px'], self.demo_data['screen']['w_px'])

  @property
  def screen_hw_mm(self):
    return (self.demo_data['screen']['h_mm'], self.demo_data['screen']['w_mm'])

  @property
  def norm_params(self):
    return dict(
      mean=self.demo_data['pre_process']['norm_mean'],
      std=self.demo_data['pre_process']['norm_std'],
    )

  @property
  def epsilon(self):
    return self.demo_data['post_process']['epsilon']

  @property
  def disp_margin(self):
    return self.demo_data['post_process']['disp_margin']

  def _calculate_face_center(self, landmarks_3d: np.ndarray):
    # Use convention of ETH-XGaze data normalization

    eyes_center = np.mean(landmarks_3d[:, 0:4], axis=1)
    mouth_center = np.mean(landmarks_3d[:, 4:6], axis=1)
    face_center = 0.5 * (eyes_center + mouth_center)

    return face_center

  def _normalize_face(self, frame: np.ndarray, look_at: np.ndarray, R1: np.ndarray):
    Kv, S, R2, W = self.normalizer.normalize_matrices(look_at, R1, self.real_intrinsic)
    face_crop = self.normalizer.warp_image(frame, W)
    return face_crop, R2

  def _denormalize_gaze(self, gaze: np.ndarray):
    gaze_3d = np.array(gaze_2d_3d_a(gaze[0], gaze[1]))
    return gaze_3d / np.linalg.norm(gaze_3d)

  def _model_data_dict(self, face_crop: np.ndarray):
    face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face, mode='RGB')
    face = TF.to_tensor(face).unsqueeze(0)
    face = TF.normalize(face, **self.norm_params)
    return dict(face=face)

  def _model_inference(self, model: XGaze224, data_dict: dict):
    data_dict = {k:v.to(self.device) for k, v in data_dict.items()}
    with torch.no_grad():
      model_outputs = model(**data_dict)
    return dict(gaze=model_outputs[0].to('cpu'))

  def _project_points_to_image(self, points_3d: np.ndarray):
    # Project 3D points from real camera space to real image space
    image_coords = np.dot(self.real_intrinsic, points_3d.T)
    image_coords = (image_coords[:2, :] / image_coords[2, :]).T
    return image_coords

  def _point_of_gaze(self, origin: np.ndarray, vector: np.ndarray):
    # Note: 1. the screen is in parallel to camera's XoY plane
    #
    # Point of Gaze: (xc, yc, zc), real camera space, 1mm as unit
    # Gaze Origin: (ox, oy, oz), real camera space, 1mm as unit
    # Gaze Vector: (vx, vy, vz), real camera space, 1mm as unit

    [ox, oy, oz], [vx, vy, vz] = origin, vector

    zc = self.real_extrinsic[2, 3]
    t = (zc - oz) / (vz + self.epsilon)

    xc = ox + t * vx
    yc = oy + t * vy

    gaze_c = np.array([xc, yc, zc], dtype=np.float32)

    R, T = self.real_extrinsic[:, :3], self.real_extrinsic[:, 3]
    gaze_s_mm = np.dot(R.T, gaze_c - T)
    gaze_s_px = np.array([
      gaze_s_mm[0] / self.screen_hw_mm[1] * self.screen_hw_px[1],
      gaze_s_mm[1] / self.screen_hw_mm[0] * self.screen_hw_px[0],
    ], dtype=np.float32)

    return gaze_c, gaze_s_px

  def _model_post_proc(self, look_at: np.ndarray, R2: np.ndarray, output_dict: dict):
    proc_dict = dict(face_center=look_at)

    gaze_3d = self._denormalize_gaze(output_dict['gaze'])

    origin_2d, target_2d = self._project_points_to_image(
      np.stack([look_at, look_at + 1e2 * gaze_3d])
    )
    proc_dict.update(origin_2d=origin_2d, target_2d=target_2d)

    gaze_c, gaze_s = self._point_of_gaze(
      origin=look_at,
      vector=np.dot(R2.T, gaze_3d),
    )
    proc_dict.update(gaze_c=gaze_c, gaze_s=gaze_s)

    return proc_dict

  def process(self, frame: np.ndarray, model: XGaze224):
    frame = cv2.undistort(frame, self.real_intrinsic, self.real_dist_coef)

    self.time.tick(tag='face-bbox')
    bboxes = self.face_bbox.process(frame, bgr2rgb=True)
    self.time.tock(tag='face-bbox')

    if bboxes is None:
      return dict(success=False, frame=frame, message='No face detected.')

    ldmks_3d, ldmks_2d = self.landmarker.process(frame, bboxes[0], bgr2rgb=True)
    landmarks_2d = ldmks_2d[[
      36, # reye, outer
      39, # reye, inner
      42, # leye, inner
      45, # leye, outer
      60, # mouth, rc
      64, # mouth, lc
    ]]
    hr, ht = self.pose_estim.estimate(landmarks_2d)
    hR = cv2.Rodrigues(hr)[0]
    landmarks_3d = np.dot(hR, self.pose_estim.face_model.T) + ht
    face_center = self._calculate_face_center(landmarks_3d)

    face_crop, R2 = self._normalize_face(frame, face_center, hR)

    data_dict = self._model_data_dict(face_crop)

    self.time.tick(tag='inference')
    output_dict = self._model_inference(model, data_dict)
    self.time.tock(tag='inference')

    proc_dict = self._model_post_proc(face_center, R2, output_dict)

    return dict(success=True, frame=frame, **output_dict, **proc_dict)

  def _draw_frame(self, canvas: np.ndarray, result_dict: dict):
    screen_h, screen_w = self.screen_hw_px

    if result_dict['success']:
      gaze_kwargs = dict(color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA)
      cv2.arrowedLine(
        result_dict['frame'],
        result_dict['origin_2d'].astype(np.int32),
        result_dict['target_2d'].astype(np.int32),
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
        result_dict['gaze_s'].astype(np.int32),
        color=(0, 0, 255),
        **gaze_kwargs,
      )

  def _draw_text(self, canvas: np.ndarray, result_dict: dict):
    screen_h, screen_w = self.screen_hw_px

    text_kwargs = dict(
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
      color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA,
    )

    if result_dict['success']:
      # Measured time for different stages
      text = ', '.join([
        f'Face-BBox: {1e3 * self.time.report(tag="face-bbox"):.2f} ms',
        f'Inference: {1e3 * self.time.report(tag="inference"):.2f} ms',
      ])
      cv2.putText(canvas, text, (30, screen_h - 30), **text_kwargs)

  def display(self, result_dict: dict):
    canvas = np.zeros(shape=(*self.screen_hw_px, 3), dtype=np.uint8)
    self._draw_frame(canvas, result_dict)
    self._draw_text(canvas, result_dict)
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
  def __call__(self, src_image: np.ndarray, set_exit_cond: Callable, model: XGaze224):
    result_dict = self.process(src_image, model)

    if not self.headless:
      exit_cond = self.display(result_dict)
      set_exit_cond(exit_cond)

    if result_dict['success']:
      gaze_dict = dict(
        success=True,
        gaze_c=result_dict['gaze_c'].tolist(),
        gaze_s=result_dict['gaze_s'].tolist(),
      )
    else:
      gaze_dict = dict(success=False, message=result_dict['message'])

    self.results_file.write(json.dumps(gaze_dict) + '\n')

  def __init__(self, image_sequence: str, headless: bool, **kwargs):
    super(ImageFrameConsumer, self).__init__(**kwargs)

    images_folder = osp.abspath(osp.dirname(image_sequence))
    self.results_path = osp.join(
      osp.dirname(images_folder),
      'XGaze224-results.jsonl',
    )
    self.headless = headless

  def __enter__(self):
    self.results_file = open(self.results_path, 'w', encoding='utf-8')

    if not self.headless:
      return super().__enter__()

    # Manage face bounding box manually in headless mode
    self.face_bbox.create()

    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.results_file.close()

    if not self.headless:
      super().__exit__(exc_type, exc_val, exc_tb)
    else:
      self.face_bbox.destroy()


# Entrypoint, Arguments and Top-Level Utilities
def load_wrapped_model(demo_data: dict, state_dict_file: str, device: torch.device):
  model = XGaze224().to(device=device)

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
    face_bbox=FaceBoundingBox(p_detection=0.8, p_suppression=0.2),
    device=torch.device(opts.device),
  )
  if opts.image_sequence:
    demo_data['camera']['capture_id'] = opts.image_sequence
    consumer = ImageFrameConsumer(
      image_sequence=opts.image_sequence,
      headless=opts.image_headless_mode,
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
  parser = argparse.ArgumentParser(description='run demo for XGaze224 model.')

  parser.add_argument(
    '--demo-data', type=str, default='',
    help='optional demo data file, use DEFAULT if empty.',
  )
  parser.add_argument(
    '--image-sequence', type=str, default='',
    help='optional input image sequence for `cv2.VideoCapture`.',
  )
  parser.add_argument(
    '--image-headless-mode', action='store_true', default=False,
    help='run headless mode for image sequence (no display).',
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
