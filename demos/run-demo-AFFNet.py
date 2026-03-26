from opengaze.runtime.camera import VideoCaptureBuilder, CaptureHandler
from opengaze.runtime.camera import use_state
from opengaze.runtime.time import TimeViaEMA
from opengaze.utils.gc import FaceLandmarks, FaceAlignment
from opengaze.model.gaze_2d import AFFNet

from PIL import Image
from typing import Callable

import argparse
import collections
import cv2
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
class FrameConsumer:
  def __call__(self, src_image: np.ndarray, set_exit_cond: Callable, model: AFFNet):
    result_dict = self.process(src_image, model)
    exit_cond = self.display(result_dict)
    set_exit_cond(exit_cond)
    return result_dict, exit_cond

  def __init__(self, demo_data: dict, landmarker: FaceLandmarks,
               alignment: FaceAlignment, device: torch.device):
    self.name = 'AFFNet Demo'

    self.demo_data = demo_data
    self.landmarker = landmarker
    self.alignment = alignment
    self.device = device

    self.time = TimeViaEMA(alpha=0.1)

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
  def source_size(self):
    return [
      self.demo_data['camera']['frame_h'],
      self.demo_data['camera']['frame_w'],
    ]

  @property
  def adjust_size(self):
    return self.demo_data['pre_process']['adjust_size']

  @property
  def norm_params(self):
    return dict(
      mean=self.demo_data['pre_process']['norm_mean'],
      std=self.demo_data['pre_process']['norm_std'],
    )

  @property
  def screen_origin(self):
    return np.array([
      self.demo_data['camera']['screen_x_mm'],
      self.demo_data['camera']['screen_y_mm'],
    ], dtype=np.float32)

  @property
  def screen_hw_px(self):
    return (self.demo_data['screen']['h_px'], self.demo_data['screen']['w_px'])

  @property
  def screen_hw_mm(self):
    return (self.demo_data['screen']['h_mm'], self.demo_data['screen']['w_mm'])

  @property
  def disp_margin(self):
    return self.demo_data['post_process']['disp_margin']

  def _adjust_image_size(self, image: np.ndarray):
    src_res, tgt_res = self.source_size, self.adjust_size

    src_asp = src_res[1] / src_res[0]
    tgt_asp = tgt_res[1] / tgt_res[0]

    if tgt_asp > src_asp:
      rescale_h = int(src_res[1] / tgt_asp)
      padding_h = (src_res[0] - rescale_h) // 2
      image = image[padding_h:-padding_h, :]
    if tgt_asp < src_asp:
      rescale_w = int(src_res[0] * tgt_asp)
      padding_w = (src_res[1] - rescale_w) // 2
      image = image[:, padding_w:-padding_w]

    dsize = (tgt_res[1], tgt_res[0])
    image = cv2.resize(image, dsize, interpolation=cv2.INTER_CUBIC)

    return image

  def _image_to_inputs(self, image: np.ndarray, hflip: bool = False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image, mode='RGB')
    image = TF.to_tensor(image).unsqueeze(0)
    image = TF.normalize(image, **self.norm_params)
    return TF.hflip(image) if hflip else image

  def _model_data_dict(self, align_dict: dict):
    face = self._image_to_inputs(align_dict['face_crop'])
    reye = TF.resize(
      self._image_to_inputs(align_dict['reye_crop'], hflip=True),
      size=(112, 112),
      interpolation=TF.InterpolationMode.BICUBIC,
    )
    leye = TF.resize(
      self._image_to_inputs(align_dict['leye_crop'], hflip=False),
      size=(112, 112),
      interpolation=TF.InterpolationMode.BICUBIC,
    )
    rect = torch.from_numpy(np.concatenate([
      align_dict['face_bbox'],
      align_dict['reye_bbox'],
      align_dict['leye_bbox'],
    ], axis=0, dtype=np.float32)).unsqueeze(0)
    return dict(face=face, reye=reye, leye=leye, rect=rect)

  def _model_inference(self, model: AFFNet, data_dict: dict):
    data_dict = {k:v.to(self.device) for k, v in data_dict.items()}
    with torch.no_grad():
      model_outputs = model(**data_dict)
    return dict(gaze=model_outputs[0].to('cpu'))

  def _rotate_gaze(self, gaze: np.ndarray, theta: float):
    radian = np.deg2rad(theta, dtype=np.float32)

    cos, sin = np.cos(radian), np.sin(radian)
    mat = np.array([[cos, -sin], [sin, cos]])

    return np.dot(mat, gaze)

  def _model_post_proc(self, align_dict: dict, output_dict: dict):
    # Note: 1. convert unit length from 1cm (camera) to 1mm (screen)
    #       2. the screen is in parallel to camera's XoY plane
    #
    # Point of Gaze: X-axis points rightward, Y-axis points upward
    #                Origin is at the center of the pinhole camera

    gaze_c = 1e1 * self._rotate_gaze(output_dict['gaze'], align_dict['theta'])

    gaze_s_mm = np.array([1.0, -1.0]) * (gaze_c - self.screen_origin)
    gaze_s_px = np.array([
      gaze_s_mm[0] / self.screen_hw_mm[1] * self.screen_hw_px[1],
      gaze_s_mm[1] / self.screen_hw_mm[0] * self.screen_hw_px[0],
    ], dtype=np.float32)

    return dict(gaze_c=gaze_c, gaze_s=gaze_s_px)

  def process(self, frame: np.ndarray, model: AFFNet):
    adjusted_frame = self._adjust_image_size(frame)

    self.time.tick(tag='mediapipe')
    landmarks = self.landmarker.process(adjusted_frame, bgr2rgb=True)
    self.time.tock(tag='mediapipe')

    if landmarks is None:
      return dict(success=False, frame=frame, message='No face detected.')

    align_dict = self.alignment.align(adjusted_frame, landmarks)

    data_dict = self._model_data_dict(align_dict)

    self.time.tick(tag='inference')
    output_dict = self._model_inference(model, data_dict)
    self.time.tock(tag='inference')

    proc_dict = self._model_post_proc(align_dict, output_dict)

    return dict(success=True, frame=frame, **proc_dict)

  def _draw_frame(self, canvas: np.ndarray, result_dict: dict):
    screen_h, screen_w = self.screen_hw_px

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
        f'MediaPipe: {1e3 * self.time.report(tag="mediapipe"):.2f} ms',
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
  def __call__(self, src_image: np.ndarray, set_exit_cond: Callable, model: AFFNet):
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
      'AFFNet-results.jsonl',
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
def load_wrapped_model(demo_data: dict, state_dict_file: str, device: torch.device):
  model = AFFNet().to(device=device)

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
    alignment=FaceAlignment(width_expand=1.6, hw_ratio=1.0),
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
  parser = argparse.ArgumentParser(description='run demo for AFFNet.')

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
