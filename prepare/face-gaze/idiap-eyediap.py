from opengaze.utils import MpiiDataNormalizer, FaceBoundingBox, SparseFaceLandmarks
from opengaze.utils.geom import PoseEstimator
from opengaze.runtime.scripts import ScriptEnv
from opengaze.runtime.log import runtime_logger
from opengaze.runtime.parallel import FunctionalTask, run_parallel

import argparse
import concurrent.futures as futures
import cv2
import h5py
import numpy as np
import os
import os.path as osp
import scipy.io as sio


rt_logger = runtime_logger(
  name='idiap-eyediap',
  log_file=ScriptEnv.log_path('prepare-idiap-eyediap.log'),
)


SAMPLE_EVERY_N_FRAMES, SAMPLE_OFFSET = 15, 0


class Camera2Normal:
  def __init__(self, cam_mat: np.ndarray, normalizer: MpiiDataNormalizer):
    self.cam_mat = cam_mat
    self.normalizer = normalizer

  def _unit_vector(self, v: np.ndarray):
    return v / np.linalg.norm(v)

  def normalize_data(self, look_at, R1, image, tgt):
    Kv, S, R2, W = self.normalizer.normalize_matrices(look_at, R1, self.cam_mat)
    warp = self.normalizer.warp_image(image, W)

    # Revisiting Data Normalization: discard the scaling component `S_x`
    gaze = self._unit_vector(np.dot(R2, tgt - look_at))
    pose = cv2.Rodrigues(np.dot(R2, R1))[0].reshape((3, ))

    return warp, gaze, pose

class HeadPoseEstimator(PoseEstimator):
  def __init__(self, cam_mat: np.ndarray):
    cam_dist = np.zeros(shape=(5, 1), dtype=np.float64)

    super().__init__(cam_mat=cam_mat, cam_dist=cam_dist)

    mpii_model_path = ScriptEnv.resource_path('face-models/mpiigaze-generic.mat')
    face_model = sio.loadmat(mpii_model_path)['model'].T
    self.face_model = face_model.reshape((6, 3)).astype(np.float32)

  def estimate(self, landmarks_2d: np.ndarray):
    return super().estimate(self.face_model, landmarks_2d)

class SessionData:
  def __init__(self, session_folder: str):
    self.record_metadata = self.load_record_metadata(session_folder)
    self.camera_params = self.load_camera_params(session_folder)
    self.screen_coords = self.load_screen_coords(session_folder)
    self.head_pose = self.load_head_pose(session_folder)
    self.eyes_data = self.load_eyes_data(session_folder)

  def load_record_metadata(self, session_folder: str):
    record_path = osp.join(session_folder, 'rgb_vga.mov')
    capture = cv2.VideoCapture(record_path, cv2.CAP_ANY)

    frame_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    n_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS)

    capture.release()

    return dict(
      frame_h=int(frame_h), frame_w=int(frame_w),
      n_frames=int(n_frames), fps=round(fps, 2),
    )

  def load_camera_params(self, session_folder: str):
    filepath = osp.join(session_folder, 'rgb_vga_calibration.txt')

    with open(filepath, 'r', encoding='utf-8') as file:
      file_contents = file.read().splitlines(keepends=False)

    assert file_contents[0] == '[resolution]'
    frame_w, frame_h = [int(x) for x in file_contents[1].split(';')]

    assert file_contents[2] == '[intrinsics]'
    cam_mat = np.array([
      [float(x) for x in file_contents[3].split(';')],
      [float(x) for x in file_contents[4].split(';')],
      [float(x) for x in file_contents[5].split(';')],
    ], dtype=np.float32)

    assert file_contents[6] == '[R]'
    R = np.array([
      [float(x) for x in file_contents[7].split(';')],
      [float(x) for x in file_contents[8].split(';')],
      [float(x) for x in file_contents[9].split(';')],
    ], dtype=np.float32)

    assert file_contents[10] == '[T]'
    T = np.array([
      float(x) for x in file_contents[11:14]
    ], dtype=np.float32)

    return dict(
      frame_h=frame_h, frame_w=frame_w,
      cam_mat=cam_mat, R=R, T=T,
    )

  def load_screen_coords(self, session_folder: str):
    filepath = osp.join(session_folder, 'screen_coordinates.txt')

    with open(filepath, 'r', encoding='utf-8') as file:
      file_contents = file.read().splitlines(keepends=False)

    assert file_contents[0].startswith('Screen')
    screen_coords = dict()  # Frame ID -> Data
    for line in file_contents[1:]:
      frame_id, *values = [float(x) for x in line.split(';')]
      screen_coords[int(frame_id)] = (
        values[0:2], values[2:5],
      ) # 2D coordinates, 3D coordinates

    return screen_coords

  def load_head_pose(self, session_folder: str):
    filepath = osp.join(session_folder, 'head_pose.txt')

    with open(filepath, 'r', encoding='utf-8') as file:
      file_contents = file.read().splitlines(keepends=False)

    assert file_contents[0].startswith('Head')
    head_pose = dict()  # Frame ID -> Data
    for line in file_contents[1:]:
      frame_id, *values = [float(x) for x in line.split(';')]
      head_pose[int(frame_id)] = (
        [values[0:3], values[3:6], values[6:9]], values[9:12],
      ) # Rotation R, Translation T

    return head_pose

  def load_eyes_data(self, session_folder: str):
    filepath = osp.join(session_folder, 'eye_tracking.txt')

    with open(filepath, 'r', encoding='utf-8') as file:
      file_contents = file.read().splitlines(keepends=False)

    assert file_contents[0].startswith('Eyes')
    eyes_data = dict()  # Frame ID -> Data
    for line in file_contents[1:]:
      frame_id, *values = [float(x) for x in line.split(';')]
      eyes_data[int(frame_id)] = (
        values[0:4], values[4:8], values[8:12], values[12:18],
      ) # RGB, Depth, HD, 3D: (Leye, Reye)

    return eyes_data

  def check_index(self, frame_id: int):
    return all([
      frame_id in self.screen_coords,
      frame_id in self.head_pose,
      frame_id in self.eyes_data,
    ])


def calculate_face_center(landmarks_3d: np.ndarray):
  # Use convention of ETH-XGaze data normalization

  eyes_center = np.mean(landmarks_3d[:, 0:4], axis=1)
  mouth_center = np.mean(landmarks_3d[:, 4:6], axis=1)
  face_center = 0.5 * (eyes_center + mouth_center)

  return face_center

def convert_gaze_target_coords(session_data: SessionData, frame_id: int):
  _, world_pt = session_data.screen_coords[frame_id]
  return 1e3 * np.array(world_pt, dtype=np.float32)

def process_session(sessions_folder, session, opt_folder):
  # Load annotations for current session
  session_folder = osp.join(sessions_folder, session)
  session_data = SessionData(session_folder)
  max_n_samples = session_data.record_metadata['n_frames']
  cam_mat = session_data.camera_params['cam_mat']

  # Create output folder for current session
  subject_opt_folder = osp.join(opt_folder, session)
  os.makedirs(subject_opt_folder, exist_ok=True)

  # Create hdf datasets
  hdf_file = h5py.File(osp.join(subject_opt_folder, f'annot.h5'), 'w')
  face_gaze = hdf_file.create_dataset(
    'face-gaze', shape=(max_n_samples, 3),
    dtype=np.float32, chunks=(1, 3),
    maxshape=(max_n_samples, 3),
  )
  face_pose = hdf_file.create_dataset(
    'face-pose', shape=(max_n_samples, 3),
    dtype=np.float32, chunks=(1, 3),
    maxshape=(max_n_samples, 3),
  )
  # Create face patch folders
  face_folder = osp.join(subject_opt_folder, 'face')
  os.makedirs(face_folder, exist_ok=True)

  # Open video capture, set frame offset
  record_path = osp.join(session_folder, 'rgb_vga.mov')
  capture = cv2.VideoCapture(record_path, cv2.CAP_ANY)
  capture.set(cv2.CAP_PROP_POS_FRAMES, SAMPLE_OFFSET)

  # Process gaze data for current session
  normalizer = MpiiDataNormalizer(960, (448, 448), distance=300)
  cam2nor = Camera2Normal(cam_mat, normalizer)
  pose_estim = HeadPoseEstimator(cam_mat)
  n_samples = 0 # Number of final samples

  with FaceBoundingBox(p_detection=0.8, p_suppression=0.2) as face_bbox:
    landmarker = SparseFaceLandmarks(width_expand=1.6, image_size=120)

    for idx in range(max_n_samples):
      ret = capture.grab()
      if not ret: continue

      if idx % SAMPLE_EVERY_N_FRAMES != 0:
        continue  # Skip intermediate frames

      ret, image = capture.retrieve()
      if not ret: continue

      if not session_data.check_index(idx):
        continue  # Skip those with missing data

      bboxes = face_bbox.process(image, bgr2rgb=True)
      if bboxes is None: continue

      _, ldmks_2d = landmarker.process(image, bboxes[0], bgr2rgb=True)

      landmarks_2d = ldmks_2d[[
        36, # reye, outer
        39, # reye, inner
        42, # leye, inner
        45, # leye, outer
        60, # mouth, rc
        64, # mouth, lc
      ]]
      hr, ht = pose_estim.estimate(landmarks_2d)
      hR = cv2.Rodrigues(hr)[0]
      landmarks_3d = np.dot(hR, pose_estim.face_model.T) + ht
      fe = calculate_face_center(landmarks_3d)

      tgt = convert_gaze_target_coords(session_data, frame_id=idx)
      img_f, gaze_f, pose_f = cam2nor.normalize_data(fe, hR, image, tgt)

      cv2.imwrite(
        osp.join(face_folder, f'{n_samples:04d}.jpg'),
        cv2.resize(img_f, (224, 224), interpolation=cv2.INTER_CUBIC),
        [cv2.IMWRITE_JPEG_QUALITY, 100],
      )
      face_gaze[n_samples] = gaze_f
      face_pose[n_samples] = pose_f

      n_samples = n_samples + 1

  face_gaze.resize(n_samples, axis=0)
  face_pose.resize(n_samples, axis=0)

  # Close hdf file
  hdf_file.close()
  # Log processing result
  rt_logger.info(f'processed: session {session}, {n_samples} samples')

def process_tasks(dataset_path, opt_folder):
  sessions_folder = osp.abspath(osp.join(dataset_path, 'Data'))
  sessions = [
    f for f in os.listdir(sessions_folder)
    if osp.isdir(osp.join(sessions_folder, f))
  ]
  non_ft_sessions = [s for s in sessions if not 'FT' in s]

  functional_tasks = []

  for session in non_ft_sessions:
    args = (sessions_folder, session, opt_folder)
    task = FunctionalTask(process_session, *args)
    functional_tasks.append(task)

  return functional_tasks


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'idiap-eyediap dataset: "{dataset_path}"')

  data_folder = ScriptEnv.data_path('idiap-eyediap')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(dataset_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for EyeDiap dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted EyeDiap dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
