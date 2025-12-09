from opengaze.utils import MpiiDataNormalizer, FaceBoundingBox, SparseFaceLandmarks
from opengaze.utils.geom import PoseEstimator
from opengaze.utils.euler import gaze_3d_2d_a, pose_3d_2d_a
from opengaze.runtime.scripts import ScriptEnv
from opengaze.runtime.log import runtime_logger
from opengaze.runtime.parallel import FunctionalTask, run_parallel

import argparse
import concurrent.futures as futures
import cv2
import h5py
import math
import multiprocessing as mp
import numpy as np
import os
import os.path as osp
import scipy.io as sio


rt_logger = runtime_logger(
  name='mit-gaze-360',
  log_file=ScriptEnv.log_path('prepare-mit-gaze-360.log'),
)


MAX_NUM_SAMPLES, N_SAMPLES_PER_TASK = 200000, 2000


class SampleLoader:
  def __init__(self, dataset_path: str):
    self.metadata = sio.loadmat(osp.join(dataset_path, 'metadata.mat'))
    self.n_samples = self.metadata['frame'].shape[1]

  def _concat_image_path(self, index: int):
    recording = self.metadata['recordings'][
      0, self.metadata['recording'][0, index]
    ].item()
    identity = self.metadata['person_identity'][0, index]
    frame = self.metadata['frame'][0, index]
    return osp.join('imgs', recording, 'head', f'{identity:06d}', f'{frame:06d}.jpg')

  def _load_split(self, index: int):
    return self.metadata['splits'][
      0, self.metadata['split'][0, index]
    ].item()

  def _load_bboxes(self, index: int):
    bboxes = dict(
      head_bbox=self.metadata['person_head_bbox'][index],
      face_bbox=self.metadata['person_face_bbox'][index],
      reye_bbox=self.metadata['person_eye_right_bbox'][index],
      leye_bbox=self.metadata['person_eye_left_bbox'][index],
    )
    return {k:v.tolist() for k, v in bboxes.items()}

  def _load_gaze_vector(self, index: int):
    return self.metadata['gaze_dir'][index].tolist()

  def load_metadata(self, index: int):
    assert 0 <= index < self.n_samples
    return dict(
      image_path=self._concat_image_path(index),
      split=self._load_split(index),
      **self._load_bboxes(index),
      gaze_vector=self._load_gaze_vector(index),
    )

class Camera2Normal:
  def __init__(self, cam_mat: np.ndarray, normalizer: MpiiDataNormalizer):
    self.cam_mat = cam_mat
    self.normalizer = normalizer

  def _unit_vector(self, v: np.ndarray):
    return v / np.linalg.norm(v)

  def normalize_data(self, look_at, R1, image, gaze):
    Kv, S, R2, W = self.normalizer.normalize_matrices(look_at, R1, self.cam_mat)
    warp = self.normalizer.warp_image(image, W)

    # Revisiting Data Normalization: discard the scaling component `S_x`
    gaze = self._unit_vector(np.dot(R2, gaze))
    pose = cv2.Rodrigues(np.dot(R2, R1))[0].reshape((3, ))

    return warp, gaze, pose

class HeadPoseEstimator(PoseEstimator):
  def __init__(self, cam_mat: np.ndarray, cam_dist: np.ndarray = None):
    super().__init__(cam_mat=cam_mat, cam_dist=cam_dist)

    mpii_model_path = ScriptEnv.resource_path('face-models/mpiigaze-generic.mat')
    face_model = sio.loadmat(mpii_model_path)['model'].T
    self.face_model = face_model.reshape((6, 3)).astype(np.float32)

  def estimate(self, landmarks_2d: np.ndarray):
    return super().estimate(self.face_model, landmarks_2d)


def filter_sample_metadata(metadata: dict):
  if np.all(metadata['face_bbox'] == np.array([-1, -1, -1, -1])):
    return False  # No valid face detected

  if metadata['gaze_vector'][2] >= 0.0:
    return False  # Not a frontal view

  return True

def filter_sample_image(image: np.ndarray):
  MIN_IMAGE_H, MIN_IMAGE_W = 120, 120

  image_h, image_w, _ = image.shape

  if image_h < MIN_IMAGE_H or image_w < MIN_IMAGE_W:
    return False  # Small image (unstable)

  return True

def estimate_camera_intrinsics(image: np.ndarray, ldmks_3d: np.ndarray, ldmks_2d: np.ndarray):
  image_h, image_w, _ = image.shape

  rms, mtx, dist, _, _ = cv2.calibrateCamera(
    [ldmks_3d], [ldmks_2d], (image_w, image_h),
    cameraMatrix=np.array([
      [1000.0, 0.0, image_w / 2.0],
      [0.0, 1000.0, image_h / 2.0],
      [0.0, 0.0, 1.0],
    ], dtype=np.float64),
    distCoeffs=np.zeros(shape=(5, 1), dtype=np.float64),
    flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO,
  )

  return mtx, dist

def calculate_face_center(landmarks_3d: np.ndarray):
  # Use convention of ETH-XGaze data normalization

  eyes_center = np.mean(landmarks_3d[:, 0:4], axis=1)
  mouth_center = np.mean(landmarks_3d[:, 4:6], axis=1)
  face_center = 0.5 * (eyes_center + mouth_center)

  return face_center

def convert_gaze_coords(gaze: np.ndarray):
  return np.array([-gaze[0], -gaze[1], gaze[2]])

def filter_sample_data(pose: np.ndarray, gaze: np.ndarray):
  MAX_POSE_P, MAX_POSE_Y = 80, 80 # Derived from trial-and-error
  MAX_GAZE_P, MAX_GAZE_Y = 80, 80 # Derived from trial-and-error

  pose_p, pose_y = [math.degrees(x) for x in pose_3d_2d_a(*pose.tolist())]
  if abs(pose_p) >= MAX_POSE_P or abs(pose_y) >= MAX_POSE_Y:
    return False  # Large head pose (unstable)

  gaze_p, gaze_y = [math.degrees(x) for x in gaze_3d_2d_a(*gaze.tolist())]
  if abs(gaze_p) >= MAX_GAZE_P or abs(gaze_y) >= MAX_GAZE_Y:
    return False  # Large gaze angle (unstable)

  return True

def process_frames(dataset_path, start_index, n_samples, opt_folder):
  # Initialize sample loader to load sample metadata
  sample_loader = SampleLoader(dataset_path=dataset_path)

  # Create in-memory store for processed data
  partial_data = dict(train=dict(), val=dict(), test=dict(), unused=dict())

  # Process gaze data for current batch of samples
  n_processed = 0 # Number of processed samples

  with FaceBoundingBox(p_detection=0.8, p_suppression=0.2) as face_bbox:
    landmarker = SparseFaceLandmarks(width_expand=1.6, image_size=120)
    normalizer = MpiiDataNormalizer(960, (448, 448), distance=300)

    end_index = min(sample_loader.n_samples, start_index + n_samples)
    for idx in range(start_index, end_index):
      metadata = sample_loader.load_metadata(idx)
      if not filter_sample_metadata(metadata): continue

      img = cv2.imread(osp.join(dataset_path, metadata['image_path']), cv2.IMREAD_UNCHANGED)
      if not filter_sample_image(img): continue

      bboxes = face_bbox.process(img, bgr2rgb=True)
      if bboxes is None: continue

      ldmks_3d, ldmks_2d = landmarker.process(img, bboxes[0], bgr2rgb=True)
      mtx, dist = estimate_camera_intrinsics(img, 1e3 * ldmks_3d, ldmks_2d)

      cam2nor = Camera2Normal(mtx, normalizer)
      pose_estim = HeadPoseEstimator(mtx, dist)

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

      gaze = convert_gaze_coords(metadata['gaze_vector'])
      img_f, gaze_f, pose_f = cam2nor.normalize_data(fe, hR, img, gaze)
      if not filter_sample_data(pose_f, gaze_f): continue

      cv2.imwrite(
        osp.join(opt_folder, metadata['split'], 'face', f'{idx:06d}.jpg'),
        cv2.resize(img_f, (224, 224), interpolation=cv2.INTER_CUBIC),
        [cv2.IMWRITE_JPEG_QUALITY, 100],
      )
      partial_data[metadata['split']][idx] = dict(gaze=gaze_f, pose=pose_f)

      n_processed = n_processed + 1

  # Log processing result
  rt_logger.info(f'processed samples: {start_index} => {end_index}, {n_processed} samples')
  # Return partial dataset to the main process
  return partial_data

def process_tasks(dataset_path, opt_folder):
  # Create output folder for processed samples
  for folder_name in ['train', 'val', 'test', 'unused']:
    folder_path = osp.join(opt_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    face_folder = osp.join(folder_path, 'face')
    os.makedirs(face_folder, exist_ok=True)

  # Collect processed data from distributed tasks
  combined_data = dict(train=dict(), val=dict(), test=dict(), unused=dict())
  shared_lock = mp.Lock()

  def done_fn(future: futures.Future):
    partial_data = future.result()
    with shared_lock:
      for split in combined_data:
        combined_data[split].update(partial_data[split])

  functional_tasks = []

  for task_index in range(MAX_NUM_SAMPLES // N_SAMPLES_PER_TASK):
    start_index = task_index * N_SAMPLES_PER_TASK
    args = (dataset_path, start_index, N_SAMPLES_PER_TASK, opt_folder)
    task = FunctionalTask(process_frames, *args, done_fn=done_fn)
    functional_tasks.append(task)

  return functional_tasks, combined_data

def post_process_combined_data(opt_folder, combined_data: dict):
  # Create hdf datasets for each split
  for split in combined_data:
    hdf_file = h5py.File(osp.join(opt_folder, split, f'annot.h5'), 'w')
    name = hdf_file.create_dataset(
      'name', shape=len(combined_data[split]),
      dtype=h5py.string_dtype(length=32),
    )
    face_gaze = hdf_file.create_dataset(
      'face-gaze', shape=(len(combined_data[split]), 3),
      dtype=np.float32, chunks=(1, 3),
    )
    face_pose = hdf_file.create_dataset(
      'face-pose', shape=(len(combined_data[split]), 3),
      dtype=np.float32, chunks=(1, 3),
    )
    for it, idx in enumerate(sorted(combined_data[split])):
      sample_data = combined_data[split][idx]
      name[it] = f'{idx:06d}.jpg'
      face_gaze[it] = sample_data['gaze']
      face_pose[it] = sample_data['pose']

    hdf_file.close()


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'mit-gaze-360 dataset: "{dataset_path}"')

  data_folder = ScriptEnv.data_path('mit-gaze-360')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks, combined_data = process_tasks(dataset_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)
  post_process_combined_data(data_folder, combined_data)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for Gaze360 dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted Gaze360 dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
