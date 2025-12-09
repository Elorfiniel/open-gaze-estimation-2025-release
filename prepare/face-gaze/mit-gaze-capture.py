# For archived GazeCapture dataset, see "prepare/mics/mit-gaze-capture-archive.py"

from opengaze.utils import MpiiDataNormalizer, FaceBoundingBox, SparseFaceLandmarks
from opengaze.utils.geom import PoseEstimator
from opengaze.runtime.scripts import ScriptEnv
from opengaze.runtime.log import runtime_logger
from opengaze.runtime.parallel import FunctionalTask, run_parallel

import argparse
import concurrent.futures as futures
import cv2
import h5py
import json
import numpy as np
import os
import os.path as osp
import scipy.io as sio
import shutil
import tarfile


rt_logger = runtime_logger(
  name='mit-gaze-capture',
  log_file=ScriptEnv.log_path('prepare-mit-gaze-capture.log'),
)


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
  def __init__(self, cam_mat: np.ndarray, cam_dist: np.ndarray = None):
    super().__init__(cam_mat=cam_mat, cam_dist=cam_dist)

    mpii_model_path = ScriptEnv.resource_path('face-models/mpiigaze-generic.mat')
    face_model = sio.loadmat(mpii_model_path)['model'].T
    self.face_model = face_model.reshape((6, 3)).astype(np.float32)

  def estimate(self, landmarks_2d: np.ndarray):
    return super().estimate(self.face_model, landmarks_2d)


def load_json_data(json_file: str):
  with open(json_file, 'r') as file:
    return json.load(file)

def save_json_data(json_data: dict, json_file: str):
  with open(json_file, 'w') as file:
    json.dump(json_data, file, indent=2)

def load_gc_annot_subject(subject_folder):
  metadata = load_json_data(osp.join(subject_folder, 'metadata.json'))

  with h5py.File(osp.join(subject_folder, 'labels.h5'), 'r') as hdf_file:
    image_names = hdf_file['labels']['image_names']
    gaze_labels_x = hdf_file['labels']['dot_x_cam']
    gaze_labels_y = hdf_file['labels']['dot_y_cam']

    face_valid = np.asarray(hdf_file['labels']['face_valid'], dtype=bool)
    reye_valid = np.asarray(hdf_file['labels']['reye_valid'], dtype=bool)
    leye_valid = np.asarray(hdf_file['labels']['leye_valid'], dtype=bool)

    label_file = np.array([
      z for z in zip(image_names, gaze_labels_x, gaze_labels_y)
    ], dtype=[('image', 'U32'), ('gx_cm', 'f4'), ('gy_cm', 'f4')])
    image_valid = face_valid & reye_valid & leye_valid

  valid_labels = label_file[image_valid]

  return metadata, valid_labels

def estimate_camera_intrinsics(image: np.ndarray, ldmks_3d: np.ndarray, ldmks_2d: np.ndarray):
  image_h, image_w, _ = image.shape

  rms, mtx, dist, _, _ = cv2.calibrateCamera(
    [ldmks_3d], [ldmks_2d], (image_w, image_h),
    cameraMatrix=np.array([
      [640.0, 0.0, image_w / 2.0],
      [0.0, 640.0, image_h / 2.0],
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

def calculate_pseudo_gaze_target(pog: np.ndarray):
  # Coordinate System: opencv pinhole camera model
  # Unit Length: millimeters (eg.  MPII dataset)
  return 10.0 * np.array([-pog[0], -pog[1], 0.0])

def process_subject(subjects_folder, subject, opt_folder):
  # Load annotations for current subject
  subject_folder = osp.join(subjects_folder, subject)
  metadata, valid_labels = load_gc_annot_subject(subject_folder)

  # Create output folder for current subject
  subject_opt_folder = osp.join(opt_folder, subject)
  os.makedirs(subject_opt_folder, exist_ok=True)

  # Create hdf datasets
  hdf_file = h5py.File(osp.join(subject_opt_folder, f'annot.h5'), 'w')
  max_n_samples = max(len(valid_labels), 1)
  name = hdf_file.create_dataset(
    'name', shape=max_n_samples,
    dtype=h5py.string_dtype(length=32),
    chunks=1, maxshape=max_n_samples,
  )
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

  # Process gaze data for current subject
  n_samples = 0 # Number of final samples

  face_bbox = FaceBoundingBox(p_detection=0.8, p_suppression=0.2)
  tarball = tarfile.open(osp.join(subject_folder, 'images.tar.gz'), 'r:gz')
  with face_bbox, tarball:
    landmarker = SparseFaceLandmarks(width_expand=1.6, image_size=120)
    normalizer = MpiiDataNormalizer(960, (448, 448), distance=300)

    for idx in range(len(valid_labels)):
      sample = valid_labels[idx]  # Numpy structured array

      try:  # Check if the image exists
        tarball.getmember(sample['image'])
      except KeyError: continue

      img_raw_data = tarball.extractfile(sample['image']).read()
      img = cv2.imdecode(
        np.frombuffer(img_raw_data, np.uint8),
        cv2.IMREAD_UNCHANGED,
      )
      pog = np.array(sample[['gx_cm', 'gy_cm']].tolist(), dtype=np.float32)

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

      tgt = calculate_pseudo_gaze_target(pog)
      img_f, gaze_f, pose_f = cam2nor.normalize_data(fe, hR, img, tgt)

      cv2.imwrite(
        osp.join(face_folder, sample['image']),
        cv2.resize(img_f, (224, 224), interpolation=cv2.INTER_CUBIC),
        [cv2.IMWRITE_JPEG_QUALITY, 100],
      )
      name[n_samples] = sample['image']
      face_gaze[n_samples] = gaze_f
      face_pose[n_samples] = pose_f

      n_samples = n_samples + 1

  name.resize(n_samples, axis=0)
  face_gaze.resize(n_samples, axis=0)
  face_pose.resize(n_samples, axis=0)

  # Close hdf file
  hdf_file.close()

  # Save metadata about subject
  save_json_data(dict(
    device=metadata['device'],
    split=metadata['split'],
    counts=n_samples,
  ), osp.join(subject_opt_folder, 'meta.json'))

  # Log processing result
  if n_samples == 0:
    rt_logger.warning(f'skipped: subject {subject}, {n_samples} samples')
    shutil.rmtree(subject_opt_folder, ignore_errors=True)
  else:
    rt_logger.info(f'processed: subject {subject}, {n_samples} samples')

def process_tasks(archive_path, opt_folder):
  subjects_folder = osp.abspath(archive_path)
  subjects = [
    f for f in os.listdir(subjects_folder)
    if osp.isdir(osp.join(subjects_folder, f))
  ]

  functional_tasks = []

  for subject in subjects:
    args = (subjects_folder, subject, opt_folder)
    task = FunctionalTask(process_subject, *args)
    functional_tasks.append(task)

  return functional_tasks


def main_procedure(cmdargs: argparse.Namespace):
  archive_path = osp.abspath(cmdargs.archive_path)
  rt_logger.info(f'mit-gaze-capture archive: "{archive_path}"')

  data_folder = ScriptEnv.data_path('mit-gaze-capture')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(archive_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for GazeCapture dataset.')

  parser.add_argument(
    '--archive-path', type=str, required=True,
    help='Path to the archived GazeCapture dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
