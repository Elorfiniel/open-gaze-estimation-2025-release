from opengaze.utils import MpiiDataNormalizer
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


rt_logger = runtime_logger(
  name='ucas-synthgaze',
  log_file=ScriptEnv.log_path('prepare-ucas-synthgaze.log'),
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
  def __init__(self, cam_mat: np.ndarray):
    cam_dist = np.zeros(shape=(5, 1), dtype=np.float64)

    super().__init__(cam_mat=cam_mat, cam_dist=cam_dist)

    mpii_model_path = ScriptEnv.resource_path('face-models/mpiigaze-generic.mat')
    face_model = sio.loadmat(mpii_model_path)['model'].T
    self.face_model = face_model.reshape((6, 3)).astype(np.float32)

  def estimate(self, landmarks_2d: np.ndarray):
    return super().estimate(self.face_model, landmarks_2d)


def load_json_file(json_file: str, **kwargs):
  with open(json_file, 'r', encoding='utf-8') as file:
    json_data = json.load(file, **kwargs)
  return json_data

def load_sample_counts(subject_folder: str):
  hparams_path = osp.join(subject_folder, 'hparams.json')
  hparams = load_json_file(hparams_path)
  n_targets = hparams['num_gaze_targets']
  n_views = hparams['num_view_per_target']
  return n_targets, n_views

def load_target_annots(target_folder: str):
  annots_dict = load_json_file(osp.join(target_folder, 'annots.json'))
  extrinsic_list = np.load(osp.join(target_folder, 'annots', 'extrinsics.npy'))
  vertex_dict = np.load(osp.join(target_folder, 'annots', 'vertices.npz'))
  visibility_dict = np.load(osp.join(target_folder, 'annots', 'visibility.npz'))
  return annots_dict, extrinsic_list, vertex_dict, visibility_dict

def load_pseudo_face_model(vertex_dict: dict):
  # Face Model: use the same convention as ETH-XGaze dataset
  landmarks = vertex_dict['face_mesh_dense'][[
    388, 387, # reye, outer
    263,      # reye, inner
    39,       # leye, inner
    164, 163, # leye, outer
    465,      # mouth, rc
    461,      # mouth, lc
  ]]
  face_model = np.stack([
    0.5 * (landmarks[0] + landmarks[1]),
    landmarks[2],
    landmarks[3],
    0.5 * (landmarks[4] + landmarks[5]),
    landmarks[6],
    landmarks[7],
  ], axis=0, dtype=np.float32)
  return face_model

def to_opencv_camera_coords(points_3d: np.ndarray, extrinsic: np.ndarray):
  # Convert 3D points from world coordinates to camera coordinates
  world_coords = np.hstack([points_3d, np.ones((len(points_3d), 1))])
  local_coords = np.linalg.inv(extrinsic) @ world_coords.T
  local_coords = local_coords[:3, :] / local_coords[3, :]

  # Axis-Mapping from Blender to OpenCV: (X, Y, Z) -> (X, -Y, -Z)
  local_coords = np.array([+1, -1, -1]).reshape(3, 1) * local_coords

  return local_coords.T

def project_points(points_3d: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray):
  local_coords = to_opencv_camera_coords(points_3d, extrinsic)
  image_coords = np.dot(intrinsic, local_coords.T)
  image_coords = image_coords[:2, :] / image_coords[2, :]
  return image_coords.T

def calculate_face_center(landmarks_3d: np.ndarray):
  # Use convention of ETH-XGaze data normalization

  eyes_center = np.mean(landmarks_3d[:, 0:4], axis=1)
  mouth_center = np.mean(landmarks_3d[:, 4:6], axis=1)
  face_center = 0.5 * (eyes_center + mouth_center)

  return face_center

def convert_gaze_target_coords(gaze_target: np.ndarray, extrinsic: np.ndarray):
  local_coords = to_opencv_camera_coords(gaze_target.reshape((1, 3)), extrinsic)
  return 1e3 * local_coords.reshape((3, ))  # From meter to millimeter

def process_target(subject_folder: str, target_idx: int, n_views: int, subject_opt_folder: str):
  target_folder = osp.join(subject_folder, f'target-{target_idx + 1:06d}')

  # Load annotations for current target
  annots_dict, extrinsic_list, vertex_dict, _ = load_target_annots(target_folder)
  # Extract annotations for current target
  intrinsic_actual = np.array(annots_dict['intrinsic']['actual'])
  intrinsic_render = np.array(annots_dict['intrinsic']['render'])
  gaze_target = np.array(annots_dict['gaze_info']['target'])

  # Process gaze data for current target
  normalizer = MpiiDataNormalizer(960, (448, 448), distance=300)
  cam2nor = Camera2Normal(intrinsic_actual, normalizer)
  pose_estim = HeadPoseEstimator(intrinsic_actual)
  face_model = load_pseudo_face_model(vertex_dict)

  gazes = np.zeros(shape=(n_views, 3), dtype=np.float32)
  poses = np.zeros(shape=(n_views, 3), dtype=np.float32)

  for view_idx in range(n_views):
    # Extract annotations for the view
    extrinsic = np.array(extrinsic_list[view_idx])

    # Process gaze data for current view
    image_path = osp.join(target_folder, 'images', f'{view_idx + 1:04d}.jpg')
    image = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    M = intrinsic_actual @ np.linalg.inv(intrinsic_render)
    dsize = [2 * int(s) for s in intrinsic_actual[:2, 2]]
    image = cv2.warpPerspective(image, M, dsize, flags=cv2.INTER_CUBIC)

    landmarks_2d = project_points(face_model, intrinsic_actual, extrinsic)

    hr, ht = pose_estim.estimate(landmarks_2d)
    hR = cv2.Rodrigues(hr)[0]
    landmarks_3d = np.dot(hR, pose_estim.face_model.T) + ht
    fe = calculate_face_center(landmarks_3d)

    tgt = convert_gaze_target_coords(gaze_target, extrinsic)
    img_f, gaze_f, pose_f = cam2nor.normalize_data(fe, hR, image, tgt)

    global_idx = target_idx * n_views + view_idx
    cv2.imwrite(
      osp.join(subject_opt_folder, 'face', f'{global_idx:04d}.jpg'),
      cv2.resize(img_f, (224, 224), interpolation=cv2.INTER_CUBIC),
      [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    gazes[view_idx] = gaze_f; poses[view_idx] = pose_f

  return gazes, poses

def process_subject(subjects_folder, subject, opt_folder):
  # Create output folder for current subject
  subject_opt_folder = osp.join(opt_folder, subject)
  os.makedirs(subject_opt_folder, exist_ok=True)

  # Load sample counts for current subject
  subject_folder = osp.join(subjects_folder, subject)
  n_targets, n_views = load_sample_counts(subject_folder)

  # Create hdf datasets
  n_samples = n_targets * n_views
  hdf_file = h5py.File(osp.join(subject_opt_folder, f'annot.h5'), 'w')
  face_gaze = hdf_file.create_dataset(
    'face-gaze', shape=(n_samples, 3),
    dtype=np.float32, chunks=(1, 3),
  )
  face_pose = hdf_file.create_dataset(
    'face-pose', shape=(n_samples, 3),
    dtype=np.float32, chunks=(1, 3),
  )
  # Create face patch folders
  face_folder = osp.join(subject_opt_folder, 'face')
  os.makedirs(face_folder, exist_ok=True)

  # Process gaze data for current subject
  for target_idx in range(n_targets):
    gazes, poses = process_target(subject_folder, target_idx, n_views, subject_opt_folder)

    chunk_start = target_idx * n_views
    chunk_end = chunk_start + n_views

    face_gaze[chunk_start:chunk_end] = gazes
    face_pose[chunk_start:chunk_end] = poses

  # Close hdf file
  hdf_file.close()
  # Log processing result
  rt_logger.info(f'processed: subject {subject}, {n_samples} samples')

def process_tasks(dataset_path, opt_folder):
  subjects_folder = osp.abspath(dataset_path)
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
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'ucas-synthgaze dataset: "{dataset_path}"')

  data_folder = ScriptEnv.data_path('ucas-synthgaze')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(dataset_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for SynthGaze dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted SynthGaze dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
