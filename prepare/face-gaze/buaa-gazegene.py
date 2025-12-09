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
import pickle
import re


rt_logger = runtime_logger(
  name='buaa-gazegene',
  log_file=ScriptEnv.log_path('prepare-buaa-gazegene.log'),
)


N_CAMERAS, N_FRAMES_PER_CAMERA = 9, 2000


def convert_head_pose(hR: np.ndarray):
  # Axis-Mapping from GazeGene to ETH-XGaze: (X, Y, Z) -> (X, -Y, -Z)
  hR = np.stack([hR[:, 0], -hR[:, 1], -hR[:, 2]], axis=1)
  return cv2.Rodrigues(hR)[0].reshape((3, ))


def load_gaze_annots(subjects_folder: str, subject_folder: str, camera_idx: int):
  annots_file = f'gaze_label_camera{camera_idx}.pkl'
  annots_path = osp.join(subjects_folder, subject_folder, 'labels', annots_file)

  with open(annots_path, 'rb') as file:
    annots = pickle.load(file)

  return annots

def process_subject(subjects_folder, subject_folder, opt_folder):
  # Load annotations for current subject
  subject_gaze_annots = {
    it:load_gaze_annots(subjects_folder, subject_folder, it)
    for it in range(N_CAMERAS)
  }

  # Create output folder for current subject
  subject = re.match(r'subject(\d+)', subject_folder).group(1)
  subject_opt_folder = osp.join(opt_folder, f'{int(subject):02d}')
  os.makedirs(subject_opt_folder, exist_ok=True)

  # Create hdf datasets
  n_samples = N_CAMERAS * N_FRAMES_PER_CAMERA
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
  for camera_idx in range(N_CAMERAS):
    images_folder = osp.join(subjects_folder, subject_folder, 'imgs', f'camera{camera_idx}')

    gaze_annots = subject_gaze_annots[camera_idx]

    for frame_idx in range(N_FRAMES_PER_CAMERA):
      flat_idx = camera_idx * N_FRAMES_PER_CAMERA + frame_idx

      img_file = f'{subject_folder}_{frame_idx:04d}.jpg'
      img = cv2.imread(osp.join(images_folder, img_file), cv2.IMREAD_UNCHANGED)
      cv2.imwrite(
        osp.join(face_folder, f'{flat_idx:05d}.jpg'),
        cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC),
        [cv2.IMWRITE_JPEG_QUALITY, 100],
      )
      face_gaze[flat_idx] = gaze_annots['gaze_C'][frame_idx]
      face_pose[flat_idx] = convert_head_pose(gaze_annots['head_R_mat'][frame_idx])

  # Close hdf file
  hdf_file.close()
  # Log processing result
  rt_logger.info(f'processed: subject {int(subject):02d}, {n_samples} samples')

def process_tasks(dataset_path, opt_folder):
  subjects_folder = osp.abspath(dataset_path)
  subject_folders = [
    f for f in os.listdir(subjects_folder)
    if osp.isdir(osp.join(subjects_folder, f))
  ]

  functional_tasks = []

  for subject_folder in subject_folders:
    args = (subjects_folder, subject_folder, opt_folder)
    task = FunctionalTask(process_subject, *args)
    functional_tasks.append(task)

  return functional_tasks


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'buaa-gazegene dataset: "{dataset_path}"')

  data_folder = ScriptEnv.data_path('buaa-gazegene')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(dataset_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for GazeGene dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted GazeGene dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
