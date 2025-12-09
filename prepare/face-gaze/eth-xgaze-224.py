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
import re


rt_logger = runtime_logger(
  name='eth-xgaze-224',
  log_file=ScriptEnv.log_path('prepare-eth-xgaze-224.log'),
)


def process_subject(subjects_folder, subject_hdf, opt_folder):
  # Load annotations for current subject
  subject_data = h5py.File(osp.join(subjects_folder, subject_hdf), 'r')

  # Create output folder for current subject
  subject = re.match(r'subject(\d+).h5', subject_hdf).group(1)
  subject_opt_folder = osp.join(opt_folder, subject)
  os.makedirs(subject_opt_folder, exist_ok=True)

  # Create hdf datasets
  n_samples = subject_data['frame_index'].len()
  hdf_file = h5py.File(osp.join(subject_opt_folder, f'annot.h5'), 'w')
  face_gaze = hdf_file.create_dataset(
    'face-gaze', shape=(n_samples, 2),
    dtype=np.float32, chunks=(1, 2),
  )
  face_pose = hdf_file.create_dataset(
    'face-pose', shape=(n_samples, 2),
    dtype=np.float32, chunks=(1, 2),
  )
  # Create face patch folders
  face_folder = osp.join(subject_opt_folder, 'face')
  os.makedirs(face_folder, exist_ok=True)

  # Process gaze data for current subject
  for idx in range(n_samples):
    face_img = np.array(subject_data['face_patch'][idx])
    cv2.imwrite(
      osp.join(face_folder, f'{idx:05d}.jpg'),
      face_img,
      [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    face_gaze[idx] = np.array(subject_data['face_gaze'][idx])
    face_pose[idx] = np.array(subject_data['face_head_pose'][idx])

  # Close hdf file
  hdf_file.close()
  # Log processing result
  rt_logger.info(f'processed: subject {subject}, {n_samples} samples')

def process_tasks(dataset_path, opt_folder):
  subjects_folder = osp.abspath(osp.join(dataset_path, 'train'))
  subject_hdfs = [
    f for f in os.listdir(subjects_folder)
    if osp.isfile(osp.join(subjects_folder, f))
  ]

  functional_tasks = []

  for subject_hdf in subject_hdfs:
    args = (subjects_folder, subject_hdf, opt_folder)
    task = FunctionalTask(process_subject, *args)
    functional_tasks.append(task)

  return functional_tasks


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'eth-xgaze-224 dataset: "{dataset_path}"')

  data_folder = ScriptEnv.data_path('eth-xgaze-224')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(dataset_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for ETH-XGaze dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the extracted ETH-XGaze dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
