# For archived GazeCapture dataset, see "prepare/mics/mit-gaze-capture-archive.py"

from opengaze.utils import FaceLandmarks
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
import shutil
import tarfile


rt_logger = runtime_logger(
  name='mit-gaze-capture',
  log_file=ScriptEnv.log_path('prepare-mit-gaze-capture.log'),
)


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
  gaze = hdf_file.create_dataset(
    'gaze', shape=(max_n_samples, 2),
    dtype=np.float32, chunks=(1, 2),
    maxshape=(max_n_samples, 2),
  )
  ldmk = hdf_file.create_dataset(
    'ldmk', shape=(max_n_samples, 478, 2),
    dtype=np.float32, chunks=(1, 478, 2),
    maxshape=(max_n_samples, 478, 2),
  )
  # Create images folders
  images_folder = osp.join(subject_opt_folder, 'images')
  os.makedirs(images_folder, exist_ok=True)

  # Process gaze data for current subject
  n_samples = 0 # Number of final samples

  landmarker = FaceLandmarks(p_detection=0.6, p_presence=0.6)
  tarball = tarfile.open(osp.join(subject_folder, 'images.tar.gz'), 'r:gz')
  with landmarker, tarball:
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

      landmarks = landmarker.process(img, bgr2rgb=True)
      if landmarks is None: continue

      cv2.imwrite(
        osp.join(images_folder, sample['image']),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 100],
      )

      name[n_samples] = sample['image']
      gaze[n_samples] = pog
      ldmk[n_samples] = landmarks

      n_samples = n_samples + 1

  name.resize(n_samples, axis=0)
  gaze.resize(n_samples, axis=0)
  ldmk.resize(n_samples, axis=0)

  # Close hdf file and tar file
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
