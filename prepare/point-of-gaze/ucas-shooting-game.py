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


rt_logger = runtime_logger(
  name='ucas-shooting-game',
  log_file=ScriptEnv.log_path('prepare-ucas-shooting-game.log'),
)


def load_json_data(json_file: str):
  with open(json_file, 'r') as file:
    return json.load(file)

def save_json_data(json_data: dict, json_file: str):
  with open(json_file, 'w') as file:
    json.dump(json_data, file, indent=2)

def load_inlier_samples(recording_folder):
  samples_dict = load_json_data(osp.join(recording_folder, 'labels', 'samples.json'))

  samples = []  # Filter for inlier samples
  for image_name, sample_info in samples_dict.items():
    if not sample_info['inlier']: continue
    sample = dict(image=image_name, label=sample_info['target_xy'])
    samples.append(sample)

  return samples

def adjust_image(image: np.ndarray, src_res: tuple, tgt_res: tuple, resize: bool = True):
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

  if resize:
    dsize = (tgt_res[1], tgt_res[0])
    image = cv2.resize(image, dsize, interpolation=cv2.INTER_CUBIC)

  return image

def adjust_mesh(mesh: np.ndarray, src_res: tuple, tgt_res: tuple):
  src_h, src_w = src_res
  tgt_h, tgt_w = tgt_res

  src_asp = src_w / src_h
  tgt_asp = tgt_w / tgt_h

  if tgt_asp > src_asp:
    rescale_h = int(src_w / tgt_asp)
    padding_h = (src_h - rescale_h) // 2
    mesh[:, 1] -= padding_h
    mesh /= rescale_h / tgt_h
  if tgt_asp < src_asp:
    rescale_w = int(src_h * tgt_asp)
    padding_w = (src_w - rescale_w) // 2
    mesh[:, 0] -= padding_w
    mesh /= rescale_w / tgt_w

  return mesh

def process_recording(dataset_path, recording, opt_folder):
  # Load annotations for current recording
  recording_folder = osp.join(dataset_path, 'recordings', recording)
  samples = load_inlier_samples(recording_folder)
  if len(samples) == 0:
    rt_logger.warning(f'skipped: recording {recording}, {len(samples)} samples')
    return  # Skip recording with no inlier samples

  # Create output folder for current recording
  recording_opt_folder = osp.join(opt_folder, recording)
  os.makedirs(recording_opt_folder, exist_ok=True)

  # Create hdf datasets
  hdf_file = h5py.File(osp.join(recording_opt_folder, f'annot.h5'), 'w')
  name = hdf_file.create_dataset(
    'name', shape=len(samples),
    dtype=h5py.string_dtype(length=32),
    chunks=1, maxshape=len(samples),
  )
  gaze = hdf_file.create_dataset(
    'gaze', shape=(len(samples), 2),
    dtype=np.float32, chunks=(1, 2),
    maxshape=(len(samples), 2),
  )
  ldmk = hdf_file.create_dataset(
    'ldmk', shape=(len(samples), 478, 2),
    dtype=np.float32, chunks=(1, 478, 2),
    maxshape=(len(samples), 478, 2),
  )
  # Create images folders
  images_folder = osp.join(recording_opt_folder, 'images')
  os.makedirs(images_folder, exist_ok=True)

  # Process gaze data for current recording
  for idx in range(len(samples)):
    sample = samples[idx] # Sample dict

    image_path = osp.join(recording_folder, 'images', sample['image'])
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    mesh_name = sample['image'].replace('.jpg', '.npy')
    mesh_path = osp.join(recording_folder, 'meshes', mesh_name)
    mesh = np.load(mesh_path)

    image_h, image_w, _ = image.shape

    image = adjust_image(image, [image_h, image_w], [480, 640])
    mesh = adjust_mesh(mesh, [image_h, image_w], [480, 640])

    cv2.imwrite(
      osp.join(images_folder, sample['image']),
      image,
      [cv2.IMWRITE_JPEG_QUALITY, 100],
    )

    name[idx] = sample['image']
    gaze[idx] = sample['label']
    ldmk[idx] = mesh

  # Close hdf file and tar file
  hdf_file.close()

  # Save metadata about recording
  save_json_data(dict(
    counts=len(samples),
  ), osp.join(recording_opt_folder, 'meta.json'))

  # Log processing result
  rt_logger.info(f'processed: recording {recording}, {len(samples)} samples')

def process_tasks(dataset_path, opt_folder):
  dataset_path = osp.abspath(dataset_path)
  recordings = [
    f for f in os.listdir(osp.join(dataset_path, 'recordings'))
    if osp.isdir(osp.join(dataset_path, 'recordings', f))
  ]

  functional_tasks = []

  split_opt_folder = osp.join(opt_folder, osp.basename(dataset_path))
  for recording in recordings:
    args = (dataset_path, recording, split_opt_folder)
    task = FunctionalTask(process_recording, *args)
    functional_tasks.append(task)

  return functional_tasks


def main_procedure(cmdargs: argparse.Namespace):
  dataset_path = osp.abspath(cmdargs.dataset_path)
  rt_logger.info(f'ucas-shooting-game dataset: "{dataset_path}"')

  data_folder = ScriptEnv.data_path('ucas-shooting-game')
  os.makedirs(data_folder, exist_ok=True)
  rt_logger.info(f'processed data: "{data_folder}"')

  tasks = process_tasks(dataset_path, data_folder)
  executor = futures.ProcessPoolExecutor(cmdargs.max_workers)
  run_parallel(executor, tasks, rt_logger)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='prepare data for ShootingGame dataset.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the ShootingGame dataset.',
  )
  parser.add_argument(
    '--max-workers', type=int, default=None,
    help='Maximum number of processes in the process pool.',
  )

  main_procedure(parser.parse_args())
