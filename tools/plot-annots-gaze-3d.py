from opengaze.utils.euler import gaze_3d_2d_a, pose_3d_2d_a

from matplotlib.colors import PowerNorm

import argparse
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


def list_h5_annots(dataset_path: str):
  return glob.glob('**/*.h5', root_dir=dataset_path, recursive=True)

def gaze_to_pitch_yaw(gaze: np.ndarray):
  if gaze.shape[1] == 2:
    pitch = np.array(gaze[:, 0], dtype=np.float32)
    yaw = np.array(gaze[:, 1], dtype=np.float32)

  if gaze.shape[1] == 3:
    data = [gaze_3d_2d_a(*g.tolist()) for g in gaze]
    pitch = np.array([d[0] for d in data], dtype=np.float32)
    yaw = np.array([d[1] for d in data], dtype=np.float32)

  pitch, yaw = np.rad2deg(pitch), np.rad2deg(yaw)

  return pitch, yaw

def pose_to_pitch_yaw(pose: np.ndarray):
  if pose.shape[1] == 2:
    pitch = np.array(pose[:, 0], dtype=np.float32)
    yaw = np.array(pose[:, 1], dtype=np.float32)

  if pose.shape[1] == 3:
    data = [pose_3d_2d_a(*p.tolist()) for p in pose]
    pitch = np.array([d[0] for d in data], dtype=np.float32)
    yaw = np.array([d[1] for d in data], dtype=np.float32)

  pitch, yaw = np.rad2deg(pitch), np.rad2deg(yaw)

  return pitch, yaw

def hist_h5_annots(root: str, h5_file: str, gaze_bins: tuple, pose_bins: tuple):
  with h5py.File(osp.join(root, h5_file), 'r') as hdf_file:
    gaze_p, gaze_y = gaze_to_pitch_yaw(hdf_file['face-gaze'])
    pose_p, pose_y = pose_to_pitch_yaw(hdf_file['face-pose'])

  gaze_hist, gaze_xedges, gaze_yedges = np.histogram2d(gaze_p, gaze_y, bins=gaze_bins)
  pose_hist, pose_xedges, pose_yedges = np.histogram2d(pose_p, pose_y, bins=pose_bins)

  return dict(
    gaze=dict(hist=gaze_hist, xedges=gaze_xedges, yedges=gaze_yedges),
    pose=dict(hist=pose_hist, xedges=pose_xedges, yedges=pose_yedges),
  )

def merge_hist_dict(hist_dict: dict, merged_hist: dict):
  if not 'hist' in merged_hist:
    merged_hist['hist'] = hist_dict['hist']
    merged_hist['xedges'] = hist_dict['xedges']
    merged_hist['yedges'] = hist_dict['yedges']
  else:
    merged_hist['hist'] += hist_dict['hist']

def accumulate_hists(root: str, h5_files: list, gaze_bins: tuple, pose_bins: tuple):
  gaze_hist, pose_hist = dict(), dict()

  for h5_file in h5_files:
    hist_dict = hist_h5_annots(root, h5_file, gaze_bins, pose_bins)
    merge_hist_dict(hist_dict['gaze'], gaze_hist)
    merge_hist_dict(hist_dict['pose'], pose_hist)

  return gaze_hist, pose_hist

def plot_hist(ax, hist_dict: dict, gamma: float = 0.0, **kwargs):
  if gamma > 0.0:
    kwargs['norm'] = PowerNorm(gamma=gamma, clip=True)

  ax.pcolormesh(
    hist_dict['xedges'], hist_dict['yedges'],
    hist_dict['hist'], cmap='jet', **kwargs,
  )

  ax.set_ylabel('Pitch [°]')
  ax.set_xlabel('Yaw [°]')
  ax.grid(True, alpha=0.2)
  ax.set_aspect('equal')

def adjust_figure(fig):
  fig.subplots_adjust(
    left=0.16, right=0.96,
    bottom=0.08, top=0.96,
    wspace=0.20, hspace=0.24,
  )


def main_procedure(cmdargs: argparse.Namespace):
  gaze_hist, pose_hist = accumulate_hists(
    root=cmdargs.dataset_path,
    h5_files=list_h5_annots(cmdargs.dataset_path),
    gaze_bins=np.linspace(-100, 100, 400),
    pose_bins=np.linspace(-100, 100, 400),
  )

  fig, (ax_gaze, ax_pose) = plt.subplots(2, 1, figsize=(4, 7))
  plot_hist(ax_gaze, gaze_hist, gamma=cmdargs.gaze_gamma)
  plot_hist(ax_pose, pose_hist, gamma=cmdargs.pose_gamma)

  fig.tight_layout()
  adjust_figure(fig)

  plt.savefig(f'{cmdargs.plot_name}.png')



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='plot annotations for gaze-3d datasets.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the prepared gaze dataset.',
  )

  parser.add_argument(
    '--gaze-gamma', type=float, default=0.0,
    help='Gamma value for the gaze plot normalization.',
  )
  parser.add_argument(
    '--pose-gamma', type=float, default=0.0,
    help='Gamma value for the pose plot normalization.',
  )
  parser.add_argument(
    '--plot-name', type=str, required=True,
    help='Name of the output annots plot.',
  )

  main_procedure(parser.parse_args())
