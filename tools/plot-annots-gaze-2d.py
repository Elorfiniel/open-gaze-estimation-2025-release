from matplotlib.colors import PowerNorm
from typing import Callable

import argparse
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp


def load_json_data(json_file: str):
  with open(json_file, 'r') as file:
    return json.load(file)

def merge_hist_dict(hist_dict: dict, merged_hist: dict):
  if not 'hist' in merged_hist:
    merged_hist['hist'] = hist_dict['hist']
    merged_hist['xedges'] = hist_dict['xedges']
    merged_hist['yedges'] = hist_dict['yedges']
  else:
    merged_hist['hist'] += hist_dict['hist']

def hist_h5_annots(annot_file: str, gaze_bins: tuple):
  with h5py.File(annot_file, 'r') as hdf:
    gaze_x, gaze_y = np.array(hdf['gaze'], dtype=np.float32).T

  hist, xedges, yedges = np.histogram2d(gaze_x, gaze_y, bins=gaze_bins)

  return dict(hist=hist, xedges=xedges, yedges=yedges)

def accumulate_hists(root: str, filter_fn: Callable, gaze_bins: tuple):
  gaze_hist = dict()

  recordings = os.listdir(root)
  for recording in recordings:
    meta_file = osp.join(root, recording, 'meta.json')
    meta = load_json_data(meta_file)
    if not filter_fn(recording, meta): continue
    annot_file = osp.join(root, recording, 'annot.h5')
    hist_dict = hist_h5_annots(annot_file, gaze_bins)
    merge_hist_dict(hist_dict, gaze_hist)

  return gaze_hist

def plot_hist(ax, hist_dict: dict, gamma: float = 0.0, **kwargs):
  if gamma > 0.0:
    kwargs['norm'] = PowerNorm(gamma=gamma, clip=True)

  ax.pcolormesh(
    hist_dict['xedges'], hist_dict['yedges'],
    hist_dict['hist'].T, cmap='jet', **kwargs,
  )

  ax.set_ylabel('Y [cm]')
  ax.set_xlabel('X [cm]')
  ax.grid(True, alpha=0.2)
  ax.set_aspect('equal')

def adjust_figure(fig):
  fig.subplots_adjust(
    left=0.16, right=0.96,
    bottom=0.08, top=0.96,
    wspace=0.20, hspace=0.24,
  )


def main_procedure(cmdargs: argparse.Namespace):
  gaze_hist = accumulate_hists(
    root=cmdargs.dataset_path,
    filter_fn=lambda recording, meta: True,
    gaze_bins=np.linspace(-20, 20, 400),
  )

  fig, ax_gaze = plt.subplots(1, 1, figsize=(4, 4))
  plot_hist(ax_gaze, gaze_hist, gamma=cmdargs.gaze_gamma)

  fig.tight_layout()
  adjust_figure(fig)

  plt.savefig(f'{cmdargs.plot_name}.png')



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='plot annotations for gaze-2d datasets.')

  parser.add_argument(
    '--dataset-path', type=str, required=True,
    help='Path to the prepared gaze dataset.',
  )

  parser.add_argument(
    '--gaze-gamma', type=float, default=0.0,
    help='Gamma value for the gaze plot normalization.',
  )
  parser.add_argument(
    '--plot-name', type=str, required=True,
    help='Name of the output annots plot.',
  )

  main_procedure(parser.parse_args())
