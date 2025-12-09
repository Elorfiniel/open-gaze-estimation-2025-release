from torch.utils.data import Dataset, ConcatDataset, Subset
from typing import Callable, Dict, Sequence, Optional, Union

from opengaze.registry import DATASETS
from opengaze.utils.dataset import build_data_pipeline

import cv2
import json
import numpy as np
import os.path as osp
import random


def _load_json_file(json_file: str, **kwargs):
  with open(json_file, 'r', encoding='utf-8') as file:
    json_data = json.load(file, **kwargs)
  return json_data

def _load_np_file(np_file: str):
  if np_file.endswith('npz'):
    with np.load(np_file) as file:
      return dict(file.items())
  return np.load(np_file)

def _load_subject_hparams(subject_folder: str):
  hparams_path = osp.join(subject_folder, 'hparams.json')
  hparams_data = _load_json_file(hparams_path)

  return dict(
    num_gaze_targets=hparams_data['num_gaze_targets'],
    num_view_per_target=hparams_data['num_view_per_target'],
  )

def _load_annots_for_target(subject_folder: str, target_idx: int):
  target_folder = osp.join(subject_folder, f'target-{target_idx + 1:06d}')
  annots_dict = _load_json_file(osp.join(target_folder, 'annots.json'))

  return dict(
    # Common annotations shared among different views for a specific target
    intrinsic_actual=np.array(annots_dict['intrinsic']['actual']),
    intrinsic_render=np.array(annots_dict['intrinsic']['render']),
    head_pose=np.array(annots_dict['head_pose']),
    reye_origin=np.array(annots_dict['gaze_info']['reye']['origin']),
    reye_vector=np.array(annots_dict['gaze_info']['reye']['vector']),
    leye_origin=np.array(annots_dict['gaze_info']['leye']['origin']),
    leye_vector=np.array(annots_dict['gaze_info']['leye']['vector']),
    gaze_target=np.array(annots_dict['gaze_info']['target']),
    # 3D keypoints in the world coordinates, measured in meters
    vertex_dict=_load_np_file(osp.join(target_folder, 'annots', 'vertices.npz')),
    # View-specific annotations, accessible via view index
    extrinsic_list=_load_np_file(osp.join(target_folder, 'annots', 'extrinsics.npy')),
    visibility_dict=_load_np_file(osp.join(target_folder, 'annots', 'visibility.npz')),
  )


class _SynthGazeSubjectData(Dataset):
  def __init__(self, root: str, subject: str):
    super(_SynthGazeSubjectData, self).__init__()

    self.subject_folder = osp.join(root, subject)
    self._info = dict(root=root, subject=subject)

    self.hparams = _load_subject_hparams(self.subject_folder)
    self.annots = [
      _load_annots_for_target(self.subject_folder, target_idx)
      for target_idx in range(self.n_targets)
    ]

  @property
  def n_targets(self):
    return self.hparams['num_gaze_targets']

  @property
  def n_views(self):
    return self.hparams['num_view_per_target']

  def __len__(self):
    return self.n_targets * self.n_views

  def __getitem__(self, idx):
    target_idx, view_idx = idx // self.n_views, idx % self.n_views
    image_path = osp.join(
      self.subject_folder, f'target-{target_idx + 1:06d}',
      'images', f'{view_idx + 1:04d}.jpg',
    )

    data = dict(
      image=cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED),
      intrinsic_actual=self.annots[target_idx]['intrinsic_actual'],
      intrinsic_render=self.annots[target_idx]['intrinsic_render'],
      head_pose=self.annots[target_idx]['head_pose'],
      reye_origin=self.annots[target_idx]['reye_origin'],
      reye_vector=self.annots[target_idx]['reye_vector'],
      leye_origin=self.annots[target_idx]['leye_origin'],
      leye_vector=self.annots[target_idx]['leye_vector'],
      gaze_target=self.annots[target_idx]['gaze_target'],
      extrinsic=self.annots[target_idx]['extrinsic_list'][view_idx],
    )
    data.update(self.annots[target_idx]['vertex_dict'])
    data.update({
      f'{verts_name}_vis':verts_list[view_idx]
      for verts_name, verts_list in self.annots[target_idx]['visibility_dict'].items()
    })
    data.update(
      subject_idx=int(self._info['subject']),
      target_idx=target_idx, view_idx=view_idx,
    )

    return data


def _create_subset(dataset: Dataset, subset: Union[int, float]):
  assert isinstance(subset, (int, float)), f'Parameter "subset" must be int or float.'

  if isinstance(subset, float):
    subset = int(subset * len(dataset))

  if isinstance(subset, int):
    assert 0 <= subset <= len(dataset), f'Subset size must be within [0, {len(dataset)}].'

  indices = random.sample(range(len(dataset)), subset)

  return Subset(dataset, indices=indices)


@DATASETS.register_module()
class SynthGaze(Dataset):
  def __init__(self, root: str, subjects: list, pipeline: Sequence[Union[Callable, Dict]],
               subset: Optional[Union[int, float]] = None):
    super(SynthGaze, self).__init__()

    _subjects_data = [_SynthGazeSubjectData(root, subject) for subject in subjects]
    if subset is not None:
      _subjects_data = [_create_subset(d, subset) for d in _subjects_data]
    self.subjects_data = ConcatDataset(_subjects_data)

    self.pipeline = build_data_pipeline(pipeline)

  def __len__(self):
    return len(self.subjects_data)

  def __getitem__(self, idx):
    return self.pipeline(self.subjects_data[idx])
