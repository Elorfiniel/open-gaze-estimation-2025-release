from torch.utils.data import Dataset, ConcatDataset, Subset

from opengaze.registry import DATASETS
from opengaze.utils.dataset import build_data_pipeline

import cv2
import h5py
import json
import numpy as np
import os
import os.path as osp
import random


class _PointOfGazeRecording(Dataset):
  def __init__(self, folder: str, meta: dict, return_meta_keys: list = None):
    super(_PointOfGazeRecording, self).__init__()

    self.folder = folder
    self.meta = meta
    self.return_meta_keys = return_meta_keys

    self.hdf_path = osp.join(folder, 'annot.h5')
    with h5py.File(self.hdf_path, 'r', swmr=True) as hdf_file:
      self.n_samples = len(hdf_file['name'])
      assert self.n_samples == meta['counts']
    self.hdf = None

    self.images_folder = osp.join(folder, 'images')

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx: int):
    if self.hdf is None:
      self.hdf = h5py.File(self.hdf_path, 'r', swmr=True)

    image_name = self.hdf['name'].asstr()[idx]
    image_path = osp.join(self.images_folder, image_name)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    gaze = np.array(self.hdf['gaze'][idx])
    ldmk = np.array(self.hdf['ldmk'][idx])

    data_dict = dict(image=image, gaze=gaze, ldmk=ldmk)
    if self.return_meta_keys is not None:
      data_dict.update(**{k:self.meta[k] for k in self.return_meta_keys})

    return data_dict


class PointOfGazeDataset(Dataset):
  def __init__(self, root: str, pipeline=None, *,
               meta_filter_configs: dict = None,
               return_meta_keys: list = None):
    super(PointOfGazeDataset, self).__init__()

    self.root = root

    self.data = ConcatDataset([
      _PointOfGazeRecording(osp.join(root, recording), meta, return_meta_keys)
      for recording, meta in self._list_recordings_with_meta(root, meta_filter_configs)
    ])

    self.pipeline = build_data_pipeline(pipeline)

  def _list_recordings_with_meta(self, root: str, meta_filter_configs: dict = None):
    recordings_with_meta = []

    for recording in os.listdir(root):
      meta_path = osp.join(root, recording, 'meta.json')
      with open(meta_path, 'r') as meta_file:
        meta = json.load(meta_file)

      skip = any([
        name not in meta or meta[name] not in candicates
        for name, candicates in meta_filter_configs.items()
      ]) if meta_filter_configs is not None else False

      if not skip: recordings_with_meta.append((recording, meta))

    return recordings_with_meta

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.pipeline(self.data[idx])

@DATASETS.register_module(name='PointOfGazeDataset')
def build_dataset(root, pipeline=None, **kwargs):
  meta_filter_configs = kwargs.get('meta_filter_configs', None)
  return_meta_keys = kwargs.get('return_meta_keys', None)
  dataset = PointOfGazeDataset(
    root, pipeline,
    meta_filter_configs=meta_filter_configs,
    return_meta_keys=return_meta_keys,
  )

  subset = kwargs.get('subset', None)
  if subset is not None:
    assert isinstance(subset, (int, float)), f'Parameter "subset" must be int or float.'
    if isinstance(subset, float):
      subset = int(subset * len(dataset))
    if isinstance(subset, int):
      assert 0 <= subset <= len(dataset), f'Subset size must be within [0, {len(dataset)}].'
    indices = random.sample(range(len(dataset)), subset)
    dataset = Subset(dataset, indices)

  return dataset
