from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Subset
from typing import Type

from opengaze.registry import DATASETS
from opengaze.utils.dataset import build_image_transform, build_data_pipeline
from opengaze.utils.euler import gaze_3d_2d_a

import cv2
import glob
import functools
import h5py
import numpy as np
import os
import os.path as osp
import random
import torch


class _BaseAnnotUnit(Dataset, ABC):
  '''Annotation unit consists of face image folder and `annot.h5`.'''

  def __init__(self, folder: str, transform=None):
    super(_BaseAnnotUnit, self).__init__()

    self.folder = folder

    self.hdf_path = osp.join(folder, 'annot.h5')
    with h5py.File(self.hdf_path, 'r', swmr=True) as hdf_file:
      self.n_samples = self.load_num_samples(hdf_file)
    self.hdf = None

    self.transform = build_image_transform(transform)

  @abstractmethod
  def load_num_samples(self, hdf_file: h5py.File):
    pass  # Load the number of samples

  @abstractmethod
  def load_image(self, hdf_file: h5py.File, idx: int) -> np.ndarray:
    pass  # Load the image data

  @abstractmethod
  def load_annot(self, hdf_file: h5py.File, idx: int) -> dict:
    pass  # Load annotation for each sample

  @abstractmethod
  def data_dict(self, image: np.ndarray, annot: dict) -> dict:
    pass  # Process the image and annotation

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx: int):
    if self.hdf is None:
      self.hdf = h5py.File(self.hdf_path, 'r', swmr=True)

    image = self.load_image(self.hdf, idx)
    annot = self.load_annot(self.hdf, idx)

    return self.data_dict(image, annot)


class _Gaze2DAnnotUnit(_BaseAnnotUnit):
  def load_num_samples(self, hdf_file: h5py.File):
    return len(hdf_file['face-gaze'])

  def data_dict(self, image: np.ndarray, annot: dict):
    face = self.transform(Image.fromarray(image, mode='RGB'))
    gaze = torch.tensor(annot['face-gaze'], dtype=torch.float32)
    return dict(face=face, gaze=gaze)


class _Gaze3DAnnotUnit(_BaseAnnotUnit):
  def load_num_samples(self, hdf_file: h5py.File):
    return len(hdf_file['face-gaze'])

  def data_dict(self, image: np.ndarray, annot: dict):
    face = self.transform(Image.fromarray(image, mode='RGB'))
    gp, gy = gaze_3d_2d_a(*annot['face-gaze'].tolist())
    gaze = torch.tensor([gp, gy], dtype=torch.float32)
    return dict(face=face, gaze=gaze)


class BUAAGazeGeneAnnotUnit(_Gaze3DAnnotUnit):
  def load_image(self, hdf_file: h5py.File, idx: int):
    image_path = osp.join(self.folder, 'face', f'{idx:05d}.jpg')
    image = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  def load_annot(self, hdf_file: h5py.File, idx: int):
    return {'face-gaze': np.array(hdf_file['face-gaze'][idx])}


class ETHXGazeAnnotUnit(_Gaze2DAnnotUnit):
  def load_image(self, hdf_file: h5py.File, idx: int):
    image_path = osp.join(self.folder, 'face', f'{idx:05d}.jpg')
    image = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  def load_annot(self, hdf_file: h5py.File, idx: int):
    return {'face-gaze': np.array(hdf_file['face-gaze'][idx])}


class IdiapEyeDiapAnnotUnit(_Gaze3DAnnotUnit):
  def load_image(self, hdf_file: h5py.File, idx: int):
    image_path = osp.join(self.folder, 'face', f'{idx:04d}.jpg')
    image = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  def load_annot(self, hdf_file: h5py.File, idx: int):
    return {'face-gaze': np.array(hdf_file['face-gaze'][idx])}


class MITGaze360AnnotUnit(_Gaze3DAnnotUnit):
  def load_image(self, hdf_file: h5py.File, idx: int):
    image_path = osp.join(self.folder, 'face', hdf_file['name'].asstr()[idx])
    image = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  def load_annot(self, hdf_file: h5py.File, idx: int):
    return {'face-gaze': np.array(hdf_file['face-gaze'][idx])}


class MPIIFaceGazeAnnotUnit(_Gaze3DAnnotUnit):
  def load_image(self, hdf_file: h5py.File, idx: int):
    image_path = osp.join(self.folder, 'face', f'{idx:04d}.jpg')
    image = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  def load_annot(self, hdf_file: h5py.File, idx: int):
    return {'face-gaze': np.array(hdf_file['face-gaze'][idx])}


class UCASSynthGazeAnnotUnit(_Gaze3DAnnotUnit):
  def load_image(self, hdf_file: h5py.File, idx: int):
    image_path = osp.join(self.folder, 'face', f'{idx:04d}.jpg')
    image = cv2.imread(image_path, flags=cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  def load_annot(self, hdf_file: h5py.File, idx: int):
    return {'face-gaze': np.array(hdf_file['face-gaze'][idx])}


class FaceGazeDataset(Dataset):
  def __init__(self, root: str, annot_units: list,
               annot_cls: Type[_BaseAnnotUnit],
               transform=None, pipeline=None):
    super(FaceGazeDataset, self).__init__()

    self.root = root

    self.data = ConcatDataset([
      annot_cls(osp.join(root, annot_unit), transform)
      for annot_unit in annot_units
    ])

    self.pipeline = build_data_pipeline(pipeline)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.pipeline(self.data[idx])


def _filter_idiap_eyediap_sessions(sessions, test_pp, train):
  if test_pp is not None:
    subjects = [str(it) for it in range(1, 16 + 1)]
    assert test_pp in subjects, f'Subject ID {test_pp} must be in range "1" - "16".'
    filter_fn = (
      lambda s: not test_pp in s
    ) if train else (
      lambda s: test_pp in s
    )
    sessions = [s for s in sessions if filter_fn(s)]

  return sessions

def _filter_mpii_facegaze_subjects(subjects, test_pp, train):
  if test_pp is not None:
    assert test_pp in subjects, f'Subject ID {test_pp} must be in range "p00" - "p14".'
    subjects = [s for s in subjects if not s == test_pp] if train else [test_pp]

  return subjects

@DATASETS.register_module(name='FaceGazeDataset')
def build_dataset(data_name, root, train=True, transform=None, pipeline=None, **kwargs):
  face_gaze_data = [
    'buaa-gazegene', 'eth-xgaze-224', 'idiap-eyediap',
    'mit-gaze-360', 'mpii-facegaze', 'ucas-synthgaze',
  ]
  assert data_name in face_gaze_data, (
    f'Data name "{data_name}" not found. Must be one of {face_gaze_data}.'
  )

  if data_name == 'buaa-gazegene':
    annot_cls = BUAAGazeGeneAnnotUnit
    subjects = sorted(os.listdir(root))
    annot_units = subjects[:-10] if train else subjects[-10:]

  if data_name == 'eth-xgaze-224':
    annot_cls = ETHXGazeAnnotUnit
    subjects = sorted(os.listdir(root))
    annot_units = subjects[:-10] if train else subjects[-10:]

  if data_name == 'idiap-eyediap':
    annot_cls = IdiapEyeDiapAnnotUnit
    annot_units = _filter_idiap_eyediap_sessions(
      sessions=sorted(os.listdir(root)),
      test_pp=kwargs.get('test_pp', None),
      train=train,
    )

  if data_name == 'mit-gaze-360':
    annot_cls = MITGaze360AnnotUnit
    annot_units = ['train' if train else 'test']

  if data_name == 'mpii-facegaze':
    annot_cls = MPIIFaceGazeAnnotUnit
    subjects = _filter_mpii_facegaze_subjects(
      subjects=sorted(os.listdir(root)),
      test_pp=kwargs.get('test_pp', None),
      train=train,
    )
    dates_list = [glob.glob(f'{s}/*', root_dir=root) for s in subjects]
    annot_units = sorted(functools.reduce(lambda a, b: a + b, dates_list))

  if data_name == 'ucas-synthgaze':
    annot_cls = UCASSynthGazeAnnotUnit
    subjects = sorted(os.listdir(root))
    annot_units = annot_units = subjects[10:] if train else subjects[:10]

  dataset = FaceGazeDataset(
    root=root,
    annot_units=annot_units,
    annot_cls=annot_cls,
    transform=transform,
    pipeline=pipeline,
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
