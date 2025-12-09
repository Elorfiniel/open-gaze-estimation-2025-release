from typing import Dict, Sequence, Union
from torch.utils.data import Dataset, ConcatDataset as TorchConcatDataset

from opengaze.registry import DATASETS


@DATASETS.register_module(force=True)
class ConcatDataset(TorchConcatDataset):
  def __init__(self, datasets: Sequence[Union[Dict, Dataset]]):
    _built_datasets = []

    for dataset in datasets:
      if not isinstance(dataset, (dict, Dataset)):
        raise TypeError(f'Expected dataset to be a dict or Dataset, but got {type(dataset)}.')

      if isinstance(dataset, dict):
        dataset = DATASETS.build(dataset)

      _built_datasets.append(dataset)

    super(ConcatDataset, self).__init__(datasets=_built_datasets)
