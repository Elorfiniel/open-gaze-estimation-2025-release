from opengaze.dataset.compose import ConcatDataset
from opengaze.registry import DATA_SAMPLERS

from mmengine.dataset.sampler import DefaultSampler
from typing import Iterator, List, Optional, Sized

import math
import torch


@DATA_SAMPLERS.register_module()
class WeightedSamplerForConcatDataset(DefaultSampler):
  def __init__(self, dataset: Sized, weights: List[float], shuffle: bool = True,
               seed: Optional[int] = None, round_up: bool = True):
    assert isinstance(dataset, ConcatDataset) and len(dataset.datasets) == len(weights)
    super(WeightedSamplerForConcatDataset, self).__init__(
      dataset=dataset, shuffle=shuffle,
      seed=seed, round_up=round_up,
    )

    # Adjust the effective size for each dataset
    self._dataset_weights = weights
    self._effective_dataset_sizes = [
      int(len(ds) * wt)
      for ds, wt in zip(dataset.datasets, weights)
    ]

    # Update the number of samples for the sampler
    _total_size = sum(self._effective_dataset_sizes)
    if self.round_up:
      self.num_samples = math.ceil(_total_size / self.world_size)
      self.total_size = self.num_samples * self.world_size
    else:
      self.num_samples = math.ceil(
        (_total_size - self.rank) / self.world_size
      )
      self.total_size = _total_size

    # Compute cumulative offsets for index conversion
    self._cumulative_offsets = [0]
    for ds in dataset.datasets:
      _offset = self._cumulative_offsets[-1] + len(ds)
      self._cumulative_offsets.append(_offset)

  def __iter__(self) -> Iterator[int]:
    # Deterministically shuffle based on epoch and seed
    g = torch.Generator()
    g.manual_seed(self.seed + self.epoch)

    # Collect indices from each component dataset
    global_indices = []
    for idx, (ds, size) in enumerate(
      zip(self.dataset.datasets, self._effective_dataset_sizes)
    ):
      if self.shuffle:
        if size <= len(ds):
          indices = torch.randperm(len(ds), generator=g)[:size].tolist()
        else:
          indices = torch.randint(0, len(ds), (size, ), generator=g).tolist()
      else:
        indices = [it % len(ds) for it in range(size)]

      offset = self._cumulative_offsets[idx]
      global_indices.extend([offset + it for it in indices])

    if self.shuffle:
      global_indices = torch.as_tensor(global_indices, dtype=torch.long)
      shuffle_idx = torch.randperm(len(global_indices), generator=g)
      global_indices = global_indices[shuffle_idx].tolist()

    if self.round_up:
      global_indices = (
        global_indices * int(self.total_size / len(global_indices) + 1)
      )[:self.total_size]

    global_indices = global_indices[self.rank:self.total_size:self.world_size]

    return iter(global_indices)
