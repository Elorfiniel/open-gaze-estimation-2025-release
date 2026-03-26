from collections import OrderedDict
from pathlib import Path

import torch as torch


def model_state_dict(checkpoint: Path, prefix: str = 'model', **kwargs):
  '''Extract model parameters from the loaded checkpoint file.

  Args:
    `checkpoint`: the checkpoint filepath.
    `prefix`: prefix of the model parameters.
    `kwargs`: additional arguments passed to `torch.load`.

  Note that prefix is removed from the model parameters, if not empty.
  '''

  checkpoint = Path(checkpoint).resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f'No checkpoint found: "{checkpoint}".')

  state_dict = torch.load(checkpoint, **kwargs)

  new_state_dict = OrderedDict()
  for key in state_dict.keys():
    new_key = key if not prefix else key.removeprefix(f'{prefix}')
    new_key = new_key.removeprefix('.')
    new_state_dict[new_key] = state_dict[key]

  return new_state_dict
