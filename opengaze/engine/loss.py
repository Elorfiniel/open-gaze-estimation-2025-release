from opengaze.registry import LOSSES

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


LOSSES.register_module(name='L1Loss', module=nn.L1Loss)
LOSSES.register_module(name='MSELoss', module=nn.MSELoss)
LOSSES.register_module(name='SmoothL1Loss', module=nn.SmoothL1Loss)


@LOSSES.register_module()
class RMSELoss(nn.Module):
  def __init__(self, dim: Optional[Union[int, Tuple[int]]] = None, reduction: str = 'mean'):
    assert reduction in ['none', 'mean', 'sum']

    super(RMSELoss, self).__init__()
    self.dim = dim
    self.reduction = reduction

  def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
    mse = F.mse_loss(y_pred, y_true, reduction='none')
    rmse = torch.sqrt(torch.sum(mse, dim=self.dim))

    if self.reduction == 'mean':
      return rmse.mean()

    if self.reduction == 'sum':
      return rmse.sum()

    return rmse
