from mmengine.model import BaseModel
from typing import Callable, Optional

from opengaze.registry import MODELS, LOSSES

import torch as torch
import torch.nn.functional as F


class DataFnMixin:
  def data_fn(self, data_dict: dict):
    '''Takes as input the data dict from mmengine, and returns the actual
    data dict that the wrapped model expects.

    Args:
      `data_dict`: a dictionary of data for batch samples.
    '''

    return data_dict


@MODELS.register_module()
class BackboneHead(BaseModel):
  '''Model wrapper for Backbone-Head architecture, which takes many input streams,
  ie. face image, face bbox, and outputs the gaze prediction for each sample.
  '''

  def __init__(self, model_cfg: dict, loss_cfg: dict):
    '''Model wrapper for Backbone-Head architecture.

    Args:
      `model_cfg`: configuration dict for registered models of type `BaseModel`.
      `loss_cfg`: configuration dict for registered loss functions.

    Note that `data_fn` processes the input data dict from mmengine, then passes
    the actual data dict to the wrapped model. When wrapping a model, remember
    to provide the actual implementation of `data_fn`. The default `data_fn` simply
    returns the input data dict (no-op transformation).
    '''

    super(BackboneHead, self).__init__()

    self.model: DataFnMixin = MODELS.build(model_cfg)
    self.loss_fn = LOSSES.build(loss_cfg)

  def forward(self, mode='tensor', **data_dict):
    '''Parse actual data dict from the input data dict provided by mmengine,
    then runs the forward pass of the wrapped model. This method bridges the
    difference between mmengine and the wrapped model, however, it requires
    that the wrapped model specifys its inputs in a kwargs style.

    Args:
      `mode`: mode of forward pass, see `BaseModel.forward` for more details.
      `data_dict`: input data dict provided by mmengine.

    For convenience, `data_dict['gaze']` provides the ground-truth gaze label,
    see the implementation of gaze datasets for more details.
    '''

    gaze = self.model(**self.model.data_fn(data_dict))

    if mode == 'loss':
      loss = self.loss_fn(gaze, data_dict['gaze'])
      return dict(loss=loss)

    if mode == 'predict':
      pred_dict = dict(gaze=gaze)
      gold_dict = dict(gaze=data_dict['gaze'])
      return pred_dict, gold_dict

    return gaze


@MODELS.register_module()
class TdGazeNetWrapper(BaseModel):
  def __init__(
    self, model_cfg: dict,
    face_kpts_loss_cfg: dict,
    eyes_kpts_loss_cfg: dict,
    gaze_origin_loss_cfg: dict,
    gaze_vector_loss_cfg: dict,
    loss_weight: dict,
    skip_kpts_loss: bool = False,
    skip_gaze_loss: bool = False,
  ) -> None:
    super(TdGazeNetWrapper, self).__init__()

    assert all([
      'reduction' in loss_cfg and loss_cfg['reduction'] == 'none'
      for loss_cfg in [
        face_kpts_loss_cfg, eyes_kpts_loss_cfg,
        gaze_origin_loss_cfg, gaze_vector_loss_cfg,
      ]
    ])
    assert all([
      loss_key in loss_weight
      for loss_key in [
        'face_kpts_loss', 'eyes_kpts_loss',
        'gaze_origin_loss', 'gaze_vector_loss',
      ]
    ])

    self.model: DataFnMixin = MODELS.build(model_cfg)

    self.face_kpts_loss = LOSSES.build(face_kpts_loss_cfg)
    self.eyes_kpts_loss = LOSSES.build(eyes_kpts_loss_cfg)

    self.gaze_origin_loss = LOSSES.build(gaze_origin_loss_cfg)
    self.gaze_vector_loss = LOSSES.build(gaze_vector_loss_cfg)

    self.loss_weight = loss_weight

    self.skip_kpts_loss = skip_kpts_loss
    self.skip_gaze_loss = skip_gaze_loss

  def _custom_forward(self, data_dict: dict):
    output = self.model(**self.model.data_fn(data_dict))
    face_kpts, eyes_kpts, eyes_gaze = output
    return face_kpts, eyes_kpts, eyes_gaze

  def _custom_kpts_loss(
    self, loss_fn_name: str,
    pred: torch.Tensor, gold: torch.Tensor,
  ) -> torch.Tensor:
    loss_fn: Callable = getattr(self, loss_fn_name)
    loss = loss_fn(pred, gold).mean(dim=-1)
    weight = self.loss_weight[loss_fn_name]
    return weight * loss.mean()

  def _custom_gaze_loss(
    self, loss_fn_name: str,
    pred: torch.Tensor, gold: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    loss_fn: Callable = getattr(self, loss_fn_name)
    loss = loss_fn(pred, gold)
    if mask is not None:
      loss = loss * mask
    weight = self.loss_weight[loss_fn_name]
    return weight * loss.mean()

  def _custom_gate_loss(
    self, pred: torch.Tensor, mask: torch.Tensor,
  ) -> torch.Tensor:
    loss_fn = F.binary_cross_entropy_with_logits
    loss = loss_fn(pred, mask, reduction='none')
    weight = self.loss_weight['eyes_gate_loss']
    return weight * loss.mean()

  def forward(self, mode='tensor', **data_dict):
    face_kpts, eyes_kpts, eyes_gaze = self._custom_forward(data_dict)

    if mode == 'loss':
      loss_dict = dict()  # Loss Dictionary

      if not self.skip_kpts_loss:
        face_kpts_loss = self._custom_kpts_loss(
          loss_fn_name='face_kpts_loss',
          pred=face_kpts, gold=data_dict['face_kpts'],
        )
        eyes_kpts_loss = self._custom_kpts_loss(
          loss_fn_name='eyes_kpts_loss',
          pred=eyes_kpts, gold=data_dict['eyes_kpts'],
        )
        loss_dict.update(
          face_kpts_loss=face_kpts_loss,
          eyes_kpts_loss=eyes_kpts_loss,
        )

      if not self.skip_gaze_loss:
        reye_origin_loss = self._custom_gaze_loss(
          loss_fn_name='gaze_origin_loss',
          pred=eyes_gaze[:, 0, 0],
          gold=data_dict['eyes_gaze'][:, 0, 0],
          mask=data_dict['reye_mask'],
        )
        reye_vector_loss = self._custom_gaze_loss(
          loss_fn_name='gaze_vector_loss',
          pred=eyes_gaze[:, 0, 1],
          gold=data_dict['eyes_gaze'][:, 0, 1],
          mask=data_dict['reye_mask'],
        )
        leye_origin_loss = self._custom_gaze_loss(
          loss_fn_name='gaze_origin_loss',
          pred=eyes_gaze[:, 1, 0],
          gold=data_dict['eyes_gaze'][:, 1, 0],
          mask=data_dict['leye_mask'],
        )
        leye_vector_loss = self._custom_gaze_loss(
          loss_fn_name='gaze_vector_loss',
          pred=eyes_gaze[:, 1, 1],
          gold=data_dict['eyes_gaze'][:, 1, 1],
          mask=data_dict['leye_mask'],
        )
        gaze_origin_loss = (reye_origin_loss + leye_origin_loss) / 2
        gaze_vector_loss = (reye_vector_loss + leye_vector_loss) / 2
        loss_dict.update(
          gaze_origin_loss=gaze_origin_loss,
          gaze_vector_loss=gaze_vector_loss,
        )

      return dict(**loss_dict, loss=sum(loss_dict.values()))

    if mode == 'predict':
      pred_dict = dict(
        face_kpts=face_kpts[..., :3],
        reye_kpts=eyes_kpts[:, 0, :, :3],
        leye_kpts=eyes_kpts[:, 1, :, :3],
        reye_origin=eyes_gaze[:, 0, 0],
        reye_vector=eyes_gaze[:, 0, 1],
        leye_origin=eyes_gaze[:, 1, 0],
        leye_vector=eyes_gaze[:, 1, 1],
      )

      gold_dict = dict(
        face_kpts=data_dict['face_kpts'],
        reye_kpts=data_dict['eyes_kpts'][:, 0],
        leye_kpts=data_dict['eyes_kpts'][:, 1],
        reye_origin=data_dict['eyes_gaze'][:, 0, 0],
        reye_vector=data_dict['eyes_gaze'][:, 0, 1],
        leye_origin=data_dict['eyes_gaze'][:, 1, 0],
        leye_vector=data_dict['eyes_gaze'][:, 1, 1],
      )

      return pred_dict, gold_dict

    return face_kpts, eyes_kpts, eyes_gaze
