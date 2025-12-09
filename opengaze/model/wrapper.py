from mmengine.model import BaseModel

from opengaze.registry import MODELS, LOSSES

import torch as torch


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
class AFFNetDWrapper(BaseModel):
  def __init__(self, model_cfg: dict, loss_cfg: dict, x_limits: tuple, y_limits: tuple):
    super(AFFNetDWrapper, self).__init__()

    self.model: DataFnMixin = MODELS.build(model_cfg)
    self.loss_fn = LOSSES.build(loss_cfg)

    self.x_limits = x_limits
    self.y_limits = y_limits

  def _filter_samples(self, pred_gaze: torch.Tensor, gold_gaze: torch.Tensor):
    x_mask = torch.logical_and(
      gold_gaze[:, 0] >= self.x_limits[0],
      gold_gaze[:, 0] <= self.x_limits[1],
    )
    y_mask = torch.logical_and(
      gold_gaze[:, 1] >= self.y_limits[0],
      gold_gaze[:, 1] <= self.y_limits[1],
    )
    mask = torch.logical_and(x_mask, y_mask)
    return pred_gaze[mask], gold_gaze[mask]

  def forward(self, mode='tensor', **data_dict):
    reye_gaze, leye_gaze = self.model(**self.model.data_fn(data_dict))

    if mode == 'loss':
      reye_pred, reye_gold = self._filter_samples(reye_gaze, data_dict['reye_gaze'])
      reye_loss = self.loss_fn(reye_pred, reye_gold)
      leye_pred, leye_gold = self._filter_samples(leye_gaze, data_dict['leye_gaze'])
      leye_loss = self.loss_fn(leye_pred, leye_gold)
      loss_dict = dict(loss=reye_loss + leye_loss)
      loss_dict.update(reye_loss=reye_loss, leye_loss=leye_loss)
      return loss_dict

    if mode == 'predict':
      pred_dict = dict(reye_gaze=reye_gaze, leye_gaze=leye_gaze)
      gold_dict = dict(
        reye_gaze=data_dict['reye_gaze'],
        leye_gaze=data_dict['leye_gaze'],
      )
      return pred_dict, gold_dict

    return reye_gaze, leye_gaze


@MODELS.register_module()
class TdGazeNetWrapper(BaseModel):
  def __init__(self, model_cfg: dict, face_kpts_loss_cfg: dict, eyes_kpts_loss_cfg: dict,
               gaze_origin_loss_cfg: dict, gaze_vector_loss_cfg: dict, loss_weight: dict):
    super(TdGazeNetWrapper, self).__init__()
    self.model: DataFnMixin = MODELS.build(model_cfg)

    self.face_kpts_loss = LOSSES.build(face_kpts_loss_cfg)
    self.eyes_kpts_loss = LOSSES.build(eyes_kpts_loss_cfg)

    self.gaze_origin_loss = LOSSES.build(gaze_origin_loss_cfg)
    self.gaze_vector_loss = LOSSES.build(gaze_vector_loss_cfg)

    _loss_names = [
      'face_kpts_loss', 'eyes_kpts_loss',
      'gaze_origin_loss', 'gaze_vector_loss',
    ]
    assert all([key in loss_weight for key in _loss_names])

    self.loss_weight = loss_weight

  def forward(self, mode='tensor', **data_dict):
    face_kpts, eyes_kpts, eyes_gaze = self.model(**self.model.data_fn(data_dict))

    if mode == 'loss':
      face_kpts_loss = self.face_kpts_loss(face_kpts, data_dict['face_kpts'])
      eyes_kpts_loss = self.eyes_kpts_loss(eyes_kpts, data_dict['eyes_kpts'])

      reye_origin_loss = self.gaze_origin_loss(
        eyes_gaze[:, 0, 0],
        data_dict['eyes_gaze'][:, 0, 0],
      )
      reye_vector_loss = self.gaze_vector_loss(
        eyes_gaze[:, 0, 1],
        data_dict['eyes_gaze'][:, 0, 1],
      )
      leye_origin_loss = self.gaze_origin_loss(
        eyes_gaze[:, 1, 0],
        data_dict['eyes_gaze'][:, 1, 0],
      )
      leye_vector_loss = self.gaze_vector_loss(
        eyes_gaze[:, 1, 1],
        data_dict['eyes_gaze'][:, 1, 1],
      )

      gaze_origin_loss = (reye_origin_loss + leye_origin_loss) / 2
      gaze_vector_loss = (reye_vector_loss + leye_vector_loss) / 2

      loss_dict = dict(
        face_kpts_loss=face_kpts_loss,
        eyes_kpts_loss=eyes_kpts_loss,
        gaze_origin_loss=gaze_origin_loss,
        gaze_vector_loss=gaze_vector_loss,
      )

      loss = sum([
        self.loss_weight['face_kpts_loss'] * face_kpts_loss,
        self.loss_weight['eyes_kpts_loss'] * eyes_kpts_loss,
        self.loss_weight['gaze_origin_loss'] * gaze_origin_loss,
        self.loss_weight['gaze_vector_loss'] * gaze_vector_loss,
      ])

      return dict(**loss_dict, loss=loss)

    if mode == 'predict':
      pred_dict = dict(
        face_kpts=face_kpts,
        reye_kpts=eyes_kpts[:, 0],
        leye_kpts=eyes_kpts[:, 1],
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
