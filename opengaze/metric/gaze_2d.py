from mmengine.evaluator import BaseMetric

from opengaze.registry import METRICS

import torch


@METRICS.register_module()
class DistanceError(BaseMetric):
  def process(self, data_dict: dict, pred_results: list):
    pred_dict, gold_dict = pred_results

    dists = torch.linalg.vector_norm(
      pred_dict['gaze'] - gold_dict['gaze'], ord=2, dim=1,
    )
    gaze_errs = torch.mean(dists).cpu()

    self.results.append(dict(gaze_errs=gaze_errs))

  def compute_metrics(self, results: list):
    return dict(
      mde=sum([r['gaze_errs'] for r in results]) / len(results),
    )


@METRICS.register_module()
class AFFNetDMetrics(BaseMetric):
  def __init__(self, *args, x_limits: tuple, y_limits: tuple, **kwargs):
    super(AFFNetDMetrics, self).__init__(*args, **kwargs)

    self.x_limits = x_limits
    self.y_limits = y_limits

  def _filter_samples(self, gaze_pred: torch.Tensor, gaze_gold: torch.Tensor):
    x_mask = torch.logical_and(
      gaze_gold[:, 0] >= self.x_limits[0],
      gaze_gold[:, 0] <= self.x_limits[1],
    )
    y_mask = torch.logical_and(
      gaze_gold[:, 1] >= self.y_limits[0],
      gaze_gold[:, 1] <= self.y_limits[1],
    )
    mask = torch.logical_and(x_mask, y_mask)
    return gaze_pred[mask], gaze_gold[mask]

  def _distance_error(self, gaze_pred: torch.Tensor, gaze_gold: torch.Tensor):
    dists = torch.linalg.vector_norm(gaze_pred - gaze_gold, ord=2, dim=1)
    return torch.mean(dists).cpu()

  def process(self, data_dict: dict, pred_results: list):
    pred_dict, gold_dict = pred_results

    reye_pred, reye_gold = self._filter_samples(pred_dict['reye_gaze'], gold_dict['reye_gaze'])
    reye_errs = self._distance_error(reye_pred, reye_gold)

    leye_pred, leye_gold = self._filter_samples(pred_dict['leye_gaze'], gold_dict['leye_gaze'])
    leye_errs = self._distance_error(leye_pred, leye_gold)

    self.results.append(dict(reye_errs=reye_errs, leye_errs=leye_errs))

  def compute_metrics(self, results: list):
    return dict(
      reye_mde=sum([r['reye_errs'] for r in results]) / len(results),
      leye_mde=sum([r['leye_errs'] for r in results]) / len(results),
    )
