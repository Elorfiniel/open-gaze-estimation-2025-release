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
