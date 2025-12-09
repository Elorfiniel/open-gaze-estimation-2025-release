from mmengine.evaluator import BaseMetric

from opengaze.registry import METRICS
from opengaze.utils.euler import gaze_2d_3d_t

import torch


@METRICS.register_module()
class AngularError(BaseMetric):
  def _gaze_2d_to_3d(self, gaze_2d: torch.Tensor):
    x, y, z = gaze_2d_3d_t(gaze_2d[:, 0], gaze_2d[:, 1])
    return torch.stack([x, y, z], dim=1)

  def process(self, data_dict: dict, pred_results: list):
    pred_dict, gold_dict = pred_results

    gaze_pred = self._gaze_2d_to_3d(pred_dict['gaze'])
    gaze_gold = self._gaze_2d_to_3d(gold_dict['gaze'])

    dot = torch.sum(gaze_pred * gaze_gold, dim=1)
    m_p = torch.linalg.vector_norm(gaze_pred, ord=2, dim=1)
    m_g = torch.linalg.vector_norm(gaze_gold, ord=2, dim=1)

    # Clamp cosine similarity to [-1, 1], to avoid NaNs
    # caused by numeric error, eg. 1.0000001
    sim = torch.clamp(dot / (m_p * m_g), min=-1.0, max=1.0)
    deg = torch.rad2deg(torch.acos(sim))

    gaze_errs = torch.mean(deg).cpu()

    self.results.append(dict(gaze_errs=gaze_errs))

  def compute_metrics(self, results: list):
    return dict(
      mae=sum([r['gaze_errs'] for r in results]) / len(results),
    )
