from mmengine.evaluator import BaseMetric

from opengaze.registry import METRICS

import torch


@METRICS.register_module()
class TdGazeNetGazeMetrics(BaseMetric):
  def _angular_error(self, gaze_vector_pred: torch.Tensor, gaze_vector_gold: torch.Tensor):
    dot = torch.sum(gaze_vector_pred * gaze_vector_gold, dim=1)
    m_p = torch.linalg.vector_norm(gaze_vector_pred, ord=2, dim=1)
    m_g = torch.linalg.vector_norm(gaze_vector_gold, ord=2, dim=1)

    # Clamp cosine similarity to [-1, 1], to avoid NaNs
    # caused by numeric error, eg. 1.0000001
    sim = torch.clamp(dot / (m_p * m_g), min=-1.0, max=1.0)
    deg = torch.rad2deg(torch.acos(sim))

    return torch.mean(deg).cpu()

  def _distance_error(self, gaze_origin_pred: torch.Tensor, gaze_origin_gold: torch.Tensor):
    dists = torch.linalg.vector_norm(gaze_origin_pred - gaze_origin_gold, ord=2, dim=1)
    return torch.mean(dists).cpu()

  def process(self, data_dict: dict, pred_results: list):
    pred_dict, gold_dict = pred_results

    reye_gaze_vector_errs = self._angular_error(
      pred_dict['reye_vector'], gold_dict['reye_vector'],
    )
    leye_gaze_vector_errs = self._angular_error(
      pred_dict['leye_vector'], gold_dict['leye_vector'],
    )

    reye_gaze_origin_errs = self._distance_error(
      pred_dict['reye_origin'], gold_dict['reye_origin'],
    )
    leye_gaze_origin_errs = self._distance_error(
      pred_dict['leye_origin'], gold_dict['leye_origin'],
    )

    self.results.append(dict(
      reye_gaze_vector_errs=reye_gaze_vector_errs,
      leye_gaze_vector_errs=leye_gaze_vector_errs,
      reye_gaze_origin_errs=reye_gaze_origin_errs,
      leye_gaze_origin_errs=leye_gaze_origin_errs,
    ))

  def compute_metrics(self, results: list):
    return dict(
      reye_gaze_vector_mae=sum([r['reye_gaze_vector_errs'] for r in results]) / len(results),
      leye_gaze_vector_mae=sum([r['leye_gaze_vector_errs'] for r in results]) / len(results),
      reye_gaze_origin_mde=sum([r['reye_gaze_origin_errs'] for r in results]) / len(results),
      leye_gaze_origin_mde=sum([r['leye_gaze_origin_errs'] for r in results]) / len(results),
    )


@METRICS.register_module()
class TdGazeNetMeshMetrics(BaseMetric):
  def _distance_error(self, mesh_pred: torch.Tensor, mesh_gold: torch.Tensor):
    dists = torch.linalg.vector_norm(mesh_pred - mesh_gold, ord=2, dim=-1)
    return torch.mean(dists).cpu()

  def process(self, data_dict: dict, pred_results: list):
    pred_dict, gold_dict = pred_results

    face_mesh_errs = self._distance_error(
      pred_dict['face_kpts'], gold_dict['face_kpts'],
    )
    reye_mesh_errs = self._distance_error(
      pred_dict['reye_kpts'], gold_dict['reye_kpts'],
    )
    leye_mesh_errs = self._distance_error(
      pred_dict['leye_kpts'], gold_dict['leye_kpts'],
    )

    self.results.append(dict(
      face_mesh_errs=face_mesh_errs,
      reye_mesh_errs=reye_mesh_errs, leye_mesh_errs=leye_mesh_errs,
    ))

  def compute_metrics(self, results: list):
    return dict(
      face_kpts_mde=sum([r['face_mesh_errs'] for r in results]) / len(results),
      reye_kpts_mde=sum([r['reye_mesh_errs'] for r in results]) / len(results),
      leye_kpts_mde=sum([r['leye_mesh_errs'] for r in results]) / len(results),
    )
