from mmengine.evaluator import BaseMetric
from typing import Dict, Optional

from opengaze.registry import METRICS

import torch


@METRICS.register_module()
class TdGazeNetGazeMetrics(BaseMetric):
  def _angular_error(
    self,
    gaze_vector_pred: torch.Tensor, gaze_vector_gold: torch.Tensor,
    mask: torch.Tensor,
  ) -> torch.Tensor:
    dot = torch.sum(gaze_vector_pred * gaze_vector_gold, dim=1)
    m_p = torch.linalg.vector_norm(gaze_vector_pred, ord=2, dim=1)
    m_g = torch.linalg.vector_norm(gaze_vector_gold, ord=2, dim=1)

    # Clamp cosine similarity to [-1, 1], to avoid NaNs
    # caused by numeric error, eg. 1.0000001
    sim = torch.clamp(dot / (m_p * m_g), min=-1.0, max=1.0)
    deg = torch.rad2deg(torch.acos(sim))

    return torch.sum(deg[mask.long()]).cpu()

  def _distance_error(
    self,
    gaze_origin_pred: torch.Tensor, gaze_origin_gold: torch.Tensor,
    mask: torch.Tensor,
  ) -> torch.Tensor:
    dists = torch.linalg.vector_norm(gaze_origin_pred - gaze_origin_gold, ord=2, dim=1)
    return torch.sum(dists[mask.long()]).cpu()

  def process(self, data_dict: dict, pred_results: list):
    pred_dict, gold_dict = pred_results

    # Shape: (B, 3)
    reye_gaze_vector_errs = self._angular_error(
      pred_dict['reye_vector'], gold_dict['reye_vector'],
      mask=data_dict['reye_mask'],
    )
    leye_gaze_vector_errs = self._angular_error(
      pred_dict['leye_vector'], gold_dict['leye_vector'],
      mask=data_dict['leye_mask'],
    )

    # Shape: (B, 3)
    reye_gaze_origin_errs = self._distance_error(
      pred_dict['reye_origin'], gold_dict['reye_origin'],
      mask=data_dict['reye_mask'],
    )
    leye_gaze_origin_errs = self._distance_error(
      pred_dict['leye_origin'], gold_dict['leye_origin'],
      mask=data_dict['leye_mask'],
    )

    reye_n_samples = torch.sum(data_dict['reye_mask']).cpu()
    leye_n_samples = torch.sum(data_dict['leye_mask']).cpu()

    self.results.append(dict(
      reye_gaze_vector_errs=reye_gaze_vector_errs,
      leye_gaze_vector_errs=leye_gaze_vector_errs,
      reye_gaze_origin_errs=reye_gaze_origin_errs,
      leye_gaze_origin_errs=leye_gaze_origin_errs,
      reye_n_samples=reye_n_samples,
      leye_n_samples=leye_n_samples,
    ))

  def compute_metrics(self, results: list):
    reye_gaze_vector_esum = sum([r['reye_gaze_vector_errs'] for r in results])
    leye_gaze_vector_esum = sum([r['leye_gaze_vector_errs'] for r in results])
    reye_gaze_origin_esum = sum([r['reye_gaze_origin_errs'] for r in results])
    leye_gaze_origin_esum = sum([r['leye_gaze_origin_errs'] for r in results])

    reye_n_samples = sum([r['reye_n_samples'] for r in results])
    leye_n_samples = sum([r['leye_n_samples'] for r in results])

    return dict(
      reye_gaze_vector_mae=reye_gaze_vector_esum / reye_n_samples,
      leye_gaze_vector_mae=leye_gaze_vector_esum / leye_n_samples,
      reye_gaze_origin_mde=reye_gaze_origin_esum / reye_n_samples,
      leye_gaze_origin_mde=leye_gaze_origin_esum / leye_n_samples,
    )


@METRICS.register_module()
class TdGazeNetMeshMetrics(BaseMetric):
  def _distance_error(
    self,
    mesh_pred: torch.Tensor, mesh_gold: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    dists = torch.linalg.vector_norm(mesh_pred - mesh_gold, ord=2, dim=-1)
    mean = torch.mean(dists, dim=-1)
    if mask is not None:
      mean = mean[mask.long()]
    return torch.sum(mean).cpu()

  def process(self, data_dict: dict, pred_results: list):
    pred_dict, gold_dict = pred_results

    # Shape: (B, K, 3)
    face_mesh_errs = self._distance_error(
      pred_dict['face_kpts'], gold_dict['face_kpts'],
      mask=None,
    )
    reye_mesh_errs = self._distance_error(
      pred_dict['reye_kpts'], gold_dict['reye_kpts'],
      mask=data_dict['reye_mask'],
    )
    leye_mesh_errs = self._distance_error(
      pred_dict['leye_kpts'], gold_dict['leye_kpts'],
      mask=data_dict['leye_mask'],
    )

    face_n_samples = len(data_dict['face_kpts'])
    reye_n_samples = torch.sum(data_dict['reye_mask']).cpu()
    leye_n_samples = torch.sum(data_dict['leye_mask']).cpu()

    self.results.append(dict(
      face_mesh_errs=face_mesh_errs, face_n_samples=face_n_samples,
      reye_mesh_errs=reye_mesh_errs, reye_n_samples=reye_n_samples,
      leye_mesh_errs=leye_mesh_errs, leye_n_samples=leye_n_samples,
    ))

  def compute_metrics(self, results: list):
    face_kpts_esum = sum([r['face_mesh_errs'] for r in results])
    reye_kpts_esum = sum([r['reye_mesh_errs'] for r in results])
    leye_kpts_esum = sum([r['leye_mesh_errs'] for r in results])

    face_n_samples = sum([r['face_n_samples'] for r in results])
    reye_n_samples = sum([r['reye_n_samples'] for r in results])
    leye_n_samples = sum([r['leye_n_samples'] for r in results])

    return dict(
      face_kpts_mde=face_kpts_esum / face_n_samples,
      reye_kpts_mde=reye_kpts_esum / reye_n_samples,
      leye_kpts_mde=leye_kpts_esum / leye_n_samples,
    )


@METRICS.register_module()
class TdGazeNetPlusMetrics(BaseMetric):
  def __init__(self, *args, **kwargs):
    self.threshold = kwargs.pop('classification_thres', 0.5)
    super(TdGazeNetPlusMetrics, self).__init__(*args, **kwargs)

  def _classification_error(
    self,
    gate_pred: torch.Tensor, gate_gold: torch.Tensor,
  ) -> torch.Tensor:
    pred_cls = (gate_pred > self.threshold).float()
    gold_cls = gate_gold.float()
    errors = torch.abs(pred_cls - gold_cls)
    return torch.sum(errors).cpu()

  def _measure_uncertainty(
    self,
    log_variance: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
  ) -> Dict[str, torch.Tensor]:
    uncertainty = torch.exp(log_variance)
    if mask is not None:
      uncertainty = uncertainty[mask.long()]
    return dict(
      min=torch.min(uncertainty).cpu(),
      max=torch.max(uncertainty).cpu(),
      sum=torch.sum(uncertainty).cpu(),
    )

  def process(self, data_dict: dict, pred_results: list):
    pred_dict, gold_dict = pred_results

    # Shape: (B, )
    reye_gate_errs = self._classification_error(
      gate_pred=pred_dict['reye_gate'],
      gate_gold=gold_dict['reye_gate'],
    )
    leye_gate_errs = self._classification_error(
      gate_pred=pred_dict['leye_gate'],
      gate_gold=gold_dict['leye_gate'],
    )

    # Shape: (B, K)
    face_uncertainty = self._measure_uncertainty(
      pred_dict['face_kpts_log_variance'],
      mask=None,
    )
    reye_uncertainty = self._measure_uncertainty(
      pred_dict['reye_kpts_log_variance'],
      mask=data_dict['reye_mask'],
    )
    leye_uncertainty = self._measure_uncertainty(
      pred_dict['leye_kpts_log_variance'],
      mask=data_dict['leye_mask'],
    )

    face_n_samples = len(pred_dict['face_kpts'])
    reye_n_samples = torch.sum(data_dict['reye_mask']).cpu()
    leye_n_samples = torch.sum(data_dict['leye_mask']).cpu()

    result_dict = dict(reye_gate_errs=reye_gate_errs, leye_gate_errs=leye_gate_errs)
    for key, value in face_uncertainty.items():
      result_dict[f'face_vars_{key}'] = value
    result_dict['face_n_samples'] = face_n_samples
    for key, value in reye_uncertainty.items():
      result_dict[f'reye_vars_{key}'] = value
    result_dict['reye_n_samples'] = reye_n_samples
    for key, value in leye_uncertainty.items():
      result_dict[f'leye_vars_{key}'] = value
    result_dict['leye_n_samples'] = leye_n_samples

    self.results.append(result_dict)

  def compute_metrics(self, results: list):
    reye_gate_esum = sum([r['reye_gate_errs'] for r in results])
    leye_gate_esum = sum([r['leye_gate_errs'] for r in results])

    face_n_samples = sum([r['face_n_samples'] for r in results])
    reye_n_samples = sum([r['reye_n_samples'] for r in results])
    leye_n_samples = sum([r['leye_n_samples'] for r in results])

    face_vars_min = min([r['face_vars_min'] for r in results])
    face_vars_max = max([r['face_vars_max'] for r in results])
    face_vars_csum = sum([r['face_vars_sum'] for r in results])

    reye_vars_min = min([r['reye_vars_min'] for r in results])
    reye_vars_max = max([r['reye_vars_max'] for r in results])
    reye_vars_csum = sum([r['reye_vars_sum'] for r in results])

    leye_vars_min = min([r['leye_vars_min'] for r in results])
    leye_vars_max = max([r['leye_vars_max'] for r in results])
    leye_vars_csum = sum([r['leye_vars_sum'] for r in results])

    return dict(
      reye_gate_acc=1.0 - reye_gate_esum / reye_n_samples,
      leye_gate_acc=1.0 - leye_gate_esum / leye_n_samples,
      face_vars_min=face_vars_min, face_vars_max=face_vars_max,
      face_vars_avg=face_vars_csum / face_n_samples,
      reye_vars_min=reye_vars_min, reye_vars_max=reye_vars_max,
      reye_vars_avg=reye_vars_csum / reye_n_samples,
      leye_vars_min=leye_vars_min, leye_vars_max=leye_vars_max,
      leye_vars_avg=leye_vars_csum / leye_n_samples,
    )
