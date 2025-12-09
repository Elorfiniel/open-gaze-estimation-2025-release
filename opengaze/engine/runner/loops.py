from opengaze.registry import LOOPS

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.runner.loops import BaseLoop
from torch.utils.data import DataLoader
from typing import Dict, Sequence, Union

import logging
import torch


@LOOPS.register_module()
class MultiSetValLoop(BaseLoop):
  '''Loop for validation on multiple dataloaders.

  Args:
    `runner`: a reference of runner.

    `dataloaders`: a dataloader object or dict to build dataloader, or a list of such objects.

    `evaluators`: an evaluator object or dict to build evaluator, used for computing metrics,
    must be a list of evaluators corresponding to each dataloader.

    `fp16`: whether to enable fp16 validation.
  '''

  def __init__(self,
               runner: Runner,
               dataloaders: Union[DataLoader, Dict, Sequence[Union[DataLoader, Dict]]],
               evaluators: Union[Evaluator, Dict, Sequence[Union[Evaluator, Dict]]],
               fp16: bool = False,
               **kwargs) -> None:
    self._runner = runner

    if isinstance(dataloaders, (DataLoader, Dict)):
      dataloaders = [dataloaders]
    for dataloader in dataloaders:
      assert isinstance(dataloader, (DataLoader, Dict)), (
        'each dataloader must be one of dict or Dataloader instance, '
        f'but got {type(dataloader)}.'
      )

    if isinstance(evaluators, (Evaluator, Dict)):
      evaluators = [evaluators]
    for evaluator in evaluators:
      assert isinstance(evaluator, (Evaluator, Dict)), (
        'each evaluator must be one of dict or Evaluator instance, '
        f'but got {type(evaluator)}.'
      )

    assert len(evaluators) == len(dataloaders), (
      'number of evaluators must be one, or equal to number of dataloaders, '
      f'but got {len(evaluators)} and {len(dataloaders)}.'
    )
    self.dataloaders = dataloaders
    self.evaluators = evaluators

    self.fp16 = fp16

  def _build_dataloader(self, dataloader: Union[DataLoader, Dict]) -> DataLoader:
    if isinstance(dataloader, Dict):
      diff_rank_seed = self.runner._randomness_cfg.get('diff_rank_seed', False)
      instance = self.runner.build_dataloader(
        dataloader, seed=self.runner.seed,
        diff_rank_seed=diff_rank_seed,
      )
    else:
      instance = dataloader

    return instance

  def _build_evaluator(self, evaluator: Union[Evaluator, Dict]) -> Evaluator:
    if isinstance(evaluator, Dict):
      instance = self.runner.build_evaluator(evaluator)
    else:
      instance = evaluator

    return instance

  def _propagate_update(self, dataloader: DataLoader, evaluator: Evaluator):
    if hasattr(dataloader.dataset, 'metainfo'):
      evaluator.dataset_meta = dataloader.dataset.metainfo
      self.runner.visualizer.dataset_meta = dataloader.dataset.metainfo
    else:
      print_log(
        f'Dataset {dataloader.dataset.__class__.__name__} has no '
        'metainfo. ``dataset_meta`` in evaluator, metric and '
        'visualizer will be None.',
        logger='current',
        level=logging.WARNING,
      )

    self.dataloader = dataloader
    self.evaluator = evaluator

  def run(self) -> dict:
    self.runner.call_hook('before_val')
    self.runner.model.eval()

    metrics = dict()  # metrics for all dataloaders combined
    for ddx, (dataloader, evaluator) in enumerate(zip(self.dataloaders, self.evaluators)):
      dataloader = self._build_dataloader(dataloader)
      evaluator = self._build_evaluator(evaluator)
      self._propagate_update(dataloader, evaluator)

      print_log(
        f'Run validation for dataloader-{ddx + 1}.',
        logger='current', level=logging.INFO,
      )

      self.runner.call_hook('before_val_epoch')

      for idx, data_batch in enumerate(dataloader):
        self.run_iter(idx, data_batch, evaluator)

      curr_metrics = evaluator.evaluate(len(dataloader.dataset))
      metrics_update_dict = {
        f'dataloader-{ddx + 1}/{k}':v
        for k, v in curr_metrics.items()
      }
      metrics.update(metrics_update_dict)

      self.runner.call_hook('after_val_epoch', metrics=curr_metrics)

    self.runner.call_hook('after_val')

    return metrics

  @torch.no_grad()
  def run_iter(self, idx: int, data_batch: Sequence[Dict], evaluator: Evaluator):
    self.runner.call_hook('before_val_iter', batch_idx=idx, data_batch=data_batch)

    with autocast(enabled=self.fp16):
      outputs = self.runner.model.val_step(data_batch)
    evaluator.process(data_samples=outputs, data_batch=data_batch)

    self.runner.call_hook(
      'after_val_iter',
      batch_idx=idx,
      data_batch=data_batch,
      outputs=outputs,
    )


@LOOPS.register_module()
class MultiSetTestLoop(BaseLoop):
  '''Loop for test on multiple dataloaders.

  Args:
    `runner`: a reference of runner.

    `dataloaders`: a dataloader object or dict to build dataloader, or a list of such objects.

    `evaluators`: an evaluator object or dict to build evaluator, used for computing metrics,
    must be a list of evaluators corresponding to each dataloader.

    `fp16`: whether to enable fp16 validation.
  '''

  def __init__(self,
               runner: Runner,
               dataloaders: Union[DataLoader, Dict, Sequence[Union[DataLoader, Dict]]],
               evaluators: Union[Evaluator, Dict, Sequence[Union[Evaluator, Dict]]],
               fp16: bool = False,
               **kwargs) -> None:
    self._runner = runner

    if isinstance(dataloaders, (DataLoader, Dict)):
      dataloaders = [dataloaders]
    for dataloader in dataloaders:
      assert isinstance(dataloader, (DataLoader, Dict)), (
        'each dataloader must be one of dict or Dataloader instance, '
        f'but got {type(dataloader)}.'
      )

    if isinstance(evaluators, (Evaluator, Dict)):
      evaluators = [evaluators]
    for evaluator in evaluators:
      assert isinstance(evaluator, (Evaluator, Dict)), (
        'each evaluator must be one of dict or Evaluator instance, '
        f'but got {type(evaluator)}.'
      )

    assert len(evaluators) == len(dataloaders), (
      'number of evaluators must be one, or equal to number of dataloaders, '
      f'but got {len(evaluators)} and {len(dataloaders)}.'
    )
    self.dataloaders = dataloaders
    self.evaluators = evaluators

    self.fp16 = fp16

  def _build_dataloader(self, dataloader: Union[DataLoader, Dict]) -> DataLoader:
    if isinstance(dataloader, Dict):
      diff_rank_seed = self.runner._randomness_cfg.get('diff_rank_seed', False)
      instance = self.runner.build_dataloader(
        dataloader, seed=self.runner.seed,
        diff_rank_seed=diff_rank_seed,
      )
    else:
      instance = dataloader

    return instance

  def _build_evaluator(self, evaluator: Union[Evaluator, Dict]) -> Evaluator:
    if isinstance(evaluator, Dict):
      instance = self.runner.build_evaluator(evaluator)
    else:
      instance = evaluator

    return instance

  def _propagate_update(self, dataloader: DataLoader, evaluator: Evaluator):
    if hasattr(dataloader.dataset, 'metainfo'):
      evaluator.dataset_meta = dataloader.dataset.metainfo
      self.runner.visualizer.dataset_meta = dataloader.dataset.metainfo
    else:
      print_log(
        f'Dataset {dataloader.dataset.__class__.__name__} has no '
        'metainfo. ``dataset_meta`` in evaluator, metric and '
        'visualizer will be None.',
        logger='current',
        level=logging.WARNING,
      )

    self.dataloader = dataloader
    self.evaluator = evaluator

  def run(self) -> dict:
    self.runner.call_hook('before_test')
    self.runner.model.eval()

    metrics = dict()  # metrics for all dataloaders combined
    for ddx, (dataloader, evaluator) in enumerate(zip(self.dataloaders, self.evaluators)):
      dataloader = self._build_dataloader(dataloader)
      evaluator = self._build_evaluator(evaluator)
      self._propagate_update(dataloader, evaluator)

      print_log(
        f'Run evaluation for dataloader-{ddx + 1}.',
        logger='current', level=logging.INFO,
      )

      self.runner.call_hook('before_test_epoch')

      for idx, data_batch in enumerate(dataloader):
        self.run_iter(idx, data_batch, evaluator)

      curr_metrics = evaluator.evaluate(len(dataloader.dataset))
      metrics_update_dict = {
        f'dataloader-{ddx + 1}/{k}':v
        for k, v in curr_metrics.items()
      }
      metrics.update(metrics_update_dict)

      self.runner.call_hook('after_test_epoch', metrics=curr_metrics)

    self.runner.call_hook('after_test')

    return metrics

  @torch.no_grad()
  def run_iter(self, idx: int, data_batch: Sequence[Dict], evaluator: Evaluator):
    self.runner.call_hook('before_test_iter', batch_idx=idx, data_batch=data_batch)

    with autocast(enabled=self.fp16):
      outputs = self.runner.model.test_step(data_batch)
    evaluator.process(data_samples=outputs, data_batch=data_batch)

    self.runner.call_hook(
      'after_test_iter',
      batch_idx=idx,
      data_batch=data_batch,
      outputs=outputs,
    )
