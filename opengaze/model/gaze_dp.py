# Models that revisit gaze estimation as a dense prediction task

from mmengine.model import BaseModule

from opengaze.registry import MODELS
from opengaze.model.wrapper import DataFnMixin

import copy
import functools
import timm
import torch as torch
import torch.nn as nn


def _infer_layer_out_features(module: nn.Module, input_sizes: list):
  module.train(mode=False)

  input_tensors = [torch.randn(*input_size) for input_size in input_sizes]
  with torch.no_grad():
    output_size = module(*input_tensors).shape

  module.train(mode=True)

  return output_size


@MODELS.register_module()
class TdGazeNetBase(DataFnMixin, BaseModule):
  '''
  Bibliography:
    Ververas, Evangelos, Polydefkis Gkagkos, Jiankang Deng, Michail Christos Doukas, Jia Guo, and Stefanos Zafeiriou.
    "3DGazeNet: Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views."

  ArXiv:
    https://arxiv.org/abs/2212.02997

  Input:
    - face crop, shape: (B, 3, 112, 112)
    - reye crop, shape: (B, 3, 112, 112)
    - leye crop, shape: (B, 3, 112, 112)

  Output:
    - face kpts, shape: (B, N1, 3)
    - eyes kpts, shape: (B, 2, N2, 3)
    - eyes gaze, shape: (B, 2, 2, 3)
  '''

  def __init__(
    self, n_face_kpts: int, n_eyes_kpts: int,
    inputs: str = 'face+eyes',
    shared_neck: bool = False, n_hidden_feats: int = 256,
  ) -> None:
    super(TdGazeNetBase, self).__init__(init_cfg=None)

    adapt_layers = self._adapt_inputs(inputs)

    pretrained = timm.models.resnet.resnet18(pretrained=True)
    self.layer1 = copy.deepcopy(pretrained.layer1)
    self.layer2 = copy.deepcopy(pretrained.layer2)
    self.layer3 = copy.deepcopy(pretrained.layer3)
    self.layer4 = copy.deepcopy(pretrained.layer4)

    neck_layers = self._build_neck(shared_neck, n_hidden_feats)

    head_layers = [
      dict(name=x)
      for x in ['face_kpts_h', 'eyes_kpts_h', 'eyes_gaze_h']
    ]
    self.face_kpts_h = nn.Linear(n_hidden_feats, n_face_kpts * 3)
    self.eyes_kpts_h = nn.Linear(n_hidden_feats, 2 * n_eyes_kpts * 3)
    self.eyes_gaze_h = nn.Linear(n_hidden_feats, 2 * 2 * 3)

    self.init_cfg = dict(
      type='Kaiming', mode='fan_in', layer=None,
      override=adapt_layers + neck_layers + head_layers,
    )

  def _adapt_inputs(self, inputs: str):
    assert inputs in ['face+eyes', 'face']

    self.inputs = inputs

    if inputs == 'face+eyes':
      self.conv1_1 = nn.Conv2d(9, 128, kernel_size=3, stride=2, padding=1)
      self.bn1_1 = nn.BatchNorm2d(num_features=128)
      self.relu1_1 = nn.ReLU(inplace=True)
      self.conv1_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
      self.bn1_2 = nn.BatchNorm2d(num_features=64)
      self.relu1_2 = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      return [dict(name='conv1_1'), dict(name='conv1_2')]

    if inputs == 'face':
      self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
      self.bn1 = nn.BatchNorm2d(num_features=64)
      self.relu1 = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      return [dict(name='conv1')]

  def _build_neck(self, shared_neck: bool, n_hidden_feats: int):
    self.shared_neck = shared_neck

    if not shared_neck:
      self.face_neck = nn.Conv2d(512, n_hidden_feats, kernel_size=3, stride=1, padding=1)
      self.eyes_neck = nn.Conv2d(512, n_hidden_feats, kernel_size=3, stride=1, padding=1)
      return [dict(name='face_neck'), dict(name='eyes_neck')]

    else:
      self.neck = nn.Conv2d(512, n_hidden_feats, kernel_size=3, stride=1, padding=1)
      return [dict(name='neck')]

  def data_fn(self, data_dict: dict):
    if self.inputs == 'face+eyes':
      return dict(face=data_dict['face'], reye=data_dict['reye'], leye=data_dict['leye'])

    if self.inputs == 'face':
      return dict(face=data_dict['face'])

  def _forward_inputs(self, **kwargs):
    if self.inputs == 'face+eyes':
      feats = torch.concat([kwargs['face'], kwargs['reye'], kwargs['leye']], dim=1)
      feats = self.relu1_1(self.bn1_1(self.conv1_1(feats)))
      feats = self.relu1_2(self.bn1_2(self.conv1_2(feats)))
      feats = self.maxpool(feats)

    if self.inputs == 'face':
      feats = self.relu1(self.bn1(self.conv1(kwargs['face'])))
      feats = self.maxpool(feats)

    return feats

  def _forward_resnet(self, feats: torch.Tensor):
    feats = self.layer1(feats)
    feats = self.layer2(feats)
    feats = self.layer3(feats)
    feats = self.layer4(feats)
    return feats

  def _reshape_neck_feats(self, feats: torch.Tensor):
    B, _, _, _ = feats.size()
    return nn.functional.adaptive_avg_pool2d(
      feats, output_size=(1, 1),
    ).view(B, -1)

  def _forward_neck(self, feats: torch.Tensor):
    B, _, _, _ = feats.size()

    if not self.shared_neck:
      face_feats = self._reshape_neck_feats(self.face_neck(feats))
      eyes_feats = self._reshape_neck_feats(self.eyes_neck(feats))

    else:
      feats = self._reshape_neck_feats(self.neck(feats))
      face_feats = eyes_feats = feats

    return face_feats, eyes_feats

  def forward(self, **kwargs):
    feats = self._forward_inputs(**kwargs)
    feats = self._forward_resnet(feats)

    B, _, _, _ = feats.size()

    face_feats, eyes_feats = self._forward_neck(feats)

    face_kpts = self.face_kpts_h(face_feats).view(B, -1, 3)
    eyes_kpts = self.eyes_kpts_h(eyes_feats).view(B, 2, -1, 3)
    eyes_gaze = self.eyes_gaze_h(eyes_feats).view(B, 2, 2, 3)

    return face_kpts, eyes_kpts, eyes_gaze


class _TdGazeNetConcatBBoxFusion(BaseModule):
  '''Feature fusion module that incorporates face bbox information
  into the main feature representation by simple concatenation.
  '''

  def __init__(self, n_view_feats: int, n_bbox_feats: int, n_hidden_feats: int):
    super(_TdGazeNetConcatBBoxFusion, self).__init__(
      init_cfg=dict(
        type='Kaiming', mode='fan_in', layer='Linear',
      ),
    )

    self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.fc = nn.Sequential(
      nn.Linear(n_view_feats + n_bbox_feats, n_hidden_feats),
      nn.SiLU(inplace=True),
      nn.Linear(n_hidden_feats, n_view_feats),
    )

  def forward(self, view_feats: torch.Tensor, bbox_feats: torch.Tensor):
    '''Fuse face bbox information with visual features.

    Args:
      `view_feats`: visual features from backbone of shape (B, C, H, W).
      `bbox_feats`: encoded face bbox features of shape (B, L).
    '''

    B, C, _, _ = view_feats.size()

    view_feats = self.gap(view_feats).view(B, C)

    feats = torch.cat([view_feats, bbox_feats], dim=1)
    fused = self.fc(feats)

    return fused

class _TdGazeNetAdaptiveBBoxFusion(BaseModule):
  '''Feature fusion module that incorporates face bbox information
  into the main feature representation using adaptive mechanisms.
  '''

  def __init__(self, n_view_feats: int, n_bbox_feats: int, n_hidden_feats: int,
               n_view_groups: int = 32, se_reduction: int = 16):
    super(_TdGazeNetAdaptiveBBoxFusion, self).__init__(
      init_cfg=dict(
        type='Kaiming', mode='fan_in', layer='Linear',
      ),
    )

    self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.modulation = nn.Sequential(
      nn.Linear(n_view_feats + n_bbox_feats, n_hidden_feats),
      nn.SiLU(inplace=True),
      nn.Linear(n_hidden_feats, 2 * n_view_feats),
    )
    self.gn = nn.GroupNorm(n_view_groups, n_view_feats)
    self.se = nn.Sequential(
      nn.Linear(n_view_feats, n_view_feats // se_reduction),
      nn.SiLU(inplace=True),
      nn.Linear(n_view_feats // se_reduction, n_view_feats),
      nn.Sigmoid(),
    )

  def forward(self, view_feats: torch.Tensor, bbox_feats: torch.Tensor):
    '''Fuse face bbox information with visual features.

    Args:
      `view_feats`: visual features from backbone of shape (B, C, H, W).
      `bbox_feats`: encoded face bbox features of shape (B, L).
    '''

    B, C, _, _ = view_feats.size()

    view_feats = self.gap(view_feats).view(B, C)

    # Combine view and bbox features for adaptive modulation
    combined_feats = torch.cat([view_feats, bbox_feats], dim=1)
    modulation_params = self.modulation(combined_feats)
    scale, shift = torch.chunk(modulation_params, 2, dim=1)

    # Apply adaptive modulation to view features
    modulated_feats = scale * self.gn(view_feats) + shift

    # Apply squeeze-and-excitation attention
    se_weights = self.se(modulated_feats)
    attended_feats = se_weights * modulated_feats

    return attended_feats

class _TdGazeNetRegHeadSimple(BaseModule):
  '''Regression head (simple-fc) for TdGazeNet to predict continuous signals.'''

  def __init__(self, n_in_feats: int, n_hidden_feats: int, out_shape: tuple):
    super(_TdGazeNetRegHeadSimple, self).__init__(
      init_cfg=dict(
        type='Kaiming', mode='fan_in', layer='Linear',
      ),
    )

    _n_out_feats = functools.reduce(lambda x, y: x * y, out_shape)
    self.fc = nn.Sequential(
      nn.Linear(n_in_feats, _n_out_feats),
    )

    self.out_shape = out_shape

  def forward(self, feats: torch.Tensor):
    return self.fc(feats).view(-1, *self.out_shape)

class _TdGazeNetRegHeadMultiTask(BaseModule):
  '''Regression head (multi-task) for TdGazeNet to predict continuous signals.'''

  def __init__(self, n_in_feats: int, n_hidden_feats: int, out_shape: tuple):
    super(_TdGazeNetRegHeadMultiTask, self).__init__(
      init_cfg=dict(
        type='Kaiming', mode='fan_in', layer='Linear',
      ),
    )

    _n_out_feats = functools.reduce(lambda x, y: x * y, out_shape)
    self.fc = nn.Sequential(
      nn.LayerNorm(n_in_feats),
      nn.Linear(n_in_feats, n_hidden_feats),
      nn.SiLU(inplace=True),
      nn.LayerNorm(n_hidden_feats),
      nn.Linear(n_hidden_feats, _n_out_feats),
    )

    self.out_shape = out_shape

  def forward(self, feats: torch.Tensor):
    return self.fc(feats).view(-1, *self.out_shape)

@MODELS.register_module()
class TdGazeNet(DataFnMixin, BaseModule):
  '''TdGazeNet, adapted from TdGazeNetBase. The main difference is the introduction of
  normalized face bbox information, which is fused into the visual features to provide
  additional information about the face crop.

  TdGazeNetBase proves in the "3DGazeNet" paper that revisiting gaze estimation as a
  dense prediction task can improve the generalization of the trained gaze estimator.
  However, TdGazeNetBase assumes weak perspective project of the eyes mesh. Without
  such assumption, the model fails to predict the location of mesh vertices in the
  physical world, ie. 3D coordinates of mesh vertices in the camera space.

  The introduction of normalized face bbox can help the network to correctly model
  the spatial prior of the human face, extending the model's ability to predict the
  location of mesh vertices precisely in real perspective projection.

  NOTE: Good Model = Sufficient Data + Strong Augmentation + Light-Weighted Network
  '''

  def __init__(
    self, backbone: str, fusion: str, reg_head: str,
    n_face_kpts: int, n_eyes_kpts: int,
    n_view_feats: int = 256, n_bbox_feats: int = 256, n_hidden_feats: int = 256,
  ) -> None:
    assert backbone in ['resnet-18', 'fastvit-sa12']
    assert fusion in ['concat', 'adaptive']
    assert reg_head in ['simple-fc', 'multi-task']

    super(TdGazeNet, self).__init__(
      init_cfg=dict(
        type='Kaiming', mode='fan_in', layer=None,
        override=[dict(name='neck'), dict(name='bbox_encode')],
      ),
    )

    if backbone == 'resnet-18':
      pretrained = timm.models.resnet.resnet18(pretrained=True)
      self.backbone = nn.Sequential(*[
        module
        for name, module in pretrained.named_children()
        if name not in ['global_pool', 'fc']
      ])

    if backbone == 'fastvit-sa12':
      pretrained = timm.models.fastvit.fastvit_sa12(pretrained=True)
      self.backbone = nn.Sequential(*[
        module
        for name, module in pretrained.named_children()
        if name not in ['head']
      ])

    _, n_bkb_feats, _, _ = _infer_layer_out_features(self.backbone, [(1, 3, 224, 224)])
    self.neck = nn.Conv2d(n_bkb_feats, n_view_feats, kernel_size=3, padding=1)

    self.bbox_encode = nn.Sequential(
      nn.Linear(4, n_bbox_feats // 2),
      nn.SiLU(inplace=True),
      nn.Linear(n_bbox_feats // 2, n_bbox_feats),
      nn.SiLU(inplace=True),
    )

    _fusion_cls = {
      'concat': _TdGazeNetConcatBBoxFusion,
      'adaptive': _TdGazeNetAdaptiveBBoxFusion,
    }[fusion]
    self.bbox_fusion = _fusion_cls(
      n_view_feats=n_view_feats,
      n_bbox_feats=n_bbox_feats,
      n_hidden_feats=n_hidden_feats,
    )

    _reg_head_cls = {
      'simple-fc': _TdGazeNetRegHeadSimple,
      'multi-task': _TdGazeNetRegHeadMultiTask,
    }[reg_head]
    self.face_kpts_h = _reg_head_cls(
      n_in_feats=n_view_feats,
      n_hidden_feats=n_hidden_feats,
      out_shape=(n_face_kpts, 3),
    )
    self.eyes_kpts_h = _reg_head_cls(
      n_in_feats=n_view_feats,
      n_hidden_feats=n_hidden_feats,
      out_shape=(2, n_eyes_kpts, 3),
    )
    self.eyes_gaze_h = _reg_head_cls(
      n_in_feats=n_view_feats,
      n_hidden_feats=n_hidden_feats,
      out_shape=(2, 2, 3),
    )

  def data_fn(self, data_dict: dict):
    return dict(face=data_dict['face'], bbox=data_dict['bbox'])

  def forward_feats(self, face: torch.Tensor, bbox: torch.Tensor):
    view_feats = self.neck(self.backbone(face))
    bbox_feats = self.bbox_encode(bbox)
    feats = self.bbox_fusion(view_feats, bbox_feats)
    return feats

  def forward(self, face: torch.Tensor, bbox: torch.Tensor):
    feats = self.forward_feats(face, bbox)

    face_kpts = self.face_kpts_h(feats)
    eyes_kpts = self.eyes_kpts_h(feats)
    eyes_gaze = self.eyes_gaze_h(feats)

    return face_kpts, eyes_kpts, eyes_gaze

