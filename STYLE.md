# Code Style Guide

This document defines the coding style conventions for the open-gaze-estimation repository. Following these guidelines ensures consistency across the codebase and improves maintainability.

## Core Rules

### 1. Use 2-Space Indentation

Always use 2 spaces for indentation. No tabs.

```python
@MODELS.register_module()
class TdGazeNetBase(DataFnMixin, BaseModule):
  def __init__(self, n_face_kpts: int, n_eyes_kpts: int, inputs: str = 'face+eyes',
               shared_neck: bool = False, n_hidden_feats: int = 256):
    super(TdGazeNetBase, self).__init__(init_cfg=None)

    if inputs == 'face+eyes':
      self.conv1_1 = nn.Conv2d(9, 128, kernel_size=3, stride=2, padding=1)
      self.bn1_1 = nn.BatchNorm2d(num_features=128)
```

### 2. Prefer Single Quotes Over Double Quotes

Use single quotes for string literals unless the string contains single quotes.

```python
# Good
self.inputs = 'face+eyes'
data_name = 'ucas-synthgaze'
assert reduction in ['none', 'mean', 'sum']

# Acceptable (when string contains single quotes)
message = "Don't use backslash escapes"
```

### 3. Write Self-Documenting Code

Use descriptive variable and function names that explain their purpose. Add comments only for complex math or non-obvious logic.

```python
# Self-documenting - no comment needed
gaze_pred = self._gaze_2d_to_3d(pred_dict['gaze'])
gaze_gold = self._gaze_2d_to_3d(gold_dict['gaze'])

# Comment helpful for complex math
# Clamp cosine similarity to [-1, 1], to avoid NaNs
# caused by numeric error, eg. 1.0000001
sim = torch.clamp(dot / (m_p * m_g), min=-1.0, max=1.0)

# Self-documenting function names
def _reshape_neck_feats(self, feats: torch.Tensor):
  B, _, _, _ = feats.size()
  return nn.functional.adaptive_avg_pool2d(
    feats, output_size=(1, 1),
  ).view(B, -1)
```

### 4. Add Extra Empty Line After Docstrings

Add a blank line between the docstring and the first line of code for better readability.

```python
def gaze_3d_2d_a(x: float, y: float, z: float):
  '''Convert 3D gaze (x, y, z) to 2D gaze (pitch, yaw).

  Args:
    `x`: x coordinate in head coordinate frame.
    `y`: y coordinate in head coordinate frame.
    `z`: z coordinate in head coordinate frame.

  Note that 3D gaze should be of unit length, eg. `||g|| = 1`.
  '''

  pitch = np.arcsin(-y)
  yaw = np.arctan2(-x, -z)

  return pitch, yaw
```

### 5. Prefer Trailing Commas in Multi-Line Collections

Use trailing commas when lists, tuples, dicts, or function arguments span multiple lines. This enables clearer diffs and easier editing.

```python
# Dictionary with trailing comma
self.init_cfg = dict(
  type='Kaiming',
  mode='fan_in',
  layer=None,
  override=adapt_layers + neck_layers + head_layers,
)

# List with trailing comma
head_layers = [
  dict(name=x)
  for x in ['face_kpts_h', 'eyes_kpts_h', 'eyes_gaze_h']
]

# Function arguments with trailing comma
def __init__(self, n_face_kpts: int, n_eyes_kpts: int, inputs: str = 'face+eyes',
             shared_neck: bool = False, n_hidden_feats: int = 256):
  super(TdGazeNetBase, self).__init__(init_cfg=None)
```

## Additional Rules

### 6. Use Type Hints Consistently

Provide type hints for function parameters and return values, especially in public APIs.

```python
def process(self, data_dict: dict, pred_results: list):
  pred_dict, gold_dict = pred_results
  # ...

def _gaze_2d_to_3d(self, gaze_2d: torch.Tensor):
  x, y, z = gaze_2d_3d_t(gaze_2d[:, 0], gaze_2d[:, 1])
  return torch.stack([x, y, z], dim=1)
```

### 7. Organize Imports in Three Groups

Organize imports in this specific order:
1. `from` statements for local imports (if any)
2. `from` statements for stdlib/thirdparty
3. `import` statements for stdlib/thirdparty

Each group separated by a blank line.

```python
# Local imports (from statements)
from opengaze.registry import MODELS
from opengaze.model.wrapper import DataFnMixin
from opengaze.utils.euler import gaze_2d_3d_t

# Stdlib/Third-party imports (from statements)
from typing import Optional, Tuple, Union
from mmengine.model import BaseModule
from mmengine.evaluator import BaseMetric

# Stdlib/Third-party imports (import statements)
import copy
import functools
import cv2
import numpy as np
import torch
import torch.nn as nn
```

### 8. Naming Conventions

- **Classes**: `PascalCase` (e.g., `AngularError`, `TdGazeNetBase`)
- **Functions and variables**: `snake_case` (e.g., `gaze_2d_3d_a`, `face_kpts_h`)
- **Private methods**: Leading underscore (e.g., `_adapt_inputs`, `_build_neck`)
- **Constants**: `UPPER_CASE` (rarely used, prefer module-level constants)

```python
class RMSELoss(nn.Module):
  def __init__(self, dim: Optional[Union[int, Tuple[int]]] = None):
    super(RMSELoss, self).__init__()
    self.dim = dim

  def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
    # Public method - no underscore
    pass

  def _private_helper(self):
    # Private method - leading underscore
    pass
```

### 9. Use Registration Decorators for Framework Integration

Register classes with appropriate registries using decorators for framework integration.

```python
@METRICS.register_module()
class AngularError(BaseMetric):
  # ... implementation

@MODELS.register_module()
class TdGazeNetBase(DataFnMixin, BaseModule):
  # ... implementation

@DATASETS.register_module(name='FaceGazeDataset')
def build_dataset(data_name, root, train=True, **kwargs):
  # ... implementation
```

### 10. Use Asserts for Input Validation

Use assertions for validating function arguments and preconditions. This is particularly common in dataset and transform code.

```python
def _adapt_inputs(self, inputs: str):
  assert inputs in ['face+eyes', 'face']
  self.inputs = inputs

def gaze_2d_3d_n(pitch: np.ndarray, yaw: np.ndarray):
  assert pitch.ndim == 1 and pitch.shape == yaw.shape
  # ...
```

### 11. Break Long Lines for Readability

Limit lines to 96 characters as a soft guideline. Lines exceeding 96 characters should be split across multiple lines. Keep lines within 96 characters single-line unless there's a compelling readability reason to split them. When breaking function calls, object constructors, or data structures across multiple lines, add trailing commas.

Compact is preferred, only split when necessary.

```python
# Line exceeds 96 chars - break it with trailing comma
self.conv1_1 = nn.Conv2d(
  in_channels=9,
  out_channels=128,
  kernel_size=3,
  stride=2,
  padding=1,
  bias=False,
)

# Split lines with very long length into multiple lines
cv2.ellipse(
  glow_overlay, (cx, cy), glow_axes.astype(np.int32),
  self.angle, 0, 360, (246, 92, 199), -1, cv2.LINE_AA,
)

# Long dict literal - break it with trailing comma
config = dict(
  type='Kaiming',
  mode='fan_in',
  layer=None,
  override=adapt_layers + neck_layers + head_layers,
)

# Lines within 96 chars - keep single-line (default behavior)
self.relu1 = nn.ReLU(inplace=True)
dataset = FaceGazeDataset(root=root, annot_units=units)
cv2.circle(canvas, center, radius, color, thickness, line_type)
rotation_speed = base_speed + eased_speed * multiplier * (0.5 + 0.5 * proximity)

# List of dicts - break with trailing commas
transform_pipeline = [
  dict(type='RandomCameraRotate3D', camera_roll=60, safe_margin=5),
  dict(type='RandomHFlip2D', swap_file=swap_file, p_hflip=0.5),
  dict(type='GetFaceAndBBox', p_noisy=0.8),
  dict(type='PrepareDataDict'),
]
```

### 13. Use Triple Single Quotes for Docstrings

Use `'''` for docstrings (not `"""`). Keep docstrings concise but informative.

```python
class TdGazeNetBase(DataFnMixin, BaseModule):
  '''
  Bibliography:
    Ververas, Evangelos, et al. "3DGazeNet: Generalizing Gaze Estimation..."

  ArXiv:
    https://arxiv.org/abs/2212.02997

  Input:
    - face crop, shape: (B, 3, 112, 112)
    - reye crop, shape: (B, 3, 112, 112)

  Output:
    - face kpts, shape: (B, N1, 3)
  '''
```

### 14. Chain Methods Readably

When chaining multiple operations, consider readability. Sometimes intermediate variables are clearer than long chains.

```python
# Acceptable - simple chain
feats = self.layer1(feats)
feats = self.layer2(feats)
feats = self.layer3(feats)

# Also acceptable - when each step is clear
x = -torch.cos(pitch) * torch.sin(yaw)
y = -torch.sin(pitch)
z = -torch.cos(pitch) * torch.cos(yaw)
```

### 15. Module-Level Comments for Complex Coordinate Systems

For complex mathematical or coordinate system concepts, add module-level comments explaining the conventions.

```python
# Notes on the coordinate systems.
#
# When the viewer is captured in a frontal view (i.e. the camera is placed
# directly in front of the viewer, targeting the face), the head coordinate
# frame is defined as follows:
#   - x axis: points to the right (from right eye to left eye)
#   - y axis: points downwards (from forehead to chin)
#   - z axis: points towards the viewer (from camera to viewer)
```

## Summary

The codebase follows a clean, functional Python style influenced by:
- **mmengine/OpenMMLab** conventions (registry system, base classes)
- **Scientific Python** practices (NumPy/PyTorch idioms)
- **Minimalist documentation** (self-documenting code with targeted comments)
- **Import organization** (local `from` → stdlib/third-party `from` → stdlib/third-party `import`)

Key priorities:
1. **Readability** through consistent formatting and naming
2. **Type safety** through type hints
3. **Framework integration** through registration decorators
4. **Maintainability** through self-documenting code, inline comments for data structures, and targeted comments for complex logic
