# dataset settings (data-name: default)
train_default = dict(
  type='SynthGaze',
  root='data/ucas-synthgaze-default',
  # For pilot study, use the following setting:
  #   subjects=[f'{subject_idx:08d}' for subject_idx in range(10, 24)],
  subjects=[f'{subject_idx:08d}' for subject_idx in range(10, 80)],
  subset=None,
  pipeline=[
    dict(type='RandomCameraRotate3D', camera_roll=80, safe_margin=5),
  ],
)

test_default = dict(
  type='SynthGaze',
  root='data/ucas-synthgaze-default',
  # For pilot study, use the following setting:
  #   subjects=[f'{subject_idx:08d}' for subject_idx in range(0, 4)],
  subjects=[f'{subject_idx:08d}' for subject_idx in range(0, 10)],
  subset=None,
  pipeline=[
    dict(type='RandomCameraRotate3D', camera_roll=80, safe_margin=5),
  ],
)

# dataset settings (data-name: extended)
train_extended = dict(
  type='SynthGaze',
  root='data/ucas-synthgaze-extended',
  # For pilot study, use the following setting:
  #   subjects=[f'{subject_idx:08d}' for subject_idx in range(0, 112)],
  subjects=[f'{subject_idx:08d}' for subject_idx in range(0, 1800)],
  subset=None,
  pipeline=[
    dict(type='RandomCameraRotate3D', camera_roll=80, safe_margin=5),
  ],
)

test_extended = dict(
  type='SynthGaze',
  root='data/ucas-synthgaze-extended',
  # For pilot study, use the following setting:
  #   subjects=[f'{subject_idx:08d}' for subject_idx in range(1800, 1832)],
  subjects=[f'{subject_idx:08d}' for subject_idx in range(1800, 2000)],
  subset=None,
  pipeline=[
    dict(type='RandomCameraRotate3D', camera_roll=80, safe_margin=5),
  ],
)


# data sampler settings
train_data_sampler = dict(type='DefaultSampler', shuffle=True)
# For weighted sampling for ConcatDataset (default+extended), use the following setting:
#   train_data_sampler = dict(
#     type='WeightedSamplerForConcatDataset',
#     weights=[1.0, 0.5], shuffle=True,
#   )
test_data_sampler = dict(type='DefaultSampler', shuffle=True)
