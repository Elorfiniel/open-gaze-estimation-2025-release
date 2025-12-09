# dataset settings
train = dict(
  type='SynthGaze',
  root='data/ucas-synthgaze',
  subjects=[f'{subject_idx:08d}' for subject_idx in range(10, 80)],
  subset=None,
  pipeline=[
    dict(type='RandomCameraRotate3D', camera_roll=80, safe_margin=5),
  ],
)

test = dict(
  type='SynthGaze',
  root='data/ucas-synthgaze',
  subjects=[f'{subject_idx:08d}' for subject_idx in range(0, 10)],
  subset=None,
  pipeline=[
    dict(type='RandomCameraRotate3D', camera_roll=80, safe_margin=5),
  ],
)
