TdGazeNetBase = dict(
  type='TdGazeNetWrapper',
  model_cfg=dict(
    type='TdGazeNetBase',
    n_face_kpts=151,
    n_eyes_kpts=110,
  ),
  face_kpts_loss_cfg=dict(type='MSELoss'),
  eyes_kpts_loss_cfg=dict(type='MSELoss'),
  gaze_origin_loss_cfg=dict(type='MSELoss'),
  gaze_vector_loss_cfg=dict(type='L1Loss'),
  loss_weight=dict(
    face_kpts_loss=1.0,
    eyes_kpts_loss=1.0,
    gaze_origin_loss=1.0,
    gaze_vector_loss=1.0,
  ),
)

TdGazeNet = dict(
  type='TdGazeNetWrapper',
  model_cfg=dict(
    type='TdGazeNet',
    backbone='fastvit-sa12',
    fusion='adaptive',
    reg_head='multi-task',
    n_face_kpts=151,
    n_eyes_kpts=110,
  ),
  face_kpts_loss_cfg=dict(type='RMSELoss', dim=-1),
  eyes_kpts_loss_cfg=dict(type='RMSELoss', dim=-1),
  gaze_origin_loss_cfg=dict(type='RMSELoss', dim=-1),
  gaze_vector_loss_cfg=dict(type='RMSELoss', dim=-1),
  loss_weight=dict(
    face_kpts_loss=1.0,
    eyes_kpts_loss=1.0,
    gaze_origin_loss=2.0,
    gaze_vector_loss=2.0,
  ),
)

TdGazeNetReal = dict(
  type='AFFNetDWrapper',
  model_cfg=dict(
    type='TdGazeNetReal',
    n_adapters=1,
    custom_init_cfg=dict(
      type='Pretrained',
      checkpoint='TdGazeNet.pth',
      prefix='model.',
    ),
    backbone='fastvit-sa12',
    fusion='adaptive',
    reg_head='multi-task',
    n_face_kpts=151,
    n_eyes_kpts=110,
  ),
  loss_cfg=dict(type='SmoothL1Loss'),
  x_limits=[-40.0, 40.0],
  y_limits=[-40.0, 40.0],
)
