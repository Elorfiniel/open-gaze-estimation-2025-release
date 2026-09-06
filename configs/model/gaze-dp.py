TdGazeNetBase = dict(
  type='TdGazeNetWrapper',
  model_cfg=dict(
    type='TdGazeNetBase',
    n_face_kpts=151,
    n_eyes_kpts=110,
  ),
  face_kpts_loss_cfg=dict(type='MSELoss', reduction='none'),
  eyes_kpts_loss_cfg=dict(type='MSELoss', reduction='none'),
  gaze_origin_loss_cfg=dict(type='MSELoss', reduction='none'),
  gaze_vector_loss_cfg=dict(type='L1Loss', reduction='none'),
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
  face_kpts_loss_cfg=dict(type='MSELoss', reduction='none'),
  eyes_kpts_loss_cfg=dict(type='MSELoss', reduction='none'),
  gaze_origin_loss_cfg=dict(type='MSELoss', reduction='none'),
  gaze_vector_loss_cfg=dict(type='L1Loss', reduction='none'),
  loss_weight=dict(
    face_kpts_loss=1.0,
    eyes_kpts_loss=1.0,
    gaze_origin_loss=1.0,
    gaze_vector_loss=1.0,
  ),
)
