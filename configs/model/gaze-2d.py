# model settings
ITracker = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='ITracker',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='MSELoss'),
)

AFFNet = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='AFFNet',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in',
        layer=['Conv2d', 'Linear'],
      ),
    ],
  ),
  loss_cfg=dict(type='SmoothL1Loss'),
)

ITrackerPlus = dict(
  type='BackboneHead',
  model_cfg=dict(
    type='ITrackerPlus',
    init_cfg=[
      dict(
        type='Kaiming', mode='fan_in', layer=None,
        override=[
          dict(name='face_fc'),
          dict(name='eyes_fc'),
          dict(name='kpts_fc'),
          dict(name='fc'),
        ],
      ),
    ],
  ),
  loss_cfg=dict(type='SmoothL1Loss'),
)
