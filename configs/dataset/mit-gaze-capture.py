# dataset settings
train = dict(
  type='PointOfGazeDataset',
  root='data/mit-gaze-capture',
  meta_filter_configs=dict(split=['train']),
)

val = dict(
  type='PointOfGazeDataset',
  root='data/mit-gaze-capture',
  meta_filter_configs=dict(split=['val']),
)

test = dict(
  type='PointOfGazeDataset',
  root='data/mit-gaze-capture',
  meta_filter_configs=dict(split=['test']),
)
