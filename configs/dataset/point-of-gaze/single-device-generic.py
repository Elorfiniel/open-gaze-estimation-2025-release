# dataset settings
train = dict(
  type='PointOfGazeDataset',
  root='data/replace-with-actual-path',
  meta_filter_configs=dict(split=['train']),
)

test = dict(
  type='PointOfGazeDataset',
  root='data/replace-with-actual-path',
  meta_filter_configs=dict(split=['test']),
)
