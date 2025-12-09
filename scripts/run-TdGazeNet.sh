# Example command to run TdGazeNet script
python run-TdGazeNet.py --mode train \
  --train-subset 0.2 --test-subset 0.4 \
  --num-workers 6 --batch-size 50 \
  --max-epochs 10 \
  --warm-up-epoch 2.0 \
  --cool-down-epoch 6.0 \
  --mixed-precision \
  --work-dir runs/TdGazeNet-ucas-synthgaze
