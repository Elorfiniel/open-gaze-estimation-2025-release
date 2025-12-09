# Example command to run TdGazeNetBase script
python run-TdGazeNetBase.py --mode train \
  --num-workers 8 --batch-size 32 \
  --base-lr 3e-5 --max-epochs 10 \
  --warm-up-ratio 0.01 --warm-up-epoch 2.0 \
  --cool-down-ratio 0.1 --cool-down-epoch 8.0 \
  --work-dir runs/TdGazeNetBase-ucas-synthgaze
