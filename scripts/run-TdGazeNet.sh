# Example command to run TdGazeNet script, train on default, test on default
python run-TdGazeNet.py --mode train \
  --model-name TdGazeNet \
  --data-name "default" \
  --train-subset 0.2 --test-subset 0.4 \
  --num-workers 6 --batch-size 50 \
  --max-epochs 10 \
  --warm-up-epoch 2.0 \
  --cool-down-epoch 6.0 \
  --mixed-precision \
  --work-dir runs/TdGazeNet-ucas-synthgaze

# Example command to run TdGazeNetPlus script, train on extended, test on extended
python run-TdGazeNet.py --mode train \
  --model-name TdGazeNetPlus \
  --data-name "extended" \
  --train-subset 0.2 --test-subset 0.4 \
  --num-workers 6 --batch-size 50 \
  --max-epochs 10 \
  --warm-up-epoch 2.0 \
  --cool-down-epoch 6.0 \
  --mixed-precision \
  --work-dir runs/TdGazeNetPlus-ucas-synthgaze

# Example command to run TdGazeNetPlus script, train on extended, test on default
python run-TdGazeNet.py --mode train \
  --model-name TdGazeNetPlus \
  --data-name "extended" \
  --test-data-name "default" \
  --train-subset 0.2 --test-subset 0.4 \
  --num-workers 6 --batch-size 50 \
  --max-epochs 10 \
  --warm-up-epoch 2.0 \
  --cool-down-epoch 6.0 \
  --mixed-precision \
  --work-dir runs/TdGazeNetPlus-ucas-synthgaze

# Example command to run TdGazeNetPlus script, train on extended, test on default and extended
python run-TdGazeNet.py --mode train \
  --model-name TdGazeNetPlus \
  --data-name extended \
  --test-data-name "default+extended" \
  --train-subset 0.2 --test-subset 0.4 \
  --num-workers 6 --batch-size 50 \
  --max-epochs 10 \
  --warm-up-epoch 2.0 \
  --cool-down-epoch 6.0 \
  --mixed-precision \
  --work-dir runs/TdGazeNetPlus-ucas-synthgaze
