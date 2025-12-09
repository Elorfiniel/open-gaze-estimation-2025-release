# Example command to run FaceGaze script
python run-FaceGaze.py --mode train \
  --model-name XGaze224 \
  --data-name ucas-synthgaze \
  --train-subset 0.2 --test-subset 0.4 \
  --num-workers 6 --batch-size 50 \
  --max-epochs 10 \
  --mixed-precision \
  --work-dir runs/XGaze224-ucas-synthgaze
