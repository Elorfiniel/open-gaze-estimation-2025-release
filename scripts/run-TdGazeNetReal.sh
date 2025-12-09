# Example command to run TdGazeNetReal script
python run-TdGazeNetReal.py --mode train \
  --data-name mit-gaze-capture \
  --num-workers 8 --batch-size 32 \
  --base-lr 1e-2 --max-epochs 2 \
  --work-dir runs/TdGazeNetReal/mit-gaze-capture
