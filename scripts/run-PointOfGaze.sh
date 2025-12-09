# Example command to run PointOfGaze script on mit-gaze-capture dataset
python run-PointOfGaze.py --mode train \
  --data-name mit-gaze-capture \
  --train-subset 0.2 --test-subset 0.4 \
  --num-workers 6 --batch-size 50 \
  --max-epochs 10 \
  --warm-up-epoch 2.0 \
  --cool-down-epoch 6.0 \
  --mixed-precision \
  --work-dir runs/AFFNet-mit-gaze-capture

# Example command to run PointOfGaze script on mit-gaze-capture dataset for abalation study on glasses
python run-PointOfGaze.py --mode train \
  --data-name mit-gaze-capture-abalation-glasses \
  --num-workers 6 --batch-size 50 \
  --base-lr 3e-5 --max-epochs 10 \
  --warm-up-ratio 0.1 --warm-up-epoch 2.0 \
  --cool-down-ratio 0.1 --cool-down-epoch 8.0 \
  --mixed-precision \
  --abalation-glasses-device "iPhone 6" \
  --abalation-glasses-n-total-samples 200000 \
  --abalation-glasses-n-glasses-samples 50000 \
  --abalation-glasses-test-size 40000 \
  --work-dir runs/AFFNet-mit-gaze-capture-abalation-glasses
