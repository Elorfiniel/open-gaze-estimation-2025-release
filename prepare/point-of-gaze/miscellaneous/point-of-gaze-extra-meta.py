from opengaze.runtime.scripts import ScriptEnv
from opengaze.runtime.log import runtime_logger

from PIL import Image

import argparse
import clip
import json
import os
import os.path as osp
import random
import torch


rt_logger = runtime_logger(
  name='point-of-gaze-extra-meta',
  log_file=ScriptEnv.log_path('point-of-gaze-extra-meta.log'),
)


def load_json_data(json_file: str):
  with open(json_file, 'r') as file:
    return json.load(file)

def save_json_data(json_data: dict, json_file: str):
  with open(json_file, 'w') as file:
    json.dump(json_data, file, indent=2)


def process_recording(data_folder: str, recording: str, extras: dict, min_samples: int = 0):
  recording_path = osp.join(data_folder, recording)
  meta_path = osp.join(recording_path, 'meta.json')

  # Merge extra metadata copied from user-specified file
  meta: dict = load_json_data(meta_path)
  meta.update(**extras)

  # Mark recording as dropped if it has less samples than specified
  meta['drop'] = meta['counts'] < min_samples
  if meta['drop']:
    rt_logger.warning(f'marked as "drop": recording {recording}, {meta["counts"]} samples')

  # Run zero-shot inference for glasses detection
  if not meta['drop']:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device)

    with_glasses = [] # Detection results, for majority vote
    n_samples = min(10, min_samples if min_samples > 0 else meta['counts'])

    classes = torch.cat([
      clip.tokenize(f'a photo of a person wearing glasses'),
      clip.tokenize(f'a photo of a person without glasses'),
    ]).to(device)
    with torch.no_grad():
      classes_feats = model.encode_text(classes)
      classes_feats /= classes_feats.norm(dim=-1, keepdim=True)

    images_folder = osp.join(recording_path, 'images')
    image_names = os.listdir(images_folder)

    image_names = random.sample(image_names, k=n_samples)
    for image_name in image_names:
      image = Image.open(osp.join(images_folder, image_name))
      image = preprocess(image).unsqueeze(0).to(device)
      with torch.no_grad():
        image_feats = model.encode_image(image)
        image_feats /= image_feats.norm(dim=-1, keepdim=True)
      probs = (100.0 * image_feats @ classes_feats.T).softmax(dim=-1)
      cls_glasses = [True, False][torch.argmax(probs, dim=-1).item()]
      with_glasses.append(cls_glasses)

    detection = sum(with_glasses) > len(with_glasses) / 2
    meta['glasses'] = detection

    rt_logger.info(f'processed: recording {recording}, glasses {detection}')

  save_json_data(meta, meta_path)


def main_procedure(cmdargs: argparse.Namespace):
  data_folder = ScriptEnv.data_path(cmdargs.data_name)
  rt_logger.info(f'use processed data: "{data_folder}"')

  extras = (
    load_json_data(cmdargs.copy_meta_from_json)
  ) if cmdargs.copy_meta_from_json else dict()
  assert isinstance(extras, dict), f'Extras must be a dict, got {type(extras)}.'

  for recording in os.listdir(data_folder):
    process_recording(data_folder, recording, extras, cmdargs.min_samples)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='add extra meta data to prepared PoG dataset.')

  parser.add_argument(
    '--data-name', type=str, required=True,
    help='Name of the prepared PoG dataset, eg. "mit-gaze-capture".',
  )
  parser.add_argument(
    '--copy-meta-from-json', type=str, default='',
    help='Copy meta data from this JSON file to all recordings.',
  )
  parser.add_argument(
    '--min-samples', type=int, default=50,
    help='Recordings with less samples will be marked as "drop".',
  )

  main_procedure(parser.parse_args())
