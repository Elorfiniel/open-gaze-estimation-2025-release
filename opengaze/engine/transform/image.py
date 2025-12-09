from opengaze.engine.transform import BaseTransform
from opengaze.registry import TRANSFORMS

import kornia.augmentation as KA
import random
import torchvision.transforms.functional as TF


@TRANSFORMS.register_module()
class RandomImageAugmentation(BaseTransform):
  def __init__(self, p: float = 1.0, n_max_effective: int = 1,
               color_jiggle_kwargs: dict = None,
               random_gray_scale_kwargs: dict = None,
               random_box_blur_kwargs: dict = None,
               random_gaussian_blur_kwargs: dict = None,
               random_median_blur_kwargs: dict = None,
               random_motion_blur_kwargs: dict = None,
               random_brightness_kwargs: dict = None,
               random_contrast_kwargs: dict = None,
               random_sharpness_kwargs: dict = None,
               random_gamma_kwargs: dict = None,
               random_hue_kwargs: dict = None,
               random_saturation_kwargs: dict = None,
               random_equalize_kwargs: dict = None,
               random_channel_dropout_kwargs: dict = None,
               random_channel_shuffle_kwargs: dict = None,
               random_gaussian_noise_kwargs: dict = None,
               random_salt_and_pepper_noise_kwargs: dict = None,
               random_posterize_kwargs: dict = None,
               random_jpeg_kwargs: dict = None,
               random_planckian_jitter_kwargs: dict = None,
               *,
               image_data_key: str = 'image',
               drop_batch_dim: bool = False,
               normalize_kwargs: dict = None):

    assert 0.0 <= p <= 1.0 and n_max_effective >= 1

    self.p = p  # Performing augmentations

    self.augmentations = []

    if color_jiggle_kwargs is not None:
      self.color_jiggle = KA.ColorJiggle(**color_jiggle_kwargs)
      self.augmentations.append('color_jiggle')

    if random_gray_scale_kwargs is not None:
      self.random_gray_scale = KA.RandomGrayscale(**random_gray_scale_kwargs)
      self.augmentations.append('random_gray_scale')

    if random_box_blur_kwargs is not None:
      self.random_box_blur = KA.RandomBoxBlur(**random_box_blur_kwargs)
      self.augmentations.append('random_box_blur')

    if random_gaussian_blur_kwargs is not None:
      self.random_gaussian_blur = KA.RandomGaussianBlur(**random_gaussian_blur_kwargs)
      self.augmentations.append('random_gaussian_blur')

    if random_median_blur_kwargs is not None:
      self.random_median_blur = KA.RandomMedianBlur(**random_median_blur_kwargs)
      self.augmentations.append('random_median_blur')

    if random_motion_blur_kwargs is not None:
      self.random_motion_blur = KA.RandomMotionBlur(**random_motion_blur_kwargs)
      self.augmentations.append('random_motion_blur')

    if random_brightness_kwargs is not None:
      self.random_brightness = KA.RandomBrightness(**random_brightness_kwargs)
      self.augmentations.append('random_brightness')

    if random_contrast_kwargs is not None:
      self.random_contrast = KA.RandomContrast(**random_contrast_kwargs)
      self.augmentations.append('random_contrast')

    if random_sharpness_kwargs is not None:
      self.random_sharpness = KA.RandomSharpness(**random_sharpness_kwargs)
      self.augmentations.append('random_sharpness')

    if random_gamma_kwargs is not None:
      self.random_gamma = KA.RandomGamma(**random_gamma_kwargs)
      self.augmentations.append('random_gamma')

    if random_hue_kwargs is not None:
      self.random_hue = KA.RandomHue(**random_hue_kwargs)
      self.augmentations.append('random_hue')

    if random_saturation_kwargs is not None:
      self.random_saturation = KA.RandomSaturation(**random_saturation_kwargs)
      self.augmentations.append('random_saturation')

    if random_equalize_kwargs is not None:
      self.random_equalize = KA.RandomEqualize(**random_equalize_kwargs)
      self.augmentations.append('random_equalize')

    if random_channel_dropout_kwargs is not None:
      self.random_channel_dropout = KA.RandomChannelDropout(**random_channel_dropout_kwargs)
      self.augmentations.append('random_channel_dropout')

    if random_channel_shuffle_kwargs is not None:
      self.random_channel_shuffle = KA.RandomChannelShuffle(**random_channel_shuffle_kwargs)
      self.augmentations.append('random_channel_shuffle')

    if random_gaussian_noise_kwargs is not None:
      self.random_gaussian_noise = KA.RandomGaussianNoise(**random_gaussian_noise_kwargs)
      self.augmentations.append('random_gaussian_noise')

    if random_salt_and_pepper_noise_kwargs is not None:
      self.random_salt_and_pepper_noise = KA.RandomSaltAndPepperNoise(**random_salt_and_pepper_noise_kwargs)
      self.augmentations.append('random_salt_and_pepper_noise')

    if random_posterize_kwargs is not None:
      self.random_posterize = KA.RandomPosterize(**random_posterize_kwargs)
      self.augmentations.append('random_posterize')

    if random_jpeg_kwargs is not None:
      self.random_jpeg = KA.RandomJPEG(**random_jpeg_kwargs)
      self.augmentations.append('random_jpeg')

    if random_planckian_jitter_kwargs is not None:
      self.random_planckian_jitter = KA.RandomPlanckianJitter(**random_planckian_jitter_kwargs)
      self.augmentations.append('random_planckian_jitter')

    self.image_data_key = image_data_key
    self.drop_batch_dim = drop_batch_dim

    self.normalize_kwargs = normalize_kwargs

    self.n_max_effective = min(n_max_effective, len(self.augmentations))

  def transform(self, results: dict):
    if self.augmentations and random.random() < self.p:
      n_augments = random.randint(1, self.n_max_effective)

      image = results[self.image_data_key]

      augments = random.sample(self.augmentations, k=n_augments)
      for augment in augments:
        image = getattr(self, augment)(image)

      if self.drop_batch_dim:
        image = image.squeeze(dim=0)

      results[self.image_data_key] = image

    if self.normalize_kwargs is not None:
      results[self.image_data_key] = TF.normalize(
        results[self.image_data_key],
        **self.normalize_kwargs,
      )

    return results
