from opengaze.engine.transform import BaseTransform
from opengaze.registry import TRANSFORMS

import cv2
import json
import math
import numpy as np
import os.path as osp
import random


_UCAS_SYNTHGAZE_LANDMARK_GROUPS_NAME = [
  'face_mesh_dense', 'face_mesh_light', 'face_mask',
  'reye_skin', 'reye_pupil', 'reye_iris', 'reye_globe',
  'leye_skin', 'leye_pupil', 'leye_iris', 'leye_globe',
]


# Transformations Stages (Numpy + OpenCV)
@TRANSFORMS.register_module()
class RandomCameraRotate3D(BaseTransform):
  '''Randomize camera pose by sampling over euler angles.'''

  LANDMARK_GROUPS_NAME = _UCAS_SYNTHGAZE_LANDMARK_GROUPS_NAME.copy()

  def __init__(self, camera_roll: float, safe_margin: float,
               rotate_along_camera_x_axis: bool = True,
               rotate_along_camera_y_axis: bool = True,
               skip_image_transformations: bool = False):
    self.camera_roll = camera_roll
    self.safe_margin = safe_margin

    self.rotate_along_camera_x_axis = rotate_along_camera_x_axis
    self.rotate_along_camera_y_axis = rotate_along_camera_y_axis

    self.skip_image_transformations = skip_image_transformations

  def _unit_vector(self, vector: np.ndarray):
    return vector / np.linalg.norm(vector)

  def _rotation_matrix(self, angle: float, axis: str):
    cos, sin = math.cos(angle), math.sin(angle)
    matrix = np.eye(4, dtype=np.float32)

    if axis == 'X':
      matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos, -sin, 0.0],
        [0.0, +sin, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ], dtype=np.float32)

    if axis == 'Y':
      matrix = np.array([
        [cos, 0.0, +sin, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sin, 0.0, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ], dtype=np.float32)

    if axis == 'Z':
      matrix = np.array([
        [cos, -sin, 0.0, 0.0],
        [+sin, cos, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ], dtype=np.float32)

    return matrix

  def _face_fov_range(self, extrinsic: np.ndarray, face_mesh: np.ndarray, axis):
    world_coords = np.hstack([face_mesh, np.ones(shape=(len(face_mesh), 1))])
    local_coords = np.linalg.inv(extrinsic) @ world_coords.T
    local_coords = local_coords[:3, :] / local_coords[3, :]

    if axis == 'X':
      rot = np.rad2deg(np.arctan2(+local_coords[0, :], -local_coords[2, :]))
    if axis == 'Y':
      rot = np.rad2deg(np.arctan2(-local_coords[1, :], -local_coords[2, :]))

    return rot.min(), rot.max()

  def _face_rot_range(self, fov: float, rot_m1: float, rot_m2: float):
    assert self.safe_margin >= 0.0

    # Positive angle means counter-clockwise rotation along the axis
    # Therefore, the two angles are swapped and negated here

    # Angle 1: the margin left for counter-clockwise rotation
    angle1 = min(0.0, -fov / 2.0 - rot_m1)
    # Angle 2: the margin left for clockwise rotation
    angle2 = max(0.0, +fov / 2.0 - rot_m2)

    # Apply safe margin to keep object completely in view
    margin = self.safe_margin * fov if self.safe_margin < 1.0 else self.safe_margin
    if abs(angle1) >= margin: angle1 = angle1 + margin
    if abs(angle2) >= margin: angle2 = angle2 - margin

    return -angle2, -angle1

  def random_rotate(self, intrinsic: np.ndarray, extrinsic: np.ndarray, face_mesh: np.ndarray):
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    ux, uy = intrinsic[0][2], intrinsic[1][2]

    fov_x = 2.0 * math.degrees(math.atan2(ux, fx))
    fov_y = 2.0 * math.degrees(math.atan2(uy, fy))

    # Axis-Mapping from Blender to OpenCV: (X, Y, Z) -> (X, -Y, -Z)
    M_cv_warp = np.eye(3, dtype=np.float32)

    # Random rotation around Z axis
    if self.camera_roll > 0.0:
      angle = math.radians(random.uniform(-self.camera_roll, +self.camera_roll))
      M_rotation = self._rotation_matrix(angle, 'Z')
      extrinsic = extrinsic @ np.linalg.inv(M_rotation)
      M_cv_warp = np.linalg.inv(M_rotation[:3, :3]) @ M_cv_warp

    # Random rotation around X axis (find FOV along Y axis)
    if self.rotate_along_camera_x_axis:
      rot_m1, rot_m2 = self._face_fov_range(extrinsic, face_mesh, 'Y')
      m1, m2 = self._face_rot_range(fov_y, rot_m1, rot_m2)
      angle = math.radians(random.uniform(m1, m2))
      M_rotation = self._rotation_matrix(angle, 'X')
      extrinsic = extrinsic @ np.linalg.inv(M_rotation)
      M_cv_warp = M_rotation[:3, :3] @ M_cv_warp

    # Random rotation around Y axis (find FOV along X axis)
    if self.rotate_along_camera_y_axis:
      rot_m1, rot_m2 = self._face_fov_range(extrinsic, face_mesh, 'X')
      m1, m2 = self._face_rot_range(fov_x, rot_m1, rot_m2)
      angle = math.radians(random.uniform(m1, m2))
      M_rotation = self._rotation_matrix(angle, 'Y')
      extrinsic = extrinsic @ np.linalg.inv(M_rotation)
      M_cv_warp = np.linalg.inv(M_rotation[:3, :3]) @ M_cv_warp

    return extrinsic, M_cv_warp

  def project_points(self, points_3d: np.ndarray, extrinsic: np.ndarray):
    world_coords = np.hstack([points_3d, np.ones((len(points_3d), 1))])

    # Project 3D points to rotated camera space (OpenCV compatible)
    local_coords = np.linalg.inv(extrinsic) @ world_coords.T
    local_coords = local_coords[:3, :] / local_coords[3, :]

    # Axis-Mapping from Blender to OpenCV: (X, Y, Z) -> (X, -Y, -Z), 1m -> 1mm
    local_coords = 1e3 * np.array([+1, -1, -1]) * local_coords.T

    return local_coords

  def project_gaze(self, gaze_origin: np.ndarray, gaze_vector: np.ndarray, extrinsic: np.ndarray):
    points_3d = np.stack([gaze_origin, gaze_origin + gaze_vector])
    origin, target = self.project_points(points_3d, extrinsic)
    vector = self._unit_vector(target - origin)
    return origin, vector

  def project_head_pose(self, head_pose: np.ndarray, extrinsic: np.ndarray):
    points_3d = np.stack([
      head_pose[:3, 3],
      head_pose[:3, 3] + head_pose[:3, 0],
      head_pose[:3, 3] + head_pose[:3, 1],
      head_pose[:3, 3] + head_pose[:3, 2],
    ])
    o, x, y, z = self.project_points(points_3d, extrinsic)
    ox, oy, oz = [self._unit_vector(p - o) for p in [x, y, z]]
    return o, ox, oy, oz

  def project_points_to_image(self, points_3d: np.ndarray, intrinsic: np.ndarray):
    # Project 3D points from camera space to image space
    image_coords = np.dot(intrinsic, points_3d.T)
    image_coords = image_coords[:2, :] / image_coords[2, :]

    return image_coords.T

  def points_2d_visibility(self, points_2d: np.ndarray, dsize: tuple):
    image_w, image_h = dsize
    return np.logical_and(
      np.logical_and(points_2d[:, 0] >= 0, points_2d[:, 0] < image_w),
      np.logical_and(points_2d[:, 1] >= 0, points_2d[:, 1] < image_h),
    )

  def transform(self, results: dict):
    updated_results = {k:v for k, v in results.items() if k.endswith('idx')}

    # Randomize camera pose, meanwhile, keep the face in view
    intrinsic = results['intrinsic_actual']
    extrinsic, M = self.random_rotate(
      intrinsic=results['intrinsic_actual'],
      extrinsic=results['extrinsic'],
      face_mesh=results['face_mesh_light'],
    )
    updated_results.update(intrinsic=intrinsic, extrinsic=extrinsic)

    # Apply perspective warping on the pre-rendered image
    M = results['intrinsic_actual'] @ M @ np.linalg.inv(results['intrinsic_render'])
    dsize = [2 * int(s) for s in results['intrinsic_actual'][:2, 2]]

    if not self.skip_image_transformations:
      image = cv2.warpPerspective(results['image'], M, dsize, flags=cv2.INTER_CUBIC)
      updated_results.update(image=image)

    # Project gaze-related annots to rotated camera space
    reye_origin, reye_vector = self.project_gaze(
      gaze_origin=results['reye_origin'],
      gaze_vector=results['reye_vector'],
      extrinsic=extrinsic,
    )
    updated_results.update(reye_origin=reye_origin, reye_vector=reye_vector)
    leye_origin, leye_vector = self.project_gaze(
      gaze_origin=results['leye_origin'],
      gaze_vector=results['leye_vector'],
      extrinsic=extrinsic,
    )
    updated_results.update(leye_origin=leye_origin, leye_vector=leye_vector)
    gaze_target = self.project_points(
      points_3d=results['gaze_target'].reshape(1, 3),
      extrinsic=extrinsic,
    ).reshape((3, ))
    updated_results.update(gaze_target=gaze_target)

    # Project head coordinates to rotated camera space
    head_o, head_ox, head_oy, head_oz = self.project_head_pose(
      head_pose=results['head_pose'],
      extrinsic=extrinsic,
    )
    updated_results.update(head_o=head_o, head_ox=head_ox, head_oy=head_oy, head_oz=head_oz)

    # Project face-related annots to rotated camera space
    for landmark_name in self.LANDMARK_GROUPS_NAME:
      points_3d = self.project_points(results[landmark_name], extrinsic)
      points_2d = self.project_points_to_image(points_3d, intrinsic)
      updated_results[f'{landmark_name}_3d'] = points_3d
      updated_results[f'{landmark_name}_2d'] = points_2d

      if f'{landmark_name}_vis' in results:
        points_2d_vis = self.points_2d_visibility(points_2d, dsize)
        updated_results[f'{landmark_name}_vis'] = np.logical_and(
          results[f'{landmark_name}_vis'], points_2d_vis,
        )

    return updated_results


@TRANSFORMS.register_module()
class RandomHFlip2D(BaseTransform):

  LANDMARK_GROUPS_NAME = _UCAS_SYNTHGAZE_LANDMARK_GROUPS_NAME.copy()

  def __init__(self, swap_file: str, p_hflip: float = 0.5, skip_images: bool = False):
    assert osp.isfile(swap_file) and 0 <= p_hflip <= 1
    self.swap_file = swap_file
    self.p_hflip = p_hflip
    self.skip_images = skip_images
    self._load_swap_data()

  def _load_swap_data(self, force: bool = False):
    if not hasattr(self, 'swap_data') or force:
      with open(self.swap_file, 'r', encoding='utf-8') as file:
        file_contents = json.load(file)

      self.swap_data = dict() # Source Mesh -> Swap Data
      for src_mesh, swap_data in file_contents.items():
        map_src2tgt = [
          src for src, _ in sorted(
            swap_data['indices_mapping'], key=lambda x: x[1],
          )
        ]
        map_tgt2src = [tgt for _, tgt in swap_data['indices_mapping']]

        self.swap_data[src_mesh] = dict(
          tgt_mesh=swap_data['swap_verts_name'],
          map_src2tgt=map_src2tgt,
          map_tgt2src=map_tgt2src,
        )

    assert self.swap_data is not None

  def _flip_points_3d(self, points_3d: np.ndarray):
    assert points_3d.ndim in [1, 2]

    points_3d = points_3d.copy()

    if points_3d.ndim == 1:
      points_3d[0] = -points_3d[0]

    if points_3d.ndim == 2:
      points_3d[:, 0] = -points_3d[:, 0]

    return points_3d

  def _flip_vector_3d(self, vector_3d: np.ndarray):
    return self._flip_points_3d(vector_3d)

  def _flip_points_2d(self, points_2d: np.ndarray, image_w: int):
    assert points_2d.ndim in [1, 2]

    points_2d = points_2d.copy()

    if points_2d.ndim == 1:
      points_2d[0] = image_w - points_2d[0]

    if points_2d.ndim == 2:
      points_2d[:, 0] = image_w - points_2d[:, 0]

    return points_2d

  def _flip_mesh_3d(self, points_3d: np.ndarray, src_mesh: str):
    tgt_mesh = self.swap_data[src_mesh]['tgt_mesh']

    map_src2tgt = self.swap_data[src_mesh]['map_src2tgt']
    points_3d = self._flip_points_3d(points_3d)[map_src2tgt]

    return points_3d, tgt_mesh

  def _flip_mesh_2d(self, points_2d: np.ndarray, image_w: int, src_mesh: str):
    tgt_mesh = self.swap_data[src_mesh]['tgt_mesh']

    map_src2tgt = self.swap_data[src_mesh]['map_src2tgt']
    points_2d = self._flip_points_2d(points_2d, image_w)[map_src2tgt]

    return points_2d, tgt_mesh

  def _flip_mesh_vis(self, vis: np.ndarray, src_mesh: str):
    tgt_mesh = self.swap_data[src_mesh]['tgt_mesh']

    map_src2tgt = self.swap_data[src_mesh]['map_src2tgt']
    vis = vis[map_src2tgt]

    return vis, tgt_mesh

  def transform(self, results: dict):
    if random.random() < self.p_hflip:
      if not self.skip_images:
        results['image'] = cv2.flip(results['image'], flipCode=1)

      results['reye_origin'], results['leye_origin'] = (
        self._flip_points_3d(results['leye_origin']),
        self._flip_points_3d(results['reye_origin']),
      )
      results['reye_vector'], results['leye_vector'] = (
        self._flip_vector_3d(results['leye_vector']),
        self._flip_vector_3d(results['reye_vector']),
      )

      results['gaze_target'] = self._flip_points_3d(results['gaze_target'])

      results['head_o'], results['head_ox'], results['head_oy'], results['head_oz'] = (
        self._flip_points_3d(results['head_o']),
        self._flip_vector_3d(-results['head_ox']),
        self._flip_vector_3d(results['head_oy']),
        self._flip_vector_3d(results['head_oz']),
      )

      update_dict = dict()  # Changed landmarks

      image_w = 2 * int(results['intrinsic_actual'][0, 2])
      for landmark_name in self.LANDMARK_GROUPS_NAME:
        if f'{landmark_name}_3d' in results:
          points_3d, tgt_mesh = self._flip_mesh_3d(
            points_3d=results[f'{landmark_name}_3d'],
            src_mesh=landmark_name,
          )
          update_dict[f'{tgt_mesh}_3d'] = points_3d

        if f'{landmark_name}_2d' in results:
          points_2d, tgt_mesh = self._flip_mesh_2d(
            points_2d=results[f'{landmark_name}_2d'],
            image_w=image_w,
            src_mesh=landmark_name,
          )
          update_dict[f'{tgt_mesh}_2d'] = points_2d

        if f'{landmark_name}_vis' in results:
          vis, tgt_mesh = self._flip_mesh_vis(
            vis=results[f'{landmark_name}_vis'],
            src_mesh=landmark_name,
          )
          update_dict[f'{tgt_mesh}_vis'] = vis

      results.update(update_dict)

    return results
