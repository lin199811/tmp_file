from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from ..utils import common_utils, box_utils, self_training_utils
from ..ops.roiaware_pool3d import roiaware_pool3d_utils


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.oss_path = self.dataset_cfg.OSS_PATH if 'OSS_PATH' in self.dataset_cfg else None
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict: dict of original data from the dataloader
                frame_id:
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    @staticmethod
    def __vis__(points, gt_boxes, ref_boxes=None, scores=None, use_fakelidar=False):
        import visual_utils.visualize_utils as vis
        import mayavi.mlab as mlab
        gt_boxes = copy.deepcopy(gt_boxes)
        if use_fakelidar:
            gt_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(gt_boxes)

        if ref_boxes is not None:
            ref_boxes = copy.deepcopy(ref_boxes)
            if use_fakelidar:
                ref_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(ref_boxes)

        vis.draw_scenes(points, gt_boxes, ref_boxes=ref_boxes, ref_scores=scores)
        mlab.show(stop=True)

    @staticmethod
    def __vis_fake__(points, gt_boxes, ref_boxes=None, scores=None, use_fakelidar=True):
        import visual_utils.visualize_utils as vis
        import mayavi.mlab as mlab
        gt_boxes = copy.deepcopy(gt_boxes)
        if use_fakelidar:
            gt_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(gt_boxes)

        if ref_boxes is not None:
            ref_boxes = copy.deepcopy(ref_boxes)
            if use_fakelidar:
                ref_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(ref_boxes)

        vis.draw_scenes(points, gt_boxes, ref_boxes=ref_boxes, ref_scores=scores)
        mlab.show(stop=True)

    @staticmethod
    def extract_fov_data(points, fov_degree, heading_angle):
        """
        Args:
            points: (N, 3 + C)
            fov_degree: [0~180]
            heading_angle: [0~360] in lidar coords, 0 is the x-axis, increase clockwise
        Returns:
        """
        half_fov_degree = fov_degree / 180 * np.pi / 2
        heading_angle = -heading_angle / 180 * np.pi
        points_new = common_utils.rotate_points_along_z(
            points.copy()[np.newaxis, :, :], np.array([heading_angle])
        )[0]
        angle = np.arctan2(points_new[:, 1], points_new[:, 0])
        fov_mask = ((np.abs(angle) < half_fov_degree) & (points_new[:, 0] > 0))
        points = points_new[fov_mask]
        return points

    @staticmethod
    def extract_fov_gt(gt_boxes, fov_degree, heading_angle):
        """
        Args:
            anno_dict:
            fov_degree: [0~180]
            heading_angle: [0~360] in lidar coords, 0 is the x-axis, increase clockwise
        Returns:
        """
        half_fov_degree = fov_degree / 180 * np.pi / 2
        heading_angle = -heading_angle / 180 * np.pi
        gt_boxes_lidar = copy.deepcopy(gt_boxes)
        gt_boxes_lidar = common_utils.rotate_points_along_z(
            gt_boxes_lidar[np.newaxis, :, :], np.array([heading_angle])
        )[0]
        gt_boxes_lidar[:, 6] += heading_angle
        gt_angle = np.arctan2(gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0])
        fov_gt_mask = ((np.abs(gt_angle) < half_fov_degree) & (gt_boxes_lidar[:, 0] > 0))
        return fov_gt_mask

    def fill_pseudo_labels(self, input_dict):
        gt_boxes = self_training_utils.load_ps_label(input_dict['frame_id'])
        gt_scores = gt_boxes[:, 8]
        gt_classes = gt_boxes[:, 7]
        gt_boxes = gt_boxes[:, :7]

        # only suitable for only one classes, generating gt_names for prepare data
        gt_names = np.array([self.class_names[int(n) - 1]  for n in gt_classes])

        input_dict['gt_boxes'] = gt_boxes
        input_dict['gt_names'] = gt_names
        input_dict['gt_classes'] = gt_classes
        input_dict['gt_scores'] = gt_scores
        input_dict['pos_ps_bbox'] = (gt_classes > 0).sum()
        input_dict['ign_ps_bbox'] = gt_boxes.shape[0] - input_dict['pos_ps_bbox']
        input_dict.pop('num_points_in_gt', None)

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def set_lidar_aug_matrix(self, data_dict):
        """
            Get lidar augment matrix (4 x 4), which are used to recover orig point coordinates.
        """
        lidar_aug_matrix = np.eye(4)
        if 'flip_y' in data_dict.keys():
            flip_x = data_dict['flip_x']
            flip_y = data_dict['flip_y']
            if flip_x:
                lidar_aug_matrix[:3,:3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3,:3]
            if flip_y:
                lidar_aug_matrix[:3,:3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ lidar_aug_matrix[:3,:3]
        if 'noise_rot' in data_dict.keys():
            noise_rot = data_dict['noise_rot']
            lidar_aug_matrix[:3,:3] = common_utils.angle2matrix(torch.tensor(noise_rot)) @ lidar_aug_matrix[:3,:3]
        if 'noise_scale' in data_dict.keys():
            noise_scale = data_dict['noise_scale']
            lidar_aug_matrix[:3,:3] *= noise_scale
        if 'noise_translate' in data_dict.keys():
            noise_translate = data_dict['noise_translate']
            lidar_aug_matrix[:3,3:4] = noise_translate.T
        data_dict['lidar_aug_matrix'] = lidar_aug_matrix
        return data_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            # filter gt_boxes without points
            num_points_in_gt = data_dict.get('num_points_in_gt', None)
            if num_points_in_gt is None:
                num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(data_dict['points'][:, :3]),
                    torch.from_numpy(data_dict['gt_boxes'][:, :7])).numpy().sum(axis=1)

            mask = (num_points_in_gt >= self.dataset_cfg.get('MIN_POINTS_OF_GT', 1))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'gt_classes' in data_dict:
                data_dict['gt_classes'] = data_dict['gt_classes'][mask]
                data_dict['gt_scores'] = data_dict['gt_scores'][mask]

            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            # at this point, points & gt_boxes would already be shifted
            # shifting them back to the original position to enable correct db sampling
            if self.dataset_cfg.get('SHIFT_COOR', None):
                data_dict['points'][:, 0:3] -= np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                data_dict['gt_boxes'][:, 0:3] -= np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

            if 'calib' in data_dict:
                calib = data_dict['calib']

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )

            # reseting the shift
            if self.dataset_cfg.get('SHIFT_COOR', None):
                data_dict['points'][:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                data_dict['gt_boxes'][:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

            # for gt_sampling
            cls_map = self.dataset_cfg.get('CLASS_MAP', None)
            if cls_map is not None:
                for i in range(data_dict['gt_names'].shape[0]):
                    data_dict['gt_names'][i] = cls_map.get(data_dict['gt_names'][i], data_dict['gt_names'][i])

            if 'calib' in data_dict:
                data_dict['calib'] = calib

        data_dict = self.set_lidar_aug_matrix(data_dict)
        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            # for pseudo label has ignore labels.
            if 'gt_classes' not in data_dict:
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            else:
                gt_classes = data_dict['gt_classes'][selected]
                data_dict['gt_scores'] = data_dict['gt_scores'][selected]
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and data_dict['points'].shape[0] == 0:
            # new_index = np.random.randint(self.__len__())
            # return self.__getitem__(new_index)
            try:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            except Exception:
                pass  # 出错就继续下一次循环
        if self.training and len(data_dict['gt_boxes']) == 0:
            # new_index = np.random.randint(self.__len__())
            # return self.__getitem__(new_index)
            try:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
            except Exception:
                pass  # 出错就继续下一次循环
        data_dict.pop('gt_names', None)
        data_dict.pop('gt_classes', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        batch_size_ratio = 1

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    if isinstance(val[0], list):
                        batch_size_ratio = len(val[0])
                        val = [i for item in val for i in item]
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    if isinstance(val[0], list):
                        val =  [i for item in val for i in item]
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_scores']:
                    max_gt = max([len(x) for x in val])
                    batch_scores = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_scores[k, :val[k].__len__()] = val[k]
                    ret[key] = batch_scores
                elif key in ['roi_boxes']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1], :] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['roi_scores', 'roi_labels']:
                    max_gt = max([x.shape[1] for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, val[0].shape[0], max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k,:, :val[k].shape[1]] = val[k]
                    ret[key] = batch_gt_boxes3d

                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                elif key in ['calib']:
                    ret[key] = val
                elif key in ["points_2d"]:
                    max_len = max([len(_val) for _val in val])
                    pad_value = 0
                    points = []
                    for _points in val:
                        pad_width = ((0, max_len-len(_points)), (0,0))
                        points_pad = np.pad(_points,
                                pad_width=pad_width,
                                mode='constant',
                                constant_values=pad_value)
                        points.append(points_pad)
                    ret[key] = np.stack(points, axis=0)
                elif key in ['camera_imgs']:
                    ret[key] = torch.stack([torch.stack(imgs,dim=0) for imgs in val],dim=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size * batch_size_ratio
        return ret

    def eval(self):
        self.training = False
        self.data_processor.eval()

    def train(self):
        self.training = True
        self.data_processor.train()
