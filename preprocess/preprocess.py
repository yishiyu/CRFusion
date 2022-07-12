import tqdm
import torch
from nuscenes.nuscenes import NuScenes
from .utils import get_sensor_sample_data, compute_overlap
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from . import radar
from .radar_projection import imageplus_creation
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image, points_in_box
from .anchors import create_anchors_xyxy_absolute
import json
import os
import sys
sys.path.append(os.getcwd())
from datasets import visualize_targets,visualize_result

class Preprocesser:
    def __init__(self, config):
        self.config = config

        # 防止重复处理同一个version的数据
        self.versions = set([config.nusc_version,
                             config.test_version,
                             config.val_version])

        # 一些固定的参数
        self.RADAR_CHANNEL = 'RADAR_FRONT'
        self.CAMERA_CHANNEL = 'CAM_FRONT'
        self.ROOT_DIR = config.data_dir
        self.OUT_DIR = config.preprocessed_data_dir
        if not os.path.exists(self.OUT_DIR):
            os.mkdir(self.OUT_DIR)
        self.CHANNELS = config.channels
        self.N_SWEEPS = config.n_sweeps
        self.ONLY_RADAR_ANNOTATED = config.only_radar_annotated
        self.cls_num = None

        # 融合参数
        self.IMAGE_TARGET_SHAPE = config.image_size
        self.HEIGHT = (0, config.radar_projection_height)

        # 创建 anchors
        self.anchors = create_anchors_xyxy_absolute().cpu().numpy()

        # 保存anchors
        with open(os.path.join(self.OUT_DIR, 'anchors_xyxy_absolute.npy'), 'wb') as file:
            np.save(file, self.anchors)

    def preprocess(self):
        for version in self.versions:
            nusc = NuScenes(version=version,
                            dataroot=self.ROOT_DIR,
                            verbose=True)

            # 创建从name<==>label的双向映射
            name2labels, labels2name = self._get_name_label_mapping(
                [c['name']for c in nusc.category],
                self.config.category_mapping
            )
            # 类别数量
            self.cls_num = len(labels2name)

            # 保存路径
            save_path = os.path.join(self.OUT_DIR, version)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            print("==========preprocess dataset {}==========".format(version))

            sample_index = 0

            for sample in tqdm.tqdm(nusc.sample):

                # 1. 获取图像数据
                camera_token, camera_sample = self._load_image(nusc, sample)

                # 2. 获取雷达数据
                radar_token, radar_sample = self._load_radar(nusc, sample)

                # 3. 数据融合
                kwargs = {
                    'pointsensor_token': radar_token,
                    'camera_token': camera_token,
                    'height': self.HEIGHT,
                    'image_target_shape': self.IMAGE_TARGET_SHAPE,
                    'clear_radar': False,
                    'clear_image': False,
                }
                image_full = imageplus_creation(nusc,
                                                image_data=camera_sample,
                                                radar_data=radar_sample,
                                                **kwargs)
                # 只取需要的通道(R,G,B,rcs,distance)
                image_full = np.array(image_full[:, :, self.CHANNELS])

                # 4. 加载标注数据
                annotations = self._load_annotations(nusc, sample, name2labels)
                # 如果当前sample中没有符合要求的目标,则放弃此sample
                if not annotations['bboxes'].shape[0]:
                    continue

                # 5. 计算标签
                regression_targets, labels_targets = self._compute_targets(
                    self.anchors, image_full, annotations, name2labels,
                    negative_overlap=self.config.positive_overlap,
                    positive_overlap=self.config.positive_overlap)

                # image_full ==> (5, 360, 640)
                image_full = image_full.transpose(2, 0, 1)

                # 6. 保存数据
                # 保存处理好的数据
                sample_path = os.path.join(
                    save_path, '{:0>5d}'.format(sample_index))
                if not os.path.exists(sample_path):
                    os.mkdir(sample_path)

                # sample数据
                meta = {
                    'sample_token': sample['token'],
                }
                with open(os.path.join(sample_path, 'meta.json'), 'w') as file:
                    json.dump(meta, file)

                # 图像数据
                with open(os.path.join(sample_path, 'image.npy'), 'wb') as file:
                    np.save(file, image_full[:3].transpose(1,2,0).astype(np.uint8))

                # 雷达数据
                with open(os.path.join(sample_path, 'radar.npy'), 'wb') as file:
                    np.save(file, image_full[3:])

                with open(os.path.join(sample_path, 'regression_targets.npy'), 'wb') as file:
                    np.save(file, regression_targets)

                with open(os.path.join(sample_path, 'labels_targets.npy'), 'wb') as file:
                    np.save(file, labels_targets)

                # 可视化
                # visualize_targets(image_full, self.anchors, regression_targets.numpy(), labels_targets.numpy(), display=True)
                # regression_gt = (annotations['bboxes']).astype(int)
                # image = (image_full[:3]).astype(np.uint8)
                # image = np.ascontiguousarray(image.transpose(1, 2, 0))
                # visualize_result(image, regression_gt, np.ones(regression_gt.shape[0], dtype=int), display=True)

                sample_index += 1
                pass
            pass
        pass

    def _load_image(self, nusc,  sample):
        camera_token = sample['data'][self.CAMERA_CHANNEL]
        # 后面的cv2.resize需要dtype=np.float32
        camera_sample = get_sensor_sample_data(
            nusc, sample, self.CAMERA_CHANNEL,
            dtype=np.float32, size=None
        )

        # TODO Add noise to the image if enabled
        # nuscenes_generator.py 354行

        return camera_token, camera_sample

    def _load_radar(self, nusc, sample):
        radar_token = sample['data'][self.CAMERA_CHANNEL]

        # TODO noise_filter
        # noise_filter没有开源出来
        # nuscenes_generator.py 374,385,392

        pcs, times = RadarPointCloud.from_file_multisweep(nusc, sample, self.RADAR_CHANNEL,
                                                          self.CAMERA_CHANNEL, nsweeps=self.N_SWEEPS, min_distance=0.0, merge=False)

        radar_sample = [radar.enrich_radar_data(pc.points) for pc in pcs]

        # radar_sample = [radar_data * frame]
        # radar_data.shape = (21, count)
        # 21: 每个雷达点有21个属性,radar.py:194行
        ## count: 雷达点数
        # 如果没有雷达点数据就创建空向量代替
        # 多个雷达帧合并
        if len(radar_sample) == 0:
            radar_sample = np.zeros(shape=(len(radar.channel_map), 0))
        else:
            radar_sample = np.concatenate(radar_sample, axis=-1)

        radar_sample = radar_sample.astype(dtype=np.float32)

        # TODO perfect_noise_filter
        # 降噪处理,后面再添加
        # TODO normalize_radar
        # 雷达点正则化处理
        # 用跟处理图像一样的方法处理
        # 具体参考nuscenes_generator.py 418

        return radar_token, radar_sample

    def _load_annotations(self, nusc,  sample, name2label):
        """加载标注数据
        """
        annotations = {
            'labels': [],       # <list of n int>
            'bboxes': [],       # <list of n x 4 float> [xmin, ymin, xmax, ymax]
            # <list of n float>  Center of box given as x, y, z.
            'distances': [],
            'visibilities': [],  # <list of n enum> nuscenes.utils.geometry_utils.BoxVisibility
            'num_radar_pts': []  # <list of n int>  number of radar points that cover that annotation
        }
        # 读取目标框和摄像机参数
        camera_data = nusc.get(
            'sample_data', sample['data'][self.CAMERA_CHANNEL])
        _, boxes, camera_intrinsic = nusc.get_sample_data(
            camera_data['token'], box_vis_level=BoxVisibility.ANY)
        # 原始图像数据以及缩放比例
        imsize_src = (camera_data['width'], camera_data['height'])
        bbox_resize = [self.IMAGE_TARGET_SHAPE[0] / camera_data['height'],
                       self.IMAGE_TARGET_SHAPE[1] / camera_data['width']]
        for box in boxes:
            # 只处理一部分目标的box
            if box.name in name2label:
                box.label = name2label[box.name]
                # BoxVisibility.ANY: box至少有一部分在图像中
                if box_in_image(box=box, intrinsic=camera_intrinsic, imsize=imsize_src, vis_level=BoxVisibility.ANY):

                    # 把box映射到2d
                    # [xmin, ymin, xmax, ymax]
                    box2d = box.box2d(camera_intrinsic)
                    box2d[0] *= bbox_resize[1]
                    box2d[1] *= bbox_resize[0]
                    box2d[2] *= bbox_resize[1]
                    box2d[3] *= bbox_resize[0]

                    annotations['bboxes'].append(box2d)
                    annotations['labels'].append(box.label)
                    annotations['num_radar_pts'].append(
                        nusc.get('sample_annotation',
                                 box.token)['num_radar_pts']
                    )
                    distance = (box.center[0]**2 +
                                box.center[1]**2 +
                                box.center[2]**2)**0.5
                    annotations['distances'].append(distance)
                    annotations['visibilities'].append(
                        int(nusc.get('sample_annotation', box.token)['visibility_token']))

        annotations['labels'] = np.array(annotations['labels'])
        annotations['bboxes'] = np.array(annotations['bboxes'])
        annotations['distances'] = np.array(annotations['distances'])
        annotations['num_radar_pts'] = np.array(annotations['num_radar_pts'])
        annotations['visibilities'] = np.array(annotations['visibilities'])

        # 筛选出至少有 n 个雷达点的box
        if self.ONLY_RADAR_ANNOTATED != 0:

            anns_to_keep = np.where(
                annotations['num_radar_pts'] == self.only_radar_annotated)[0]

            for key in annotations:
                annotations[key] = annotations[key][anns_to_keep]

        # TODO filter_annotations_enabled
        # 额外的筛选

        return annotations

    def _compute_targets(self, anchors, images, annotations,
                         name2labels=None, distance=False,
                         negative_overlap=0.4,
                         positive_overlap=0.5,
                         distance_scaling=100):
        """根据anchors和annotations生成targets

        Args:
            anchors (np.array): (N,4)==>(x1,y1,x2,y2)
            images (np.array): 图像数据
            annotations (dict): 标注数据
            distance (bool, optional): 是否生成距离标签. Defaults to False.
            negative_overlap (float, optional): IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative). Defaults to 0.4.
            positive_overlap (float, optional): IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive). Defaults to 0.5.
            distance_scaling (int, optional): 距离上限-. Defaults to 100.
        Return:
            regression_targets: 目标框
            labels_targets: 分类标签
        """
        # 对应anchor_calc.py 48
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."

        # regression_targets ==> (N, x1, y1, x2, y2, states)
        # states:-1 for ignore, 0 for bg, 1 for fg
        # labels_targets ==> (N, cls_num+1)
        regression_targets = torch.zeros(
            (anchors.shape[0], 4+1), dtype=torch.float)
        labels_targets = torch.zeros(
            (anchors.shape[0], self.cls_num+1), dtype=torch.float)
        # 将默认类别设为bg
        labels_targets[:, name2labels['bg']] = 1
        # distance_targets = torch.zeros((anchors.shape[0], 1+1), dtype=torch.float, device=device)

        # 该场景中存在目标
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = \
                self._compute_gt_annotations(
                    anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_targets[ignore_indices, -1] = -1
            labels_targets[positive_indices, -1] = 1

            regression_targets[ignore_indices, -1] = -1
            regression_targets[positive_indices, -1] = 1

            # distance_batch[index, ignore_indices, -1]   = -1
            # distance_batch[index, positive_indices, -1] = 1

            # compute target class labels
            pos_overlap_inds = [argmax_overlaps_inds[positive_indices]]
            label_indices = annotations['labels'][tuple(
                pos_overlap_inds)].astype(int)

            labels_targets[positive_indices, name2labels['bg']] = 0
            labels_targets[positive_indices, label_indices] = 1

            regression_targets[:, :-1] = torch.tensor(self._bbox_transform(
                anchors, annotations['bboxes'][argmax_overlaps_inds, :]))

        return regression_targets, labels_targets

    def _bbox_transform(self, anchors, gt_boxes, mean=None, std=None):
        """Compute bounding-box regression targets for an image."""

        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError(
                'Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError(
                'Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]

        targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
        targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
        targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
        targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

        targets = np.stack(
            (targets_dx1, targets_dy1, targets_dx2, targets_dy2))
        targets = targets.T

        targets = (targets - mean) / std

        return targets

    def _compute_gt_annotations(
        self,
        anchors,
        annotations,
        negative_overlap=0.4,
        positive_overlap=0.5
    ):
        """ Obtain indices of gt annotations with the greatest overlap.

        Args
            anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
            annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
            negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
            positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

        Returns
            positive_indices: indices of positive anchors
            ignore_indices: indices of ignored anchors
            argmax_overlaps_inds: ordered overlaps indices
        """

        # overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
        overlaps = compute_overlap(anchors, annotations)
        # 该anchor最接近哪个object(下标)
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        # 该anchor对应的重合度最高的目标框的iou
        max_overlaps = overlaps[np.arange(
            overlaps.shape[0]), argmax_overlaps_inds]

        # assign "dont care" labels
        positive_indices = max_overlaps >= positive_overlap
        ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

        return positive_indices, ignore_indices, argmax_overlaps_inds

    def _get_name_label_mapping(self, category_names, category_mapping):
        """获取name-label的双向映射

        Args:
            category_names (list): 数据集中所有的标签
            category_mapping (dict): name->name的映射(可能会将几个相近的类映射成同一个类)

        Returns:
            dict: 数据集中类别name向label(数字)的映射
            dict: label(数字)向类别name的映射
        """

        # 数据集中的标签(加上background)
        original_category_names = category_names.copy()
        original_category_names.insert(0, 'bg')

        # 处理后的标签列表(加上background)
        selected_category_names = list(set(category_mapping.values()))
        selected_category_names.sort()
        selected_category_names.insert(0, 'bg')

        labels2name = {label: name
                       for label, name in enumerate(selected_category_names)}
        name2labels = {}

        for label, name in labels2name.items():
            # 数据集标签中name对应类的所有子类
            original_names = [original_name
                              for original_name in original_category_names if name in original_name]

            for original_name in original_names:
                # 出现歧义(一个原始类对应多个映射后的类)
                assert original_name not in name2labels.keys(
                ), 'ambigous mapping found for (%s->%s)' % (original_name, name)

                name2labels[original_name] = label

        actual_labels = name2labels.values()
        expected_labels = range(0, max(actual_labels)+1)
        assert all([label in actual_labels for label in expected_labels]
                   ), 'Expected labels do not match actual labels'

        return name2labels, labels2name
