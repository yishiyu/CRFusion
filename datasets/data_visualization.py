from pathlib import Path
from colors import label_color
import numpy as np
import torch

import cv2


def visualization_preprocessed(data_path: str, anchors_path: str):
    """可视化预处理数据
    """

    data_path = Path(data_path)

    image_full_file = data_path / 'image_full.npy'
    labels_targets_file = data_path / 'labels_targets.npy'
    regression_targets_file = data_path / 'regression_targets.npy'

    with open(image_full_file, 'rb') as file:
        image_full = np.load(file)

    with open(labels_targets_file, 'rb') as file:
        labels_targets = np.load(file)

    with open(regression_targets_file, 'rb') as file:
        regression_targets = np.load(file)

    with open(anchors_path, 'rb') as file:
        anchors = np.load(file)

    # 恢复原始图片
    # image_full.shape = (5, 360, 640)
    image_full = (image_full[:3]*255).astype(np.uint8)
    image = np.ascontiguousarray(image_full.transpose(1, 2, 0))
    # image.shape = (360, 640, 3)

    # 恢复目标框
    # regression_targets.shape = (42975, 5) = (anchors_num, 4+1)
    # regression_targets[:,-1] = anchor_state (-1=ignore, 0=ground, 1=object)
    object_index = np.where(regression_targets[:, -1] == 1)
    regression_targets = regression_targets[object_index]
    regression_targets = regression_targets[:, :-1]

    # 根据anchors恢复
    anchors = anchors[object_index]
    regression_targets = (regression_targets + anchors).astype(int)

    # 恢复标签
    # labels_targets.shape = (42975, 9) = (anchors_num, cls_num+1)
    # # labels_targets[:,-1] = anchor_state (-1=ignore, 0=ground, 1=object)
    labels_targets = labels_targets[object_index]
    labels_targets = labels_targets[:, :-1]
    labels_targets = np.argmax(labels_targets, axis=1)

    # 绘制参数
    thickness = 2
    for regression, label in zip(regression_targets, labels_targets):
        color = label_color(label)
        # 绘制目标框
        cv2.rectangle(image,
                      (regression[0], regression[1]),
                      (regression[2], regression[3]),
                      color, thickness, cv2.LINE_AA)
        # 绘制
        cv2.putText(image, str(label),
                    (regression[0], regression[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), 1)

    cv2.imshow("debug", image)
    cv2.waitKey(0)


if __name__ == '__main__':

    import os
    import sys
    sys.path.append(os.getcwd())

    from utils.config import get_config
    config = get_config()

    data_root = os.path.join(config.preprocessed_data_dir, config.nusc_version)
    data_path = os.path.join(data_root, '00000')
    anchor_path = os.path.join(
        config.preprocessed_data_dir, 'anchors_xyxy_absolute.npy')

    visualization_preprocessed(data_path, anchor_path)
