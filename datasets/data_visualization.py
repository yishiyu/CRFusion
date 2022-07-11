from pathlib import Path
import numpy as np
import cv2
import warnings


def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.

    Args
        label: The label to get the color for.

    Returns
        A list of three values representing a RGB color.

        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn(
            'Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)


tum_colors = {
    'bg': np.array([0, 0, 0])/255,
    'human': np.array([34, 114, 227]) / 255,
    'vehicle.bicycle': np.array([0, 182, 0])/255,
    'vehicle.bus': np.array([84, 1, 71])/255,
    'vehicle.car': np.array([189, 101, 0]) / 255,
    'vehicle.motorcycle': np.array([159, 157, 156])/255,
    'vehicle.trailer': np.array([0, 173, 162])/255,
    'vehicle.truck': np.array([89, 51, 0])/255
}


colors = [
    [31, 0, 255],
    [0, 159, 255],
    [255, 95, 0],
    [255, 19, 0],
    [255, 0, 0],
    [255, 38, 0],
    [0, 255, 25],
    [255, 0, 133],
    [255, 172, 0],
    [108, 0, 255],
    [0, 82, 255],
    [0, 255, 6],
    [255, 0, 152],
    [223, 0, 255],
    [12, 0, 255],
    [0, 255, 178],
    [108, 255, 0],
    [184, 0, 255],
    [255, 0, 76],
    [146, 255, 0],
    [51, 0, 255],
    [0, 197, 255],
    [255, 248, 0],
    [255, 0, 19],
    [255, 0, 38],
    [89, 255, 0],
    [127, 255, 0],
    [255, 153, 0],
    [0, 255, 255],
    [0, 255, 216],
    [0, 255, 121],
    [255, 0, 248],
    [70, 0, 255],
    [0, 255, 159],
    [0, 216, 255],
    [0, 6, 255],
    [0, 63, 255],
    [31, 255, 0],
    [255, 57, 0],
    [255, 0, 210],
    [0, 255, 102],
    [242, 255, 0],
    [255, 191, 0],
    [0, 255, 63],
    [255, 0, 95],
    [146, 0, 255],
    [184, 255, 0],
    [255, 114, 0],
    [0, 255, 235],
    [255, 229, 0],
    [0, 178, 255],
    [255, 0, 114],
    [255, 0, 57],
    [0, 140, 255],
    [0, 121, 255],
    [12, 255, 0],
    [255, 210, 0],
    [0, 255, 44],
    [165, 255, 0],
    [0, 25, 255],
    [0, 255, 140],
    [0, 101, 255],
    [0, 255, 82],
    [223, 255, 0],
    [242, 0, 255],
    [89, 0, 255],
    [165, 0, 255],
    [70, 255, 0],
    [255, 0, 172],
    [255, 76, 0],
    [203, 255, 0],
    [204, 0, 255],
    [255, 0, 229],
    [255, 133, 0],
    [127, 0, 255],
    [0, 235, 255],
    [0, 255, 197],
    [255, 0, 191],
    [0, 44, 255],
    [50, 255, 0]
]


def visualize_preprocessed(data_path: str, anchors_path: str):
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

    visualize_targets(image_full, anchors, regression_targets,
                      labels_targets, display=True)


def visualize_targets(image_full, anchors,  regression_targets, labels_targets, display=False, save_path=None):
    """可视化标签数据
    """

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
    mean = np.array([0, 0, 0, 0])
    std = np.array([0.2, 0.2, 0.2, 0.2])
    anchors = anchors[object_index]
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    regression_targets = (regression_targets*std + mean)
    regression_targets[:, 0] = regression_targets[:, 0]*anchor_widths
    regression_targets[:, 1] = regression_targets[:, 1]*anchor_heights
    regression_targets[:, 2] = regression_targets[:, 2]*anchor_widths
    regression_targets[:, 3] = regression_targets[:, 3]*anchor_heights

    regression_targets = (regression_targets + anchors).astype(int)

    # 恢复标签
    # labels_targets.shape = (42975, 9) = (anchors_num, cls_num+1)
    # # labels_targets[:,-1] = anchor_state (-1=ignore, 0=ground, 1=object)
    labels_targets = labels_targets[object_index]
    labels_targets = labels_targets[:, :-1]
    labels_targets = np.argmax(labels_targets, axis=1)

    visualize_result(image, regression_targets,
                     labels_targets, display, save_path)


def visualize_result(image, bboxes, labels, display=False, save_path=None):
    """可视化预测框
    """
    # 绘制参数
    thickness = 2
    for regression, label in zip(bboxes, labels):
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

    if display:
        cv2.imshow("debug", image)
        cv2.waitKey(0)

    if save_path:
        cv2.imwrite(save_path, image)


if __name__ == '__main__':

    import os
    import sys
    sys.path.append(os.getcwd())

    from utils.config import get_config
    config = get_config()

    data_root = os.path.join(config.preprocessed_data_dir, config.nusc_version)
    data_path = os.path.join(data_root, '00046')
    anchor_path = os.path.join(
        config.preprocessed_data_dir, 'anchors_xyxy_absolute.npy')

    visualize_preprocessed(data_path, anchor_path)
