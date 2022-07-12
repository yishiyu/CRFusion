import numpy as np
from nuscenes.nuscenes import NuScenes
from os import path
from PIL import Image
from nuscenes.utils.data_classes import RadarPointCloud
from .radar import enrich_radar_data
# from radar import enrich_radar_data


def get_sensor_sample_data(nusc: NuScenes,
                           sample,
                           sensor_channel='RADAR',
                           dtype=np.float32,
                           size=None):
    """根据sample,sensor_channel获取对应的传感器数据(RADAR/CAM)

    Args:
        nusc (NuScenes): nuScenes对象

        sample (dict): sample
        sensor_channel (str): RADAR/CAM. Defaults to RADAR.
        dtype (type, optional): the target numpy type. Defaults to np.float32.
        size (_type_, optional): _description_. Defaults to None.

    Return:
        Image Format:
            - Shape: h x w x 3
            - Channels: RGB
            - size:
                - [int] size to limit image size
                - [tuple[int]] size to limit image size
    """

    # 获取文件路径
    sample_data = nusc.get('sample_data', sample['data'][sensor_channel])
    file_name = path.join(nusc.dataroot, sample_data['filename'])

    if not path.exists(file_name):
        raise FileNotFoundError(
            "nuscenes data must be located in %s" % file_name)

    # 雷达数据
    if 'RADAR' in sensor_channel:
        pc = RadarPointCloud.from_file(file_name)
        data = pc.points.astype(dtype)
        data = enrich_radar_data(data)
    elif 'CAM' in sensor_channel:
        img = Image.open(file_name)

        if size is not None:
            try:
                _ = iter(size)
            except TypeError:
                # not iterable
                # limit both dimension to size, but keep aspect ration
                size = (size, size)
                img.thumbnail(size=size)
            else:
                size = size[::-1]  # revert dimensions
                img = img.resize(size=size)

        data = np.array(img, dtype=dtype)
    else:
        raise Exception("\"%s\" is not supported" % sensor_channel)

    return data

def compute_overlap(
    boxes: np.array,
    query_boxes: np.array
):
    """计算overlap

    Args:
        boxes (np.array): (N, 4)
        query_boxes (np.array): (K, 4)
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float64)
    # box_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
    #             (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw*ih/ua

    return overlaps

