import numpy as np
from nuscenes.nuscenes import NuScenes
from os import path
from PIL import Image
from nuscenes.utils.data_classes import RadarPointCloud


def enrich_radar_data(data):
    """为雷达数据计算额外的特征(距离-distance,方位角-azimuth,径向速度-radial_velocity)()

    Args:
        data (ndarray): radar data
            [0]: x front距离
            [1]: y left距离
            [2]: z z轴信息(全为0)
            [3]: dyn_prop 运动特征(Dynamic property),(0-moving,1-stationary...)
            [4]: id
            [5]: rcs 雷达截面积(Radar cross-section)
            [6]: vx 原始x速度
            [7]: vy 原始y速度
            [8]: vx_comp ego-motion补偿后的x速度
            [9]: vy_comp ego-motion补偿后的y速度
            [10]: is_quality_valid 雷达点簇状态(state of Cluster validity state)
            [11]: ambig_state State of Doppler (radial velocity) ambiguity solution.
            [12]: x_rms (均方根误差,root mean square?猜测)
            [13]: y_rms
            [14]: invalid_state
            [15]: pdh0 False alarm probability of cluster
            [16]: vx_rms
            [17]: vy_rms

    Return:
        data(ndarray):
            ...
            [18]: distance 距离
            [19]: azimuth 方位角
            [20]: radial_velocity 径向距离
    """
    # 计算距离(distance)
    # dist.shape = (1, 雷达点数)
    dist = np.sqrt(data[0, :]**2 + data[1, :]**2)
    dist = np.expand_dims(dist, axis=0)

    # 计算方位角(azimuth)
    azimuth = np.arctan2(data[1, :], data[0, :])
    azimuth = np.expand_dims(azimuth, axis=0)

    # 计算径向速度radial_velocity
    # 归一化距离向量,
    radial = np.array([data[0, :], data[1, :]])
    radial = radial / np.linalg.norm(radial, axis=0, keepdims=True)
    # 将速度向量投影到距离向量上
    v = np.array([data[8, :], data[9, :]])
    radial_velocity = np.sum(v*radial, axis=0, keepdims=True)

    data_collections = [
        data,
        dist,
        azimuth,
        radial_velocity
    ]
    enriched_radar_data = np.concatenate(data_collections, axis=0)

    return enriched_radar_data


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

        # 将img像素映射到[0,1]
        if np.issubdtype(dtype, np.floating):
            data = data / 255
    else:
        raise Exception("\"%s\" is not supported" % sensor_channel)

    return data
