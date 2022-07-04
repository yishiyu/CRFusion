"""
This is for preprocessing the radar. e.g. normalization

The std_map and mean_map has been calculated by 
using all radar data from nuScenes
"""

# 3rd Party Libraries
import numpy as np

# Constants
MINIMAL_STD = 0.1

# Local Libraries
channel_map = {
    0: 'x',
    1: 'y',
    2: 'z',
    3: 'dyn_prop',
    4: 'id',
    5: 'rcs',
    6: 'vx',
    7: 'vy',
    8: 'vx_comp',
    9: 'vy_comp',
    10: 'is_quality_valid',
    11: 'ambig_state',
    12: 'x_rms',
    13: 'y_rms',
    14: 'invalid_state',
    15: 'pdh0',
    16: 'vx_rms',
    17: 'vy_rms',
    18: 'distance',
    19: 'azimuth',
    20: 'vrad_comp'
 }

mean_map =  {
    'ambig_state': 3.0,
    'azimuth' : 0.0,
    'distance': 53.248833,
    'dyn_prop': 1.6733711,
    'id': 50.728333,
    'invalid_state': 0.0,
    'is_quality_valid': 1.0,
    'pdh0': 1.0315487,
    'rcs': 8.17769,
    'vrad_comp' : 0.0,
    'vx': 1.3686545,
    'vx_comp': -0.022377603,
    'vx_rms': 16.305613,
    'vy': 0.049794517,
    'vy_comp': -0.009300133,
    'vy_rms': 3.0,
    'x': 49.94819,
    'x_rms': 19.530258,
    'y': 0.0,
    'y_rms': 19.876368,
    'z': 0.0,
 }


std_map = {
    'ambig_state': 2.3841858e-07,
    'azimuth' : 0.41397777,
    'distance': 36.195225,
    'dyn_prop': 1.3584259,
    'id': 35.54741,
    'invalid_state': 0.0,
    'is_quality_valid': 0.0,
    'pdh0': 0.24920322,
    'rcs': 7.5784483,
    'vrad_comp' : 1.9210424,
    'vx': 6.286945,
    'vx_comp': 1.4791956,
    'vx_rms': 0.5934909,
    'vy': 4.8759995,
    'vy_comp': 0.37958348,
    'vy_rms': 2.3841858e-07,
    'x': 36.01416,
    'x_rms': 0.796998,
    'y': 18.806744,
    'y_rms': 1.2855062,
    'z': 0.0,
 } 

normalizing_mask = {
    'ambig_state': True,
    'dyn_prop': True,
    'id': False,
    'invalid_state': True,
    'is_quality_valid': True,
    'pdh0': True,
    'rcs': True,
    'vx': True,
    'vx_comp': True,
    'vx_rms': True,
    'vy': True,
    'vy_comp': True,
    'vy_rms': True,
    'x': True,
    'x_rms': True,
    'y': True,
    'y_rms': True,
    'z': True,
    'distance' : True,
    'azimuth' : True,
    'vrad_comp' : True
 }

# mapping from name to id
channel_map_inverted = {v:k for k,v in channel_map.items()}


def normalize(channel, value, normalization_interval=(-1,1), sigma_factor=1):
    """
    :param channel: the radar channel of the corresponding radar_channel map
    :param value: <float or numpy.array> the value to normalize
    :param sigma_factor: multiples of sigma used for normalizing the value

    :returns: the normalized channel values
    """
    if isinstance(channel, int):
        # convert channel integer into string
        channel = channel_map[channel]

    if normalizing_mask[channel]:
        std = max(std_map[channel], MINIMAL_STD) # we do not want to divide by 0
        normalized_value = (value - mean_map[channel]) / (std*sigma_factor) # standardize to [-1, 1]
        normalized_value = ((normalized_value + 1) / 2) # standardize to [0, 1]
        normalized_value = (normalized_value * (normalization_interval[1] - normalization_interval[0])) + normalization_interval[0] # normalization interval
        return normalized_value
    else:
        # The value is ignored by the normalizing mask
        return value


def denormalize(channel, value, normalization_interval=(-1,1), sigma_factor=1):
    """
    :param channel: the radar channel of the corresponding radar_channel map
    :param value: <float or numpy.array> the value to normalize
    :param sigma_factor: multiples of sigma used for normalizing the value

    :returns: the normalized channel values
    """
    if isinstance(channel, int):
        # convert channel integer into string
        channel = channel_map[channel]

    if normalizing_mask[channel]:
        std = max(std_map[channel],MINIMAL_STD)

        denormalized_value = (value - normalization_interval[0]) / (normalization_interval[1] - normalization_interval[0]) # [0,1]
        denormalized_value = ((denormalized_value * 2) -1) # standardize to [-1, 1]
        denormalized_value = denormalized_value* (std*sigma_factor) + mean_map[channel]
        return denormalized_value
    else:
        # The value is ignored by the normalizing mask
        return value

def filter_radar_byDist(radar_data, distance):
    """
    :param radar_data: axis0 is channels, axis1 is points
    :param distance: [float] -1 for no distance filtering
    """
    if distance > 0:
        no_of_points = radar_data.shape[1]
        deleter = 0
        for point in range(0, no_of_points):
            dist = np.sqrt(radar_data[0,point - deleter]**2 + radar_data[1,point - deleter]**2)
            if dist > distance:
                radar_data = np.delete(radar_data, point - deleter, 1) 
                deleter += 1
    
    return radar_data

def calculate_distances(radar_data):
    """
    :param radar_data: axis0 is channels, axis1 is points
    """
    dist = np.sqrt(radar_data[0,:]**2 + radar_data[1,:]**2)
    return dist


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
