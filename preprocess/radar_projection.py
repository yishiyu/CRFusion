import numpy as np


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes
    from utils import get_sensor_sample_data

    nusc = NuScenes(version='v1.0-mini',
                    dataroot='data/nuscenes',
                    verbose=True)
    radar_channel = 'RADAR_FRONT'
    camera_channel = 'CAM_FRONT'

    # 获取sample
    scene_tokens = nusc.scene[0]['token']
    scene_record = nusc.get('scene', scene_tokens)
    sample_record = nusc.get('sample', scene_record['first_sample_token'])

    # 获取sensor token
    radar_token = sample_record['data'][radar_channel]
    camera_token = sample_record['data'][camera_channel]

    # 获取sensor data
    radar_data = get_sensor_sample_data(nusc, sample_record, radar_channel)
    image_data = get_sensor_sample_data(nusc, sample_record, camera_channel)

    # 融合参数
    image_target_shape = (450, 450)
    #'FOV'
    height = (0, 3)


    pass
