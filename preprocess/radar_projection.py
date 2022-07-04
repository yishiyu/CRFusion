import numpy as np
import cv2
from nuscenes.utils.data_classes import PointCloud
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
import radar

# radar data(enriched)
# [0]: x front距离
# [1]: y left距离
# [2]: z z轴信息(全为0)
# [3]: dyn_prop 运动特征(Dynamic property),(0-moving,1-stationary...)
# [4]: id
# [5]: rcs 雷达截面积(Radar cross-section)
# [6]: vx 原始x速度
# [7]: vy 原始y速度
# [8]: vx_comp ego-motion补偿后的x速度
# [9]: vy_comp ego-motion补偿后的y速度
# [10]: is_quality_valid 雷达点簇状态(state of Cluster validity state)
# [11]: ambig_state State of Doppler (radial velocity) ambiguity solution.
# [12]: x_rms (均方根误差,root mean square?猜测)
# [13]: y_rms
# [14]: invalid_state
# [15]: pdh0 False alarm probability of cluster
# [16]: vx_rms
# [17]: vy_rms
# [18]: distance 距离
# [19]: azimuth 方位角
# [20]: radial_velocity 径向距离


def _resize_image(image_data: np.array, target_shape: tuple):
    """调整图像大小并计算resize matrix

    Args:
        image_data (np.array): 图片数据(height x width x 3)
        target_shape (tuple): 目标大小(width, height)

    Returns:
        resized_image: resize后的图像(height x width x 3)
        resize_matrix: (3 x 3)
    """
    # cv2 shape(height, width)
    target_shape = (target_shape[1], target_shape[0])
    resized_image = cv2.resize(image_data, target_shape)
    resize_matrix = np.eye(3, dtype=resized_image.dtype)
    resize_matrix[1, 1] = target_shape[0]/image_data.shape[0]
    resize_matrix[0, 0] = target_shape[1]/image_data.shape[1]
    return resized_image, resize_matrix


def _radar_transformation(radar_data: np.array, height=(0, 3)):
    """在y轴上扩展雷达点

    Args:
        radar_data (numpy.array): [num_points x 21](inplace 地修改 radar_data[2])
        height (tuple, optional): 雷达点y轴扩展后的范围. Defaults to (0, 3).

    Returns:
        _type_: _description_
    """

    RADAR_HEIGHT = 0.5
    # 雷达点数
    num_points = radar_data.shape[1]

    # TODO 不知道为什么要往下移动0.5
    # 雷达点扩展后终点
    radar_xyz_endpoint = radar_data[0:3, :].copy()
    radar_xyz_endpoint[2, :] = np.ones(
        (num_points,)) * (height[1] - RADAR_HEIGHT)

    # 雷达点扩展后起点
    radar_data[2, :] = np.ones((num_points,)) * (height[0] - RADAR_HEIGHT)

    return radar_data, radar_xyz_endpoint


def _create_vertical_line(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    :param P1: [numpy array] that consists of the coordinate of the first point (x,y)
    :param P2: [numpy array] that consists of the coordinate of the second point (x,y)
    :param img: [numpy array] the image being processed

    :return itbuffer: [numpy array] that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y])     
    """
    # define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    P1_y = int(P1[1])
    P2_y = int(P2[1])
    dX = 0
    dY = P2_y - P1_y
    if dY == 0:
        dY = 1
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(
        shape=(np.maximum(int(dYa), int(dXa)), 2), dtype=np.float32)
    itbuffer.fill(np.nan)

    # vertical line segment
    itbuffer[:, 0] = int(P1[0])
    if P1_y > P2_y:
        # Obtain coordinates along the line using a form of Bresenham's algorithm
        itbuffer[:, 1] = np.arange(P1_y - 1, P1_y - dYa - 1, -1)
    else:
        itbuffer[:, 1] = np.arange(P1_y+1, P1_y+dYa+1)

    # Remove points outside of image
    colX = itbuffer[:, 0].astype(int)
    colY = itbuffer[:, 1].astype(int)
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) &
                        (colX < imageW) & (colY < imageH)]

    return itbuffer


def _radar2camera(image_data, radar_data, radar_xyz_endpoints, clear_radar=False):
    """

    Calculates a line of two radar points and puts the radar_meta data as additonal layers to the image -> image_plus


    :param image_data: [numpy array (900 x 1600 x 3)] of image data
    :param radar_data: [numpy array (xyz+meta x no of points)] that consists of the transformed radar points with z = 0
        default semantics: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms distance
    :param radar_xyz_endpoints: [numpy array (3 x no of points)] that consits of the transformed radar points z = height
    :param clear_radar: [boolean] True if radar data should be all zero

    :return image_plus: a numpy array (900 x 1600 x (3 + number of radar_meta (e.g. velocity)))
    """

    radar_meta_count = radar_data.shape[0]-3
    radar_extension = np.zeros(
        (image_data.shape[0], image_data.shape[1], radar_meta_count), dtype=np.float32)
    no_of_points = radar_data.shape[1]

    if clear_radar:
        pass  # we just don't add it to the image
    else:
        for radar_point in range(0, no_of_points):
            projection_line = _create_vertical_line(
                radar_data[0:2, radar_point], radar_xyz_endpoints[0:2, radar_point], image_data)

            for pixel_point in range(0, projection_line.shape[0]):
                y = projection_line[pixel_point, 1].astype(int)
                x = projection_line[pixel_point, 0].astype(int)

                # Check if pixel is already filled with radar data and overwrite if distance is less than the existing
                if not np.any(radar_extension[y, x]) or radar_data[-1, radar_point] < radar_extension[y, x, -1]:
                    radar_extension[y, x] = radar_data[3:, radar_point]

    image_plus = np.concatenate((image_data, radar_extension), axis=2)

    return image_plus


def map_pointcloud_to_image(nusc, radar_points, pointsensor_token, camera_token, target_resolution=(None, None)):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param radar_pints: [list] list of radar points
    :param pointsensor_token: [str] Lidar/radar sample_data token.
    :param camera_token: [str] Camera sample_data token.
    :param target_resolution: [tuple of int] determining the output size for the radar_image. None for no change

    :return (points <np.float: 2, n)
    """

    # Initialize the database
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)

    pc = PointCloud(radar_points)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor',
                         pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).

    # intrinsic_resized = np.matmul(camera_resize, np.array(cs_record['camera_intrinsic']))
    view = np.array(cs_record['camera_intrinsic'])
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = pc.points
    points[:3] = view_points(
        pc.points[:3], view, normalize=True)  # resize here

    # Resizing to target resolution
    if target_resolution[1]:  # resizing width
        points[0, :] *= (target_resolution[1]/cam['width'])

    if target_resolution[0]:  # resizing height
        points[1, :] *= (target_resolution[0]/cam['height'])

    # actual_resolution = (cam['height'], cam['width'])
    # for i in range(len(target_resolution)):
    #     if target_resolution[i]:
    #         points[i,:] *= (target_resolution[i]/actual_resolution[i])

    return points


def imageplus_creation(nusc, image_data: np.array, radar_data: np.array,
                       pointsensor_token: str, camera_token: str,
                       height=(0, 3), image_target_shape=(900, 1600),
                       clear_radar=False, clear_image=False):
    """_summary_

    Args:
        nusc (NuScenes): NuScenes数据集对象
        image_data (np.array): 图像数据(900x1600x3)
        radar_data (np.array): enriched radar data
        pointsensor_token (str): sample token
        camera_token (str): camera token
        height (tuple, optional): 映射后的雷达高度. Defaults to (0, 3).
        image_target_shape (tuple, optional): 目标大小. Defaults to (900, 1600).
        clear_radar (bool, optional): 是否清除雷达. Defaults to False.
        clear_image (bool, optional): 是否清除图像(BlackIn). Defaults to False.

    Returns:
        image_plus: image_target_shape x 21(RGB通道+Radar的3-21通道)
    """

    # Resize 图像
    cur_img, camera_resize = _resize_image(image_data, image_target_shape)

    # 获取雷达点映射后的坐标
    radar_points, radar_xyz_endpoint = _radar_transformation(
        radar_data, height)

    if clear_image:
        cur_img.fill(0)

    radar_points = map_pointcloud_to_image(
        nusc, radar_points, pointsensor_token=pointsensor_token, camera_token=camera_token, target_resolution=image_target_shape)
    radar_xyz_endpoint = map_pointcloud_to_image(
        nusc, radar_xyz_endpoint, pointsensor_token=pointsensor_token, camera_token=camera_token, target_resolution=image_target_shape)

    image_plus = _radar2camera(
        cur_img, radar_points, radar_xyz_endpoint, clear_radar=clear_radar)

    return image_plus


def create_imagep_visualization(image_plus_data, color_channel="distance",
                                draw_circles=False, cfg=None, radar_lines_opacity=1.0):
    """
    Visualization of image plus data

    Parameters:
        :image_plus_data: a numpy array (900 x 1600 x (3 + number of radar_meta (e.g. velocity)))
        :image_data: a numpy array (900 x 1600 x 3)
        :color_channel: <str> Image plus channel for colorizing the radar lines. according to radar.channel_map.
        :draw_circles: Draws circles at the bottom of the radar lines
    Returns:
        :image_data: a numpy array (900 x 1600 x 3)
    """
    # read dimensions
    image_plus_height = image_plus_data.shape[0]
    image_plus_width = image_plus_data.shape[1]
    n_channels = image_plus_data.shape[2]

    ##### Extract the image Channels #####
    if cfg is None:
        image_channels = [0, 1, 2]
    else:
        image_channels = [i_ch for i_ch in cfg.channels if i_ch in [0, 1, 2]]
    image_data = np.ones(shape=(*image_plus_data.shape[:2], 3))
    if len(image_channels) > 0:
        image_data[:, :, image_channels] = image_plus_data[:, :,
                                                           image_channels].copy()  # copy so we dont change the old image

    # Draw the Horizon
    image_data = np.array(image_data*255).astype(np.uint8)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

    ##### Paint every augmented pixel on the image #####
    if n_channels > 3:
        # transfer it to the currently selected channels
        if cfg is None:
            print("Warning, no cfg provided. Thus, its not possible to find out \
                which channel shall be used for colorization")
            # we expect the channel index to be the last axis
            radar_img = np.zeros(image_plus_data.shape[:-1])
        else:
            available_channels = {
                radar.channel_map[ch]: ch_idx for ch_idx, ch in enumerate(cfg.channels) if ch > 2}
            ch_idx = available_channels[color_channel]
            # Normalize the radar
            if cfg.normalize_radar:  # normalization happens from -127 to 127
                radar_img = image_plus_data[..., ch_idx] + 127.5
            else:
                radar_img = radar.normalize(color_channel, image_plus_data[..., ch_idx],
                                            normalization_interval=[0, 255], sigma_factor=2)

            radar_img = np.clip(radar_img, 0, 255)

        radar_colormap = np.array(cv2.applyColorMap(
            radar_img.astype(np.uint8), cv2.COLORMAP_AUTUMN))

        for x in range(0, image_plus_width):
            for y in range(0, image_plus_height):
                radar_channels = image_plus_data[y, x, 3:]
                pixel_contains_radar = np.count_nonzero(radar_channels)
                if not pixel_contains_radar:
                    continue

                radar_color = radar_colormap[y, x]
                for pixel in [(y, x)]:  # [(y,x-1),(y,x),(y,x+1)]:
                    if image_data.shape > pixel:

                        # Calculate the color
                        pixel_color = np.array(
                            image_data[pixel][0:3], dtype=np.uint8)
                        pixel_color = np.squeeze(cv2.addWeighted(
                            pixel_color, 1-radar_lines_opacity, radar_color, radar_lines_opacity, 0))

                        # Draw on image
                        image_data[pixel] = pixel_color

                # only if some radar information is there
                if draw_circles:
                    if image_plus_data.shape[0] > y+1 and not np.any(image_plus_data[y+1, x, 3:]):
                        cv2.circle(image_data, (x, y), 3, color=radar_colormap[(
                            y, x)].astype(np.float), thickness=1)

    return image_data


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
    image_target_shape = (900, 1600)
    height = (0, 3)

    image_plus_data = imageplus_creation(nusc,
                                         image_data, radar_data,
                                         radar_token, camera_token,
                                         height, image_target_shape,
                                         clear_radar=False, clear_image=False)

    # Visualize the result
    imgp_viz = create_imagep_visualization(image_plus_data)
    cv2.imshow('image', imgp_viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
