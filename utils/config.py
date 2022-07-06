import configparser
import ast


def get_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    class Configuration():
        def __init__(self):
            # 路径参数
            self.data_dir = config['PATH']['data_dir']
            self.preprocessed_data_dir = config['PATH']['preprocessed_data_dir']
            self.checkpoints_dir = config['PATH']['checkpoints_dir']

            # 数据集参数
            self.nusc_version = config['DATASET']['nusc_version']
            self.test_version = config['DATASET']['test_version']
            self.val_version = config['DATASET']['val_version']
            self.n_sweeps = config.getint('DATASET', 'n_sweeps')

            try:
                self.category_mapping = dict(config['CATEGORY_MAPPING'])
            except:
                self.category_mapping = {
                    "vehicle.car": "vehicle.car",
                    "vehicle.motorcycle": "vehicle.motorcycle",
                    "vehicle.bicycle": "vehicle.bicycle",
                    "vehicle.bus": "vehicle.bus",
                    "vehicle.truck": "vehicle.truck",
                    "vehicle.emergency": "vehicle.truck",
                    "vehicle.trailer": "vehicle.trailer",
                    "human": "human", }

            # 可预测类别+背景
            self.cls_num = 7+1

            # 融合参数
            self.image_size = (config.getint('DATAFUSION', 'image_width'),
                               config.getint('DATAFUSION', 'image_height'))
            self.radar_projection_height = \
                config.getfloat('DATAFUSION', 'radar_projection_height')
            self.channels = \
                ast.literal_eval(config.get('DATAFUSION', 'channels'))
            self.only_radar_annotated = \
                config.getint('PREPROCESSING', 'only_radar_annotated')

    cfg = Configuration()
    return cfg
