import configparser
import ast


class Configuration():
    def __init__(self, config):
        # 路径参数
        self.data_dir = config['PATH']['data_dir']
        self.preprocessed_data_dir = config['PATH']['preprocessed_data_dir']
        self.checkpoints_dir = config['PATH']['checkpoints_dir']
        self.log_dir = config['PATH']['log_dir']

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
        self.image_size = (config.getint('DATAFUSION', 'image_height'),
                           config.getint('DATAFUSION', 'image_width'))
        self.radar_projection_height = \
            config.getfloat('DATAFUSION', 'radar_projection_height')
        self.channels = \
            ast.literal_eval(config.get('DATAFUSION', 'channels'))

        # 预处理参数
        self.only_radar_annotated = \
            config.getint('PREPROCESSING', 'only_radar_annotated')
        self.positive_overlap = \
            config.getfloat('PREPROCESSING', 'positive_overlap')
        self.negative_overlap = \
            config.getfloat('PREPROCESSING', 'negative_overlap')
        self.process_pool_size = \
            config.getint('PREPROCESSING','process_pool_size')

        # 训练超参数
        self.focal_loss_alpha = config.getfloat(
            'HYPERPARAMETERS', 'focal_loss_alpha')
        self.focal_loss_gamma = config.getfloat(
            'HYPERPARAMETERS', 'focal_loss_gamma')
        self.learning_rate = config.getfloat(
            'HYPERPARAMETERS', 'learning_rate')
        self.num_workders = config.getint(
            'HYPERPARAMETERS', 'num_workders')
        self.batchsize = config.getint('HYPERPARAMETERS', 'batchsize')
        self.epochs = config.getint('HYPERPARAMETERS', 'epochs')
        self.start_epoch = config.getint('HYPERPARAMETERS', 'start_epoch')
        self.image_dropout = config.getfloat(
            'HYPERPARAMETERS', 'image_dropout')

        # 模型Filter
        self.nms = config.getboolean('MODELFILTER', 'nms')
        self.nms_threshold = config.getfloat(
            'MODELFILTER', 'nms_threshold')
        self.score_threshold = config.getfloat(
            'MODELFILTER', 'score_threshold')
        self.max_detections = config.getint(
            'MODELFILTER', 'max_detections')


def get_config(config_file='config/default.cfg'):
    config = configparser.ConfigParser()
    config.read(config_file)

    cfg = Configuration(config)
    return cfg
