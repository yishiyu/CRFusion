import os
import argparse
from utils.config import get_config
from datasets.nuscenes_dataset import NuscenesDataset


if __name__ == '__main__':
    FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='config/default.cfg')
    args = parser.parse_args()

    # 读取配置文件
    config_file = os.path.join(FILE_DIRECTORY, args.config)
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            "ERROR: Config file \"%s\" not found" % (config_file))
    else:
        config = get_config(config_file)

    NuscenesDataset(version='v1.0-mini', config=config)
