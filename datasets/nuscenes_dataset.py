import os
import torch
import numpy as np
from torch.utils import data
from pathlib import Path


class NuscenesDataset(data.Dataset):

    def __init__(self,
                 version='v1.0-mini',
                 config=None):

        self.config = config

        self.ROOT = Path(os.path.join(config.preprocessed_data_dir, version))
        assert self.ROOT.exists(), "dataset {} does not exist.".format(version)

        # 获取该路径下的文件夹目录
        self.samples = list(self.ROOT.iterdir())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image_full_file = sample / 'image_full.npy'
        labels_targets_file = sample / 'labels_targets.npy'
        regression_target_file = sample / 'regression_targets.npy'

        with open(image_full_file, 'rb') as file:
            image_full = torch.from_numpy(np.load(file))

        with open(labels_targets_file, 'rb') as file:
            labels_targets = torch.from_numpy(np.load(file))

        with open(regression_target_file, 'rb') as file:
            regression_target = torch.from_numpy(np.load(file))

        return image_full, regression_target, labels_targets

    @staticmethod
    def collate_fn(image_dropout):
        """
        指导dataloader如何把不同数据中不同数量的目标(框)合并成一个batch
        调用后生成一个合并函数
        image_dropout: 图片清空的概率
        """

        def collecter(batch):
            images = []
            bboxes = []
            labels = []

            for b in batch:
                if np.random.rand() < image_dropout:
                    images.append(torch.zeros(b[0].shape))
                else:
                    images.append(torch.tensor(b[0]))
                bboxes.append(b[1])
                labels.append(b[2])

            images = torch.stack(images, dim=0)
            bboxes = torch.stack(bboxes, dim=0)
            labels = torch.stack(labels, dim=0)

            return images, bboxes, labels

        return collecter
