import os
import torch
import numpy as np
from torch.utils import data
from pathlib import Path
from torchvision import transforms


class NuscenesDataset(data.Dataset):

    std = [0.229, 0.224, 0.225]
    mean=[0.485, 0.456, 0.406]

    def __init__(self,
                 version='v1.0-mini',
                 config=None):

        self.config = config

        self.ROOT = Path(os.path.join(config.preprocessed_data_dir, version))
        assert self.ROOT.exists(), "dataset {} does not exist.".format(version)

        # 获取该路径下的文件夹目录
        self.samples = list(self.ROOT.iterdir())
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=NuscenesDataset.mean, std=NuscenesDataset.std),
        ])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image_file = sample / 'image.npy'
        radar_file = sample / 'radar.npy'
        labels_targets_file = sample / 'labels_targets.npy'
        regression_target_file = sample / 'regression_targets.npy'

        # 图像数据
        with open(image_file, 'rb') as file:
            image = np.load(file).astype(np.uint8)

        # 雷达数据
        with open(radar_file, 'rb') as file:
            radar = torch.from_numpy(np.load(file))

        with open(labels_targets_file, 'rb') as file:
            labels_targets = torch.from_numpy(np.load(file))

        with open(regression_target_file, 'rb') as file:
            regression_target = torch.from_numpy(np.load(file))

        image = self.preprocess(image)
        image_full = torch.cat((image,radar))

        return image_full, regression_target, labels_targets

    @staticmethod
    def collate_fn(image_dropout):
        """
        指导dataloader如何把不同数据中不同数量的目标(框)合并成一个batch
        调用后生成一个合并函数
        image_dropout: 图片清空的概率
        """

        def collecter(batch):
            
            images = torch.zeros((len(batch),*batch[0][0].shape))
            bboxes = torch.zeros((len(batch),*batch[0][1].shape))
            labels = torch.zeros((len(batch),*batch[0][2].shape))

            for i,(image, bbox, label) in enumerate(batch):
                if np.random.rand() < image_dropout:
                    pass
                else:
                    images[i] = image
                bboxes[i] = bbox
                labels[i] = label

            return images, bboxes, labels

        return collecter
