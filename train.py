from torch.utils.data import DataLoader
import os
import argparse
from datasets.data_visualization import visualize_result
from utils.config import get_config
from datasets.nuscenes_dataset import NuscenesDataset
from model import CRFNet, CRFLoss
import torch
import time
from utils.utils import save_checkpoint
from datasets import visualize_targets
import numpy as np


def get_data_loader(config):
    train_dataset = NuscenesDataset(version=config.nusc_version, config=config)
    # test_dataset = NuscenesDataset(version=config.test_version, config=config)
    # val_dataset = NuscenesDataset(version=config.val_version, config=config)

    train_loader = DataLoader(train_dataset,  batch_size=config.batchsize,
                              collate_fn=train_dataset.collate_fn(
                                  image_dropout=config.image_dropout),
                              shuffle=True, pin_memory=True,
                              num_workers=config.num_workders)

    # test_loader = DataLoader(train_dataset,  batch_size=config.batchsize,
    #                           collate_fn=train_dataset.collate_fn(
    #                               image_dropout=config.image_dropout),
    #                           shuffle=True, pin_memory=True,
    #                           num_workers=config.num_workders)

    # val_loader = DataLoader(train_dataset,  batch_size=config.batchsize,
    #                           collate_fn=train_dataset.collate_fn(
    #                               image_dropout=config.image_dropout),
    #                           shuffle=True, pin_memory=True,
    #                           num_workers=config.num_workders)
    return train_loader, train_loader, train_loader


def train(train_loader, model, loss_fn, optimizer, epoch, print_freq=10):
    """
    One epoch's training.
    """
    model.train()

    start = time.time()

    for i, (images, bboxes, labels) in enumerate(train_loader):
        # 数据加载时间
        data_time = (time.time() - start)

        # move to default device
        images = images.to(device)
        bboxes = bboxes.to(device)
        labels = labels.to(device)

        # 正向传播
        predicted_loc, predicted_cls = model(images)

        # 计算loss
        loss = loss_fn(predicted_loc, predicted_cls, bboxes, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        batch_time = (time.time() - start)
        start = time.time()

        # 输出训练状态
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'Data Time {data_time:.3f}\t'
                  'Loss {loss:.4f}\t'.format(epoch, i, len(train_loader),
                                             batch_time=batch_time,
                                             data_time=data_time,
                                             loss=loss))
    # 清除变量,节省内存
    del predicted_loc, predicted_cls, images, bboxes, labels


def evaluate(val_loader, model, save_path=None, render=False):
    images, bboxes_gt, labels_gt = next(iter(val_loader))
    
    model.eval()

    filtered_result = model.predict(images.to(device))

    if save_path and not os.path.exists(save_path):
        os.mkdir(save_path)

    # 多个batch
    for i, (boxes, scores, labels) in enumerate(filtered_result):
        boxes = boxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        image = images[i].cpu().detach().numpy()
        image = (image[:3]*255).astype(np.uint8)
        image = np.ascontiguousarray(image.transpose(1, 2, 0))

        # select indices which have a score above the threshold
        indices = np.where(scores>config.score_threshold)[0]

        # select those scores
        scores = scores[indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:config.max_detections]

        # select detections
        image_boxes      = boxes[indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[indices[scores_sort]]

        image_path = os.path.join(save_path, '{}.jpg'.format(i))
        visualize_result(image, image_boxes, image_labels, save_path=image_path)
    pass


if __name__ == '__main__':
    FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    # 读取配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default=os.path.join(FILE_DIRECTORY, "config/default.cfg"))
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            "ERROR: Config file \"%s\" not found" % (args.config))
    else:
        config = get_config(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 训练数据,测试数据集,验证数据集
    train_loader, test_loader, val_loader = get_data_loader(config)

    # 训练模型参数
    checkpoint_dir = config.checkpoints_dir
    start_epoch = config.start_epoch
    epochs = config.epochs

    if start_epoch != 0:
        # 从checkpoint中恢复训练
        filename = os.path.join(
            checkpoint_dir, 'checkpoint_crfnet_{0:{1}3d}.pth.tar'.format(start_epoch-1, '0'))
        checkpoint = torch.load(filename)
        print('Load checkpoint from epoch {}'.format(start_epoch-1))
        model = checkpoint['model']
        model.classification.load_activation_layer()
        model.regression.load_activation_layer()
        optimizer = checkpoint['optimizer']

        evaluate(val_loader, model, save_path='log/epoch{}'.format(start_epoch))

    else:
        # 创建模型
        model = CRFNet(opts=config, load_pretrained_vgg=True).to(device)
        evaluate(val_loader, model, save_path='log/epoch{}'.format(start_epoch))

        # 训练参数
        lr = config.learning_rate
        # TODO 在 ecay_lr_at 个epoch后调整学习率
        # decay_lr_at = [5, 8]
        momentum = 0.3
        weight_decay = 5e-4

        parameters = []
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                parameters.append(param)
        optimizer = torch.optim.SGD(parameters,
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    model = model.to(device)
    crf_loss = CRFLoss(num_classes=config.cls_num,
                       focal_loss_alpha=config.focal_loss_alpha,
                       focal_loss_gamma=config.focal_loss_gamma
                       ).to(device)

    for epoch in range(start_epoch, epochs):
        # 调整lr

        # 训练一个epoch
        train(train_loader, model, crf_loss, optimizer, epoch)

        # save checkpoint
        save_checkpoint(checkpoint_dir, epoch, model, optimizer)

        # 使用验证集验证
        evaluate(val_loader, model, save_path='./log/epoch{}'.format(epoch))
