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
from utils.utils import AverageMeter


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

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    start = time.time()

    for i, (images, bboxes, labels) in enumerate(train_loader):
        # ??????????????????
        data_time.update(time.time() - start)

        # move to default device
        images = images.to(device)
        bboxes = bboxes.to(device)
        labels = labels.to(device)

        # ????????????
        predicted_loc, predicted_cls = model(images)

        # ??????loss
        loss = loss_fn(predicted_loc, predicted_cls, bboxes, labels)

        # ????????????
        optimizer.zero_grad()
        loss.backward()

        # ????????????
        optimizer.step()

        losses.update(loss)
        batch_time.update(time.time() - start)
        start = time.time()

        # ??????????????????
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    # ????????????,????????????
    del predicted_loc, predicted_cls, images, bboxes, labels


def evaluate(val_loader, model, save_path=None, render=False):
    images, bboxes_gt, labels_gt = next(iter(val_loader))
    
    model.eval()

    filtered_result = model.predict(images.to(device))

    if save_path and not os.path.exists(save_path):
        os.mkdir(save_path)

    # ??????batch
    for i, (boxes, scores, labels) in enumerate(filtered_result):
        boxes = boxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        image = images[i][:3].cpu().detach().numpy()
        for j in range(3):
            image[j] = image[j]*NuscenesDataset.std[j] + NuscenesDataset.mean[j]
        image = (image*255).astype(np.uint8)
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

    # ??????????????????
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default=os.path.join(FILE_DIRECTORY, "config/default.cfg"))
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            "ERROR: Config file \"%s\" not found" % (args.config))
    else:
        config = get_config(args.config)

    if not os.path.exists(config.checkpoints_dir):
        os.mkdir(config.checkpoints_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ????????????,???????????????,???????????????
    train_loader, test_loader, val_loader = get_data_loader(config)

    # ??????????????????
    checkpoint_dir = config.checkpoints_dir
    start_epoch = config.start_epoch
    epochs = config.epochs

    if start_epoch != 0:
        # ???checkpoint???????????????
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
        # ????????????
        model = CRFNet(opts=config, load_pretrained_vgg=True).to(device)
        evaluate(val_loader, model, save_path='log/epoch{}'.format(start_epoch))

        # ????????????
        lr = config.learning_rate
        # TODO ??? ecay_lr_at ???epoch??????????????????
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
        # ??????lr

        # ????????????epoch
        train(train_loader, model, crf_loss, optimizer, epoch)

        # save checkpoint
        save_checkpoint(checkpoint_dir, epoch, model, optimizer)

        # ?????????????????????
        evaluate(val_loader, model,
                 save_path=os.path.join(config.log_dir,
                                        'epoch{}'.format(epoch)))
