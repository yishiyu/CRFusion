import os
import torch

def save_checkpoint(directory, epoch, model, optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = os.path.join(
        directory, 'checkpoint_crfnet_{0:{1}3d}.pth.tar'.format(epoch,'0'))
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(state, filename)