import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.utils import make_grid


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def plot_loss(x_axis, y_axis, title, epoch, save_path, xlabel="Epoch", ylabel="Loss"):
    plt.figure(figsize=(18, 10))
    plt.plot(x_axis, y_axis)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()
    plt.savefig(os.path.join(save_path, title + "_epoch_" + str(epoch) + ".png"))
    plt.clf()
    plt.close()


def show_tensor_images(
    image_tensor, save_path, num_images=25, size=(1, 28, 28), show=False
):
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()
