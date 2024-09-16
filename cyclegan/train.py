import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from dataloader import TrainDatasets
from model.cycle_gan import Discriminator, Generator
from model.loss_functions import get_disc_loss, get_gen_loss
from utils import plot_loss, show_tensor_images, weights_init

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_A",
    type=str,
    help="Path to the folder of the dataset A",
)
parser.add_argument(
    "--dataset_B",
    type=str,
    help="Path to the folder of the dataset B",
)
parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
parser.add_argument(
    "--i2i_save_model",
    type=bool,
    default=True,
    help="Flag that specifies whether to save model or not",
)
parser.add_argument(
    "--i2i_save_all",
    type=bool,
    default=False,
    help="Flag that specifies whether to save all weights or only generator",
)
parser.add_argument(
    "--num_channels_A", type=int, default=3, help="Number of channels of dataset A"
)
parser.add_argument(
    "--num_channels_B", type=int, default=3, help="Number of channels of dataset B"
)
parser.add_argument("--target_shape", type=str, default=256)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--n_epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lambda_identity", type=float, default=1.0)
parser.add_argument("--lambda_cycle", type=int, default=10)
parser.add_argument("--lambda_adv", type=int, default=1)
parser.add_argument("--pretrained", type=int, default=False)
parser.add_argument("--pretrained_weights", type=int, default=None)

opt = parser.parse_args()
print(opt)

now = datetime.now()

ROOT = "outputs/"
NAME = opt.experiment_name + now.strftime("%d%m%Y_%H%M%S")

OUTPUT_DIR = os.path.join(ROOT, NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "configuration.txt"), "w") as f:
    json.dump(opt.__dict__, f, indent=2)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    os.makedirs(os.path.join(OUTPUT_DIR, "training", "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "training", "losses"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "training", "weights"), exist_ok=True)

    datasets = TrainDatasets(
        dataset_A=opt.dataset_A,
        dataset_B=opt.dataset_B,
        transform=transform,
    )

    adv_criterion = nn.MSELoss()
    recon_criterion = nn.L1Loss()

    mean_generator_loss = 0
    mean_discriminator_loss = 0

    datasets = DataLoader(datasets, batch_size=opt.batch_size, shuffle=True)

    gen_AB = Generator(opt.num_channels_A, opt.num_channels_B).to(opt.device)
    gen_BA = Generator(opt.num_channels_B, opt.num_channels_A).to(opt.device)
    gen_opt = torch.optim.Adam(
        list(gen_AB.parameters()) + list(gen_BA.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999),
    )
    disc_A = Discriminator(opt.num_channels_A).to(opt.device)
    disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    disc_B = Discriminator(opt.num_channels_B).to(opt.device)
    disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    if opt.pretrained:
        pre_dict = torch.load(opt.pretrained_weights)
        gen_AB.load_state_dict(pre_dict["gen_AB"])
        gen_BA.load_state_dict(pre_dict["gen_BA"])
        gen_opt.load_state_dict(pre_dict["gen_opt"])
        disc_A.load_state_dict(pre_dict["disc_A"])
        disc_A_opt.load_state_dict(pre_dict["disc_A_opt"])
        disc_B.load_state_dict(pre_dict["disc_B"])
        disc_B_opt.load_state_dict(pre_dict["disc_B_opt"])
    else:
        gen_AB = gen_AB.apply(weights_init)
        gen_BA = gen_BA.apply(weights_init)
        disc_A = disc_A.apply(weights_init)
        disc_B = disc_B.apply(weights_init)

    gen_loss_list = []
    disc_loss_list = []

    for epoch in range(opt.n_epochs):
        disc_loss_sum = 0
        gen_loss_sum = 0
        step = 0
        for real_A, real_B in tqdm(datasets):
            real_A = nn.functional.interpolate(real_A, size=opt.target_shape).to(
                opt.device
            )
            real_B = nn.functional.interpolate(real_B, size=opt.target_shape).to(
                opt.device
            )

            ### Update discriminator A ###
            disc_A_opt.zero_grad()
            with torch.no_grad():
                fake_A = gen_BA(real_B)

            disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True)
            disc_A_opt.step()

            ### Update discriminator B ###
            disc_B_opt.zero_grad()
            with torch.no_grad():
                fake_B = gen_AB(real_A)

            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True)
            disc_B_opt.step()

            ### Update generator ###
            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A,
                real_B,
                gen_AB,
                gen_BA,
                disc_A,
                disc_B,
                adv_criterion,
                recon_criterion,
                recon_criterion,
                opt.lambda_identity,
                opt.lambda_cycle,
            )
            gen_loss.backward()
            gen_opt.step()

            disc_loss_sum += disc_A_loss.item()
            gen_loss_sum += gen_loss.item()

            step += 1

        mean_disc_loss = disc_loss_sum / step
        mean_gen_loss = gen_loss_sum / step

        gen_loss_list.append(mean_gen_loss)
        disc_loss_list.append(mean_disc_loss)

        ### Visualization code ###
        print(
            f"Epoch {epoch}: Step {step}: Generator (U-Net) loss: {mean_gen_loss}, Discriminator loss: {mean_disc_loss}"
        )
        show_tensor_images(
            torch.cat([real_A, real_B]),
            save_path=os.path.join(
                OUTPUT_DIR, "training", "images", "img_e{}_realA_realB".format(epoch)
            ),
            size=(opt.num_channels_A, opt.target_shape, opt.target_shape),
        )
        show_tensor_images(
            torch.cat([fake_B, fake_A]),
            save_path=os.path.join(
                OUTPUT_DIR, "training", "images", "img_e{}_fakeB_fakeA".format(epoch)
            ),
            size=(opt.num_channels_B, opt.target_shape, opt.target_shape),
        )

        plot_loss(
            x_axis=range(len(gen_loss_list)),
            y_axis=gen_loss_list,
            title="generator loss",
            epoch=epoch,
            save_path=os.path.join(OUTPUT_DIR, "training", "losses"),
        )
        plot_loss(
            x_axis=range(len(disc_loss_list)),
            y_axis=disc_loss_list,
            title="discriminator loss",
            epoch=epoch,
            save_path=os.path.join(OUTPUT_DIR, "training", "losses"),
        )

        plt.figure(figsize=(18, 10))
        plt.plot(range(len(gen_loss_list)), gen_loss_list, label="generator loss")
        plt.plot(range(len(disc_loss_list)), disc_loss_list, label="discriminator loss")
        plt.xlabel("Epoch")
        plt.ylabel("Losses")
        plt.legend()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "training",
                "losses",
                "all_losses_epoch_" + str(epoch) + ".png",
            )
        )
        plt.clf()
        plt.close()

        if epoch % 5 == 0:
            torch.save(
                {
                    "gen_AB": gen_AB.state_dict(),
                    "gen_BA": gen_BA.state_dict(),
                    "gen_opt": gen_opt.state_dict(),
                    "disc_A": disc_A.state_dict(),
                    "disc_A_opt": disc_A_opt.state_dict(),
                    "disc_B": disc_B.state_dict(),
                    "disc_B_opt": disc_B_opt.state_dict(),
                },
                os.path.join(
                    OUTPUT_DIR,
                    "training",
                    "weights",
                    "cycleGAN_epoch" + str(epoch) + ".pth",
                ),
            )

    if opt.i2i_save_model:
        if opt.i2i_save_all:
            torch.save(
                {
                    "gen_AB": gen_AB.state_dict(),
                    "gen_BA": gen_BA.state_dict(),
                    "gen_opt": gen_opt.state_dict(),
                    "disc_A": disc_A.state_dict(),
                    "disc_A_opt": disc_A_opt.state_dict(),
                    "disc_B": disc_B.state_dict(),
                    "disc_B_opt": disc_B_opt.state_dict(),
                },
                os.path.join(OUTPUT_DIR, "final_weights.pth"),
            )
        else:
            torch.save(
                {
                    "gen_AB": gen_AB.state_dict(),
                    "gen_BA": gen_BA.state_dict(),
                },
                os.path.join(OUTPUT_DIR, "final_weights.pth"),
            )
