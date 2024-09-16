import argparse
import os

import cv2 as cv
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloader import InferenceDataset
from model.cycle_gan import Generator

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    help="Path to the input dataset",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="outputs/inference/test",
    help="Path to the output folder",
)
parser.add_argument(
    "--model_path", type=str, default=None, help="Path to the i2i model weights file"
)
parser.add_argument(
    "--translation_direction",
    type=str,
    default="BA",
    help="Direction of translation, 'BA' or 'AB'",
)
parser.add_argument(
    "--num_channels_A", type=int, default=3, help="Number of channels of dataset A"
)
parser.add_argument(
    "--num_channels_B", type=int, default=3, help="Number of channels of dataset B"
)
parser.add_argument("--device", type=str, default="cuda")

opt = parser.parse_args()
print(opt)

os.makedirs(opt.output_path, exist_ok=True)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    dataset_B = InferenceDataset(
        folder_path=opt.dataset,
        transform=transform,
    )

    if opt.model_path:
        model_dict = torch.load(opt.model_path, weights_only=True)

        if opt.translation_direction == "BA":
            model_gen_dict = model_dict["gen_BA"]
            generator = Generator(opt.num_channels_B, opt.num_channels_A).to(opt.device)
        elif opt.translation_direction == "AB":
            model_gen_dict = model_dict["gen_AB"]
            generator = Generator(opt.num_channels_A, opt.num_channels_B).to(opt.device)
        else:
            assert (
                "Invalid input for translation_direction: %s"
                % opt.translation_direction
            )

        generator.load_state_dict(model_gen_dict)
        generator.eval()

        dataset_B = DataLoader(dataset_B, batch_size=1, shuffle=False)

        for img, filename in dataset_B:
            fake = generator(img.to(opt.device))
            fake = (fake.squeeze().permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5) * 255
            fake_save_path = os.path.join(opt.output_path, filename[0])
            cv.imwrite(fake_save_path, cv.cvtColor(fake, cv.COLOR_RGB2BGR))

    else:
        assert "Model for i2i translation is missing!"
