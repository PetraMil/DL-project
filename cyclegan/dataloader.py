import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class TrainDatasets(Dataset):
    def __init__(self, dataset_A, dataset_B, transform, do_randperm=True):
        self.transform = transform
        self.do_randperm = do_randperm
        self.files_A = sorted(glob.glob(dataset_A + "/*.png"))
        self.files_B = sorted(glob.glob(dataset_B + "/*.png"))

        self.min_len = min(len(self.files_A), len(self.files_B))
        self.max_len = max(len(self.files_A), len(self.files_B))

        if self.do_randperm:
            self.new_perm()

    def new_perm(self):
        self.randperm = torch.randperm(self.max_len)[: self.min_len]

    def __getitem__(self, index):
        if self.do_randperm:
            if len(self.files_A) > len(self.files_B):
                item_A = self.transform(Image.open(self.files_A[self.randperm[index]]))
                item_B = self.transform(Image.open(self.files_B[index % self.min_len]))
            else:
                item_A = self.transform(Image.open(self.files_A[index % self.min_len]))
                item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        else:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        if index == len(self) - 1:
            self.new_perm()

        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return self.min_len


class InferenceDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.transform = transform
        self.files = sorted(glob.glob(folder_path + "/*.png"))

    def __getitem__(self, index):
        image = self.transform(Image.open(self.files[index % len(self.files)]))

        return (image - 0.5) * 2, os.path.basename(self.files[index % len(self.files)])

    def __len__(self):
        return len(self.files)
