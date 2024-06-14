from config import DATASET_PATH

import cv2
import h5py
import numpy as np

from torch import from_numpy, float32
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

with h5py.File(DATASET_PATH, "r") as file:
    good_hyper_images = file["good_hyper_images"][:]
    bad_hyper_images = file["bad_hyper_images"][:]
    good_labels = file["good_labels"][:]
    bad_labels = file["bad_labels"][:]


class Dataset:
    def __init__(self, input: np.ndarray, labels: np.ndarray) -> None:
        self.input = from_numpy(input).type(float32)
        self.labels = from_numpy(labels)
        self.len = len(labels)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index):
        return self.input[index], self.labels[index]


dset_train = Dataset(
    np.vstack((good_hyper_images[:8], bad_hyper_images[:7])),
    np.hstack((good_labels[:8], bad_labels[:7])),
)
dset_test = Dataset(
    np.vstack((good_hyper_images[8:], bad_hyper_images[7:])),
    np.hstack((good_labels[8:], bad_labels[7:])),
)

train_loader = DataLoader(dset_train, batch_size=1, shuffle=True)
test_loader = DataLoader(dset_test, batch_size=1, shuffle=False)
