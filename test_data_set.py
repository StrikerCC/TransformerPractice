import torch

from dataset.kagglecatsanddogs_5340_dataset import KaggleCatsAndDogs5340Dataset
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def main():
    dataset = KaggleCatsAndDogs5340Dataset()
    print(dataset.classes)
    print(dataset.img_fps)
    print(dataset.labels)

    # cv2.imshow('', dataset[0])
    # cv2.waitKey(0)
    # cv2.imshow('', dataset[1])
    # cv2.waitKey(0)

    dataloader = DataLoader(dataset)
    img, label = next(iter(dataloader))
    print(img.shape)
    print(label)

    plt.imshow(img.squeeze().permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    main()
