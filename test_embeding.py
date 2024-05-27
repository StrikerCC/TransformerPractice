import torch

import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.kagglecatsanddogs_5340_dataset import KaggleCatsAndDogs5340Dataset
from model.embeding import PatchEmbedding
from model.encoder_decoder import Encoder


def main():
    dataset = KaggleCatsAndDogs5340Dataset()
    print(dataset.classes)
    print(dataset.img_fps)
    print(dataset.labels)

    dataloader = DataLoader(dataset)
    img, label = next(iter(dataloader))
    print(img.shape)
    print(label)

    # plt.imshow(img.squeeze().permute(1, 2, 0))
    # plt.show()

    embedding = PatchEmbedding()
    encoder0 = Encoder(c_in=embedding.patch_size**2*3)
    encoder1 = Encoder()
    encoder2 = Encoder()
    encoder3 = Encoder()

    x0 = embedding(img)
    print('x0', x0.shape)

    x1 = encoder0(x0)
    x2 = encoder1(x1)
    x3 = encoder2(x2)
    x4 = encoder3(x3)
    print('x4', x4.shape)


if __name__ == '__main__':
    main()
