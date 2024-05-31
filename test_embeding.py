import torch

import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.kagglecatsanddogs_5340_dataset import KaggleCatsAndDogs5340Dataset
from model.embeding import PatchEmbedding
from model.encoder_decoder import Encoder, Decoder, Classifier


def main():
    dataset = KaggleCatsAndDogs5340Dataset()
    print(dataset.classes)
    print(dataset.img_fps)
    print(dataset.labels)

    dataloader = DataLoader(dataset)
    img, label = next(iter(dataloader))
    print('img.shape', img.shape)
    print(label)

    # plt.imshow(img.squeeze().permute(1, 2, 0))
    # plt.show()
    c_start = 512
    embedding = PatchEmbedding()
    encoder0 = Encoder(c_in=embedding.patch_size**2*3, c_out=c_start)
    encoder1 = Encoder(c_in=c_start, c_out=c_start*2)
    encoder2 = Encoder(c_in=c_start*2, c_out=c_start*4)
    encoder3 = Encoder(c_in=c_start*4, c_out=c_start*8)

    decoder0 = Decoder(c_in=c_start*8, c_out=c_start*4)

    classifier = Classifier(c_in=c_start*4, c_out=2)

    x0 = embedding(img)
    print('x0', x0.shape)

    x1 = encoder0(x0)
    x2 = encoder1(x1)
    x3 = encoder2(x2)
    x4 = encoder3(x3)
    print('x4', x4.shape)

    x5 = decoder0(x4)
    print('x5', x5.shape)

    y = classifier(x5)
    print('y', y.shape)


if __name__ == '__main__':
    main()
