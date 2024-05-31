import torch

import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.kagglecatsanddogs_5340_dataset import KaggleCatsAndDogs5340Dataset
from model.embeding import PatchEmbedding
from model.encoder_decoder import Encoder, Decoder
from model.classformer import Classformer


def main():
    dataset = KaggleCatsAndDogs5340Dataset()
    dataloader = DataLoader(dataset, batch_size=2)
    classformer = Classformer()

    print(dataset.classes)
    print(dataset.img_fps)
    print(dataset.labels)

    img, label = next(iter(dataloader))
    print('img.shape', img.shape)
    print('label_shape', label.shape)
    print('label', label)

    # plt.imshow(img.squeeze().permute(1, 2, 0))
    # plt.show()

    pred = classformer(img)
    # gt = torch.Tensor()
    # gt.__deepcopy__(pred)


    print('pred', pred.shape)
    print('pred', pred)


if __name__ == '__main__':
    main()
