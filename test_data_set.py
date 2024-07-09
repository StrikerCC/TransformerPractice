import numpy as np
import torch
import shutil

from dataset.kagglecatsanddogs_5340_dataset import KaggleCatsAndDogs5340Dataset
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def main():
    dataset = KaggleCatsAndDogs5340Dataset()
    # print(dataset.classes)
    # print(dataset.img_fps)
    # print(dataset.labels)

    # cv2.imshow('', dataset[0])
    # cv2.waitKey(0)
    # cv2.imshow('', dataset[1])
    # cv2.waitKey(0)

    dataloader = DataLoader(dataset)
    print('dataloader: ', len(dataloader))

    data_iter = iter(dataloader)
    for i in range(len(dataloader)):

        # try:
        #     img, label = next(data_iter)
        #     img = np.asarray(img, dtype=np.uint8)
        #     img = img.squeeze(0).transpose(1, 2, 0)
        #     # print(img.shape)
        #     # print(label.shape)
        #     # print(label)
        #
        #     # plt.imshow(img)
        #     # plt.show()
        # except:
        #     print(i)
        #     print(dataset.img_fps[i])

        img_fp = dataset.img_fps[i]
        img = cv2.imread(img_fp)
        # print(img.shape)
        # print(label.shape)
        # print(label)

        if img is None:
            print(i)
            src = dataset.img_fps[i]
            tgt = './data/kagglecatsanddogs_5340/badPetImages/' + dataset.img_fns[i]
            print('src', src)
            print('tgt', tgt)
            shutil.move(src, tgt)


if __name__ == '__main__':
    main()

