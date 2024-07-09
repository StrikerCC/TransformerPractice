import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.kagglecatsanddogs_5340_dataset import KaggleCatsAndDogs5340Dataset
from model.classformer import Classformer

import cv2


def main():
    dataset = KaggleCatsAndDogs5340Dataset()
    dataloader = DataLoader(dataset, batch_size=1)
    classformer = Classformer()
    classformer.load_state_dict(torch.load('./log/classformer.pth'))

    print(dataset.classes)
    print(dataset.img_fps)
    print(dataset.labels)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classformer.parameters(),
                                lr=0.00001,
                                momentum=0.9)
    test_iter = iter(dataloader)
    num_right = 0
    for i_data in range(len(dataloader)):
        input, label = next(test_iter)
        optimizer.zero_grad()
        output = classformer(input)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        img = np.asarray(input[0]).transpose(1, 2, 0)
        img = img[:, :, ::-1]
        id_classified = int(torch.argmax(output, dim=1))
        fp = './log/' + str(i_data) + '_' + str(dataset.classes[id_classified]) + '.png'

        print(i_data, 'model out', id_classified, 'label is', int(label))
        if id_classified == int(label): num_right += 1

    print(num_right/len(dataloader))
        # print(fp)
        # cv2.imwrite(fp, img)


if __name__ == '__main__':
    main()
