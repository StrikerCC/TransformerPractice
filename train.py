import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.kagglecatsanddogs_5340_dataset import KaggleCatsAndDogs5340Dataset
from model.classformer import Classformer


def main():
    dataset = KaggleCatsAndDogs5340Dataset()
    dataloader = DataLoader(dataset, batch_size=1)
    classformer = Classformer()
    classformer.train(True)

    print(dataset.classes)
    print(dataset.img_fps)
    print(dataset.labels)

    num_epoch = 100
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classformer.parameters(),
                                lr=0.0001,
                                momentum=0.9)

    for i_epoch in range(num_epoch):
        print('i_epoch', i_epoch, ' start')
        train_iter = iter(dataloader)
        running_loss = 0

        for i_data in range(len(dataloader)):
            img, label = next(train_iter)
            optimizer.zero_grad()
            output = classformer(img)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss
        print('i_epoch', i_epoch, ' done')
        print('running_loss', running_loss)

    torch.save(classformer.state_dict(), './log/classformer.pth')


if __name__ == '__main__':
    main()
