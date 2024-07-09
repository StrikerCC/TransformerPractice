import os
import cv2
import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision.io import read_image


class KaggleCatsAndDogs5340Dataset(Dataset):
    def __init__(self, dp='./data/kagglecatsanddogs_5340/'):
        super().__init__()
        self.img_fps = []
        self.img_fns = []
        self.labels = []
        self.classes = []

        self.tf = torchvision.transforms.Resize(size=(300, 500))
        self.load_from_dir(dp)
        return

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):
        img = read_image(self.img_fps[idx]).to(torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # label = torch.zeros(len(self.classes), dtype=torch.float32)
        # label[self.labels[idx]] = 1
        return img, label

    def load_from_dir(self, dp='./data/kagglecatsanddogs_5340/'):
        root_dp = os.path.join(dp, 'PetImages')

        for ins_dn in os.listdir(root_dp):
            self.classes.append(ins_dn)
            imgs_dp = os.path.join(root_dp, ins_dn)
            if not os.path.isdir(imgs_dp): continue
            for img_fn in os.listdir(imgs_dp):

                if img_fn.find('.jpg') == -1:
                    print(img_fn)

                self.img_fps.append(os.path.join(imgs_dp, img_fn))
                self.img_fns.append(ins_dn+'/'+img_fn)
                self.labels.append(len(self.classes) - 1)
