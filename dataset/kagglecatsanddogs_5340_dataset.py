import os


class KaggleCatsAndDogs5340Dataset():
    def __init__(self, dp='./data/kagglecatsanddogs_5340/'):
        self.img_fps = []
        self.anno = []
        self.classes = []
        return

    def __len__(self):
        return

    def __getitem__(self, item):
        return

    def load_from_dir(self, dp='./data/kagglecatsanddogs_5340/'):
        img_dps = os.path.join(dp, 'PetImages')

        for ins_dn in os.listdir(img_dps):
            self.classes.append(ins_dn)
            imgs_dp = os.path.join(img_dps, ins_dn)
            for img_fp in os.listdir(imgs_dp):
                self.img_fps.append(img_fp)
                self.anno.append(len(self.classes))
        return


def main():
    dataset = KaggleCatsAndDogs5340Dataset()


if __name__ == '__main__':
    main()
