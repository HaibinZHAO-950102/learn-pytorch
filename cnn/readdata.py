from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

class MyData(Dataset):
    def __init__(self, root_dir, type=None, transformer=None):
        super(MyData, self).__init__()

        type_indicator = False;

        self.type = type
        self.transformer = transformer

        if self.type == 'train':
            self.path = root_dir + '/train'
            type_indicator = True
        elif self.type == 'test':
            self.path = root_dir + '/test'
            type_indicator = True
        else:
            print('''The second parameter of MyData() should be 'train' or 'test'.''')

        if type_indicator:
            filenames = os.listdir(self.path)
            self.imgnames = []
            self.labels = []
            for filename in filenames:
                if filename[-3:] == 'png':
                    self.imgnames.append(filename)
                    self.labels.append(filename[0])
                else:
                    pass
        else:
            pass

    def __getitem__(self, item):
        img_path = self.path + '/' + self.imgnames[item]
        img = plt.imread(img_path)
        img = self.transformer(np.array(img))

        label = self.labels[item]
        if label == '0':
            label = 0
        else:
            label = 1

        return img, label

    def __len__(self):
        return len(self.imgnames)

