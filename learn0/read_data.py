from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_path = os.listdir(self.root_dir)

    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        label = img_name.split('.')[0]
        return img, label

    def __len__(self):
        return len(self.img_path)

