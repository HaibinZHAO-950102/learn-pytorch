# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar('y=x^2',i**2, i)

dir = '/Users/haibinzhao/Desktop/TECO/Software/Pytorch/learn0/train/cat.0.jpg'
img_PIL = Image.open(dir)
img = np.array(img_PIL)
writer.add_image('test', img, 1, dataformats='HWC')

writer.close()

