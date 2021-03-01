import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import readdata
import os

EPOCH = 1000
LR = 0.0025
DOWNLOAD_MNIST = False
PATH = '/Users/haibinzhao/Desktop/TECO/Software/Pytorch/cnn/dataset'

if os.path.exists('loss.txt'):
    os.remove('loss.txt')
if os.path.exists('acc.txt'):
    os.remove('acc.txt')

imgtotensor = transforms.ToTensor()
imgresize = transforms.Resize([64, 64])
transformer = transforms.Compose([imgtotensor, imgresize])

train_data = readdata.MyData(PATH, type='train', transformer=transformer)
test_data = readdata.MyData(PATH, type='test', transformer=transformer)

train_loader = Data.DataLoader(dataset=train_data, batch_size=6, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=30, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,          # 图片的维度，黑白1，RGB3
                out_channels=10,        # 输出特征
                kernel_size=5,
                stride=1,               # 卷积核每次移动的像素
                padding=2,              # 扩边
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 30, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.out1 = nn.Sequential(
            nn.Linear(30 * 8 * 8, 30),
            nn.ReLU()
        )

        self.out2 = nn.Sequential(
            nn.Linear(30, 30),
            nn.ReLU()
        )

        self.out3 = nn.Sequential(
            nn.Linear(30, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.out1(x)
        x = self.out2(x)
        output = self.out3(x)
        return output

cnn = CNN()
print(cnn)
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()




for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)

        loss = loss_func(output, b_y)
        file_loss = open('loss.txt', 'a')
        file_loss.write(str(loss.detach().numpy()))
        file_loss.write('\n')
        file_loss.close()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for i, (test_x, test_y) in enumerate(test_loader):
        b_x_test = Variable(test_x)
        b_y_test = Variable(test_y)

        output_test = cnn(b_x_test)
        prediction_y = torch.max(output_test, 1)[1].data.squeeze()
        # print('Result: \t\t' , prediction_y.numpy())
        # print('Ground Truth: \t' , b_y_test.numpy())
        accuracy = sum(prediction_y == b_y_test) / b_y_test.size(0)
        print('Epoch: ', epoch, '| Accuracy: ', accuracy.numpy())
        file_acc = open('acc.txt', 'a')
        file_acc.write(str(accuracy.numpy()))
        file_acc.write('\n')
        file_acc.close()

def draw_them():
    file_loss = 'loss.txt'
    file_acc = 'acc.txt'
    loss_record = []
    accuracy_record = []
    with open(file_loss, 'r') as f:
        lines = f.readlines()
        for line in lines:
            loss_record.append(float(line))
    with open(file_acc, 'r') as f:
        lines = f.readlines()
        for line in lines:
            accuracy_record.append(float(line))

    plt.subplot(1,2,1)
    plt.plot(loss_record)
    plt.title('Loss', fontsize='large')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    plt.subplot(1,2,2)
    plt.plot(accuracy_record)
    plt.title('Accuracy', fontsize='large')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.show()

draw_them()