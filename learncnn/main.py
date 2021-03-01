import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(0, 2, 100), dim=1)
y = torch.sin(3.14 * x) + 0.1 * torch.rand(x.size())

print(torch.linspace(0, 2, 100).size())
print(x.size())
x, y = Variable(x), Variable(y)


# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, nfeature, nhidden, noutput):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(nfeature, nhidden)
        self.hidden2 = torch.nn.Linear(nhidden, nhidden)
        self.hidden3 = torch.nn.Linear(nhidden, nhidden)
        self.hidden4 = torch.nn.Linear(nhidden, nhidden)
        self.hidden5 = torch.nn.Linear(nhidden, nhidden)
        self.hidden6 = torch.nn.Linear(nhidden, nhidden)
        self.hidden7 = torch.nn.Linear(nhidden, nhidden)
        self.hidden8 = torch.nn.Linear(nhidden, nhidden)
        self.predict = torch.nn.Linear(nhidden, noutput)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = self.predict(x)
        return x


net = Net(1, 30, 1)
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, )
loss_func = torch.nn.MSELoss()

for t in range(10000):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.ylim(-1.2,1.2)
        plt.text(0.5,0,t+10)
        plt.pause(0.1)

plt.ioff()
plt.show()
