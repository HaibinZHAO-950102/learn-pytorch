import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable


def checker(A, B):
    if B == True:
        print(A)
    else:
        pass

check = False;

n_data = torch.ones(100, 2)
checker(n_data.size(), check)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
checker(x0, check)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x2 = torch.normal(2 * n_data, 1)
x2[:,0] = x2[:,0] * -1
y2 = torch.ones(100) * 2

x = torch.cat((x0, x1, x2), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1, y2), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, nfeature, nhidden, nout):
        super(Net, self).__init__()
        self.ein = torch.nn.Linear(nfeature, nhidden)
        self.hidden = torch.nn.Linear(nhidden, nhidden)
        self.predict = torch.nn.Linear(nhidden, nout)

    def forward(self, x):
        x = F.relu(self.ein(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(2, 20, 3)
checker(net, check)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_function = torch.nn.CrossEntropyLoss()

for i in range(100):
    out = net(x)

    loss = loss_function(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=pred_y)
        accuracy = sum(pred_y == target_y) / 300
        plt.text(1.5, -2, 'Accuracy %.2f' % accuracy)
        n = i + 2
        plt.text(1.5, -3, 'Iteration %d / 100' % n)
        plt.pause(0.1)
        # print(prediction)

plt.ioff()
plt.show()

torch.save(net, 'nn.pkl')
net2 = torch.load('nn.pkl')

out2 = net2(x)
prediction2 = torch.max(F.softmax(out2), 1)[1]
pred_y2 = prediction2.data.numpy()
target_y = y.data.numpy()
plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=pred_y2)
plt.show()