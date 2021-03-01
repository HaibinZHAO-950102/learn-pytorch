import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

plt.subplot(2,2,1)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.legend(loc='best')

plt.subplot(2,2,2)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.legend(loc='best')

plt.subplot(2,2,3)
plt.plot(x_np, y_tanh, c='red', label='tahn')
plt.legend(loc='best')

plt.subplot(2,2,4)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.legend(loc='best')

plt.show()