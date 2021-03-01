import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.01

t = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(t)
y_np = np.cos(t)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):

        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)

        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None

for step in range(1):
    start, end = step * np.pi, (step + 1) * np.pi

    t_train = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_train = np.sin(t_train)
    y_train = np.cos(t_train)

    x = Variable(torch.from_numpy(x_train[np.newaxis, :, np.newaxis])) # (batch_size=1, time_step, input_size)
    y = Variable(torch.from_numpy(y_train[np.newaxis, :, np.newaxis]))

    prediction, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(t_train, y_train, 'b-')
    plt.plot(t_train, prediction.detach().view(-1).numpy(), 'r-')

plt.show()

