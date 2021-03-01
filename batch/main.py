import torch
import torch.utils.data as Data

BATCH_SIZE = 3

x = torch.linspace(0,1,10)
y = torch.normal(x)

print(x.size(), y.size())

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch: ', epoch, '\tStep: ', step, '\tx: ', batch_x.data.numpy(), '\ty: ', batch_y.data.numpy())