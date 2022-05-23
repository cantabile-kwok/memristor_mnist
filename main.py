import torch
from torch.autograd import Variable
import memtorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memtorch.utils import LoadMNIST
import numpy as np

device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test(model, test_loader):
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()

    return 100. * float(correct) / float(len(test_loader.dataset))


if __name__ == '__main__':
    epochs = 10
    learning_rate = 1e-1
    step_lr = 5
    batch_size = 256
    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=batch_size, validation=False)
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0
    for epoch in range(0, epochs):
        print('Epoch: [%d]\t\t' % (epoch + 1), end='')
        if epoch % step_lr == 0:
            learning_rate = learning_rate * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()

        accuracy = test(model, test_loader)
        print('%2.2f%%' % accuracy)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'trained_model.pt')
            best_accuracy = accuracy