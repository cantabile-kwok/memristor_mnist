import torch
from torch.autograd import Variable
# import memtorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from memtorch.utils import LoadMNIST
import numpy as np
import argparse
from loader import LoadMNIST
import os

# device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def test(model, test_loader):
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()

    return 100. * float(correct) / float(len(test_loader.dataset))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-b', '--bsz', default=16, type=int)
    args.add_argument('-l', '--lr', default=1e-3, type=float)
    args.add_argument("-m", "--model", choices=['MLP', "CNN"], default="CNN", type=str)
    args = args.parse_args()

    output_dir = f"exp/[b]{args.bsz}-[lr]{args.lr}-[m]{args.model}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = open(f"{output_dir}/train.log", 'w')

    epochs = 10
    learning_rate = args.lr
    batch_size = args.bsz

    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=batch_size, validation=False)

    if args.model == 'CNN':
        model = Net().to(device)
    elif args.model == 'MLP':
        model = MLP().to(device)
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0
    for epoch in range(0, epochs):
        print('Epoch: [%d]\t\t' % (epoch + 1), file=logger, flush=True)

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            print(f"Epoch: {epoch}, batch: {batch_idx}, loss: {loss.item()}", file=logger, flush=True)
            loss.backward()
            optimizer.step()

        accuracy = test(model, test_loader)
        print(f"Epoch: {epoch}, test accuracy: {accuracy}", file=logger, flush=True)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), f'{output_dir}/model.pt')
            best_accuracy = accuracy

    logger.close()
