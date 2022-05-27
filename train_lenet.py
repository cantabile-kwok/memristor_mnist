import torch
from torch.autograd import Variable
# import memtorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from memtorch.utils import LoadMNIST
import numpy as np
from tqdm import tqdm
import argparse
from loader import LoadMNIST
from main import test

import os


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=128, validation=False)

    epochs = 100
    model = LeNet5()
    model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    logger = open('lenet_train.log', 'w')
    best_accuracy = 0

    for epoch in tqdm(range(0, epochs)):

        print('Epoch: [%d]\t\t' % (epoch + 1), file=logger, flush=True)

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            # print(data.shape)
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            print(f"Epoch: {epoch}, batch: {batch_idx}, loss: {loss.item()}", file=logger, flush=True)
            loss.backward()
            optimizer.step()

        accuracy = test(model, test_loader)
        print(f"Epoch: {epoch}, test accuracy: {accuracy}", file=logger, flush=True)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), f'lenet.pt')
            best_accuracy = accuracy
