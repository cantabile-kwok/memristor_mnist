import os.path

import torch
import torch.nn as nn
from main import test
import json
from loader import LoadMNIST
import torch.optim as optim


class VariableMLP(nn.Module):
    def __init__(self, n):
        super(VariableMLP, self).__init__()
        # self.layers = nn.ModuleList()
        # input: 28*28
        if n == 1:
            self.layers = nn.Linear(28 * 28, 10)
        elif n == 2:
            self.layers = nn.Sequential(
                nn.Linear(28 * 28, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
        elif n == 3:
            self.layers = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
        elif n == 4:
            self.layers = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        elif n == 5:
            self.layers = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = nn.Flatten()(x.squeeze())
        logits = self.layers(x)
        return logits


if __name__ == '__main__':
    bsz = 128
    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=bsz, validation=False, num_workers=0)
    lr = 0.001
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    n2acc = {}
    for n in range(1, 6):
        print(f"======================== {n} layers ================")
        exp_dir = f'exp_mlp/{n}layers'
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        logger = open(f"{exp_dir}/train.log", 'w')

        model = VariableMLP(n)
        model.to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)
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
            print(f"Epoch: {epoch}, test accuracy: {accuracy}")

            if accuracy > best_accuracy:
                torch.save(model.state_dict(), f'{exp_dir}/model.pt')
                best_accuracy = accuracy
        n2acc[n] = best_accuracy
    with open('n2acc.json', 'w') as f:
        json.dump(n2acc, f)
