from train_lenet import LeNet5
import torch
import torch.nn as nn
from eval_noise import add_noise_to_weights
from loader import LoadMNIST
from main import test
from tqdm import tqdm
import pandas as pd

train_loader, validation_loader, test_loader = LoadMNIST(batch_size=128, validation=False)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

lenet = LeNet5()
lenet.load_state_dict(torch.load("lenet.pt", map_location=device))
print(test(lenet, test_loader))
exit(0)


res = []
for module in tqdm(['conv1', 'conv2', 'fc1', 'fc2', 'fc3']):
    for r in range(20):
        lenet = LeNet5()
        lenet.load_state_dict(torch.load("lenet.pt", map_location=device))
        add_noise_to_weights(getattr(lenet, module), 0, 1.2)
        # print(test(lenet, test_loader))
        acc = test(lenet, test_loader)
        res.append({
            'module': module,
            'acc': acc
        })
res = pd.DataFrame(res)
res.to_csv("lenet_layers.csv")
