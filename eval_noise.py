from main import Net
import torch
import torch.nn as nn
from main import test
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
import seaborn as sns

# from memtorch.utils import LoadMNIST
from loader import LoadMNIST

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def add_noise_to_weights(mdl, mean, std):
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
    """
    # device = 'cpu'
    gassian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        for param in mdl.parameters():
            param.mul_(torch.exp(gassian_kernel.sample(param.size()).to(device)))

if __name__ == '__main__':

    model = Net()
    model.load_state_dict(torch.load("cnn_baseline.pt", map_location=device))

    bsz = 16
    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=bsz, validation=False)
    print("Original model acc: ", test(model, test_loader))

    res = []

    for std in tqdm([0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 1., 1.2, 1.5]):
        for r in range(10):
            proxy_model = Net()
            proxy_model.load_state_dict(model.state_dict())

            add_noise_to_weights(proxy_model, 0, std)
            print(f"Run {r}")
            a = test(proxy_model, test_loader)
            res.append({
                "std": std,
                "acc": a,
            })
            # print(test(proxy_model, test_loader))
    res = pd.DataFrame(res)
    res.to_csv("std_vs_acc.csv")
