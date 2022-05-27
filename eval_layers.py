from main import test
from loader import LoadMNIST
import os
import torch
from eval_noise import add_noise_to_weights
from train_mlp import VariableMLP
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=128, validation=False, num_workers=0)

    res = []
    for n in tqdm(range(1, 6)):
        exp_dir = f"{n}layers"

        for r in range(25):
            model = VariableMLP(n)
            model.to(device)
            model.load_state_dict(torch.load(f"exp_mlp/{exp_dir}/model.pt", map_location=device))
            add_noise_to_weights(model, 0, 1.2)
            acc = test(model, test_loader=test_loader)
            res.append({
                "layers": n,
                "acc": acc
            })
    res = pd.DataFrame(res)
    res.to_csv('layers_acc.csv')
