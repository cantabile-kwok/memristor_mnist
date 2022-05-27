import pandas as pd
import torch
import torch.nn as nn
from main import Net, MLP
import os
import re

from main import test
import json
from loader import LoadMNIST
from tqdm import tqdm

from eval_noise import add_noise_to_weights

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

res = []
clean_acc = {}
for exp_dir in tqdm(os.listdir('exp')):
    bsz = re.findall(r"\[b\]\d+", exp_dir)[0].strip('[b]')
    bsz = int(bsz)
    lr = re.findall(r'\[lr\]0\.\d+', exp_dir)[0].strip('[lr]')
    lr = float(lr)
    m = re.findall(r"\[m\].*$", exp_dir)[0].strip('[m]')

    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=bsz, validation=False)

    if m == 'MLP':
        model = MLP()
    elif m == 'CNN':
        model = Net()
    else:
        raise NotImplementedError

    model.load_state_dict(torch.load(f"exp/{exp_dir}/model.pt", map_location=device))
    model.to(device)
    acc_without_noise = test(model, test_loader=test_loader)
    print(f"[b] {bsz}\t [lr]{lr} \t [m] {m}: \t acc: {acc_without_noise}")
    clean_acc[exp_dir] = acc_without_noise

    # for r in range(10):
    #     noise_model = MLP() if m == 'MLP' else Net()
    #     noise_model.to(device)
    #     noise_model.load_state_dict(model.state_dict())
    #     add_noise_to_weights(noise_model, 0, 1.2)
    #     # noise_model
    #     acc = test(noise_model, test_loader)
    #     res.append({
    #         "bsz": bsz,
    #         "lr": lr,
    #         "model": m,
    #         "acc": acc
    #     })

# res = pd.DataFrame(res)
# res.to_csv('bsz_lr_model.csv')

with open("bsz_lr_model_clean.json", 'w') as f:
    json.dump(clean_acc, f)