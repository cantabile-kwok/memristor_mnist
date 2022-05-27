import memtorch
import torch
from memtorch.utils import LoadMNIST
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import copy
from memtorch.mn.Module import patch_model
import torch.nn as nn
from memtorch.map.Input import naive_scale
import time
import matplotlib.pyplot as plt
from memtorch.map.Parameter import naive_map

from main import Net
#from models import XzrCNN
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
class XzrCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# torch.seed(1234)
torch.manual_seed(1234)

reference_memristor = memtorch.bh.memristor.VTEAM
# reference_memristor = memtorch.bh.memristor.Stanford_PKU
# reference_memristor = memtorch.bh.memristor.LinearIonDrift
# reference_memristor = memtorch.bh.memristor.Data_Driven

reference_memristor_params = {'time_series_resolution': 1e-10}
memristor = reference_memristor(**reference_memristor_params)
# memristor.plot_hysteresis_loop()
# memristor.plot_bipolar_switching_behaviour()

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
#device = torch.device('cpu')
#model = Net().to(device)
model = XzrCNN()
model.to(device)
#model.load_state_dict(torch.load('train_schcombine_model.pt', map_location=device), strict=False)
model.load_state_dict(torch.load('xzr_baseline.pt', map_location=device), strict=False)


s = time.time()

t = time.time()
print(f"Patched model takes {t-s} seconds")
print("Begins to tune")
s = time.time()

t = time.time()
print(f"Tune takes {t-s} seconds")
def test(model, test_loader):
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()

    return 100. * float(correct) / float(len(test_loader.dataset))
# patched_model2 = patch_model(copy.deepcopy(model),
#                              memristor_model=reference_memristor,
#                              memristor_model_params=reference_memristor_params,
#                              module_parameters_to_patch=[torch.nn.Conv2d],
#                              mapping_routine=naive_map,
#                              transistor=True,  # true -> 1T1R
#                              programming_routine=None,
#                              tile_shape=(128, 128),  # size of the crossbar
#                              max_input_voltage=0.3,
#                              scaling_routine=naive_scale,
#                              ADC_resolution=8,
#                              ADC_overflow_rate=0.,
#                              quant_method='linear')
# patched_model2.tune_()

# patched_model_ = apply_nonidealities(copy.deepcopy(patched_model),
#                                      non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
#                                      lrs_proportion=0.25,
#                                      hrs_proportion=0.10,
#                                      electroform_proportion=0)
# patched_model_.tune_()

data = torch.randn(size=(1, 1, 28, 28))
# print(patched_model(data))
# print(model(data))
# for p1, p2 in zip(list(patched_model.parameters()), list(model.parameters())):
#     print('='*20)
#     print(((p1-p2)**2).sum())

# for i in range(3):
# plt.plot(nn.Softmax(dim=-1)(model(data)).squeeze().detach().numpy(), label="model", marker='o', markersize=5,
#          linestyle='-.')
# plt.plot(nn.Softmax(dim=-1)(patched_model(data)).squeeze().detach().numpy(), label="patch1", marker='o', markersize=5,
#          linestyle='-.')
# # plt.plot(nn.Softmax()(patched_model2(data)).squeeze().detach().numpy(), label="patch2")
# plt.plot(nn.Softmax(dim=-1)(patched_model_(data)).squeeze().detach().numpy(), label="nonideality", marker='o',
#          markersize=5, linestyle='-.')
# plt.legend()
# plt.show()
print('=' * 20)

batch_size = 32
train_loader, validation_loader, test_loader = LoadMNIST(batch_size=batch_size, validation=True)

with torch.no_grad():
    print(test(model, test_loader))
    patched_model = patch_model(copy.deepcopy(model),
                                memristor_model=reference_memristor,
                                memristor_model_params=reference_memristor_params,
                                module_parameters_to_patch=[torch.nn.Linear, torch.nn.Conv2d],
                                mapping_routine=naive_map,
                                transistor=True,  # true -> 1T1R
                                programming_routine=None,
                                tile_shape=(64, 64),  # size of the crossbar
                                max_input_voltage=0.3,
                                scaling_routine=naive_scale,
                                ADC_resolution=8,
                                ADC_overflow_rate=0.,
                                quant_method='linear',
                                verbose=True)
    patched_model.tune_()
    print(test(patched_model, validation_loader))

# for img, label in test_loader:
#     plt.plot(nn.Softmax(dim=-1)(model(img)).squeeze().detach().numpy(), label="model", marker='o', markersize=5,
#              linestyle='-.')
#     plt.plot(nn.Softmax(dim=-1)(patched_model(img)).squeeze().detach().numpy(), label="patch1", marker='o',
#              markersize=5,
#              linestyle='-.')
#     # plt.plot(nn.Softmax()(patched_model2(data)).squeeze().detach().numpy(), label="patch2")
#     plt.plot(nn.Softmax(dim=-1)(patched_model_(img)).squeeze().detach().numpy(), label="nonideality", marker='o',
#              markersize=5, linestyle='-.')
#     plt.legend()
#     plt.show()
#
#     break