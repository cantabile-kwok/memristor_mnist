import torch
import torch.nn as nn
import numpy as np
import memtorch
from memtorch.utils import LoadMNIST
import os
import copy
from memtorch.mn.Module import patch_model
from memtorch.map.Input import naive_scale
import matplotlib.pyplot as plt
from tqdm import tqdm
from memtorch.map.Parameter import naive_map
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
from models import CNN
import torch.optim as optim
import logging

import argparse

torch.manual_seed(0)
np.random.seed(0)
def test_acc(model, loader):
    correct = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(loader):
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()
    return 100. * float(correct) / float(len(loader.dataset))

def test_noise_model(model,loader,std):
    acc = 0
    for i in range(10):
        patched_model = add_noise_to_weights(model, device, std)
        acc += test_acc(patched_model,loader)
    return acc/10.0
def mem_from_cnn(cnn_model):
    reference_memristor = memtorch.bh.memristor.VTEAM
    reference_memristor_params = {'time_series_resolution': 1e-10}
    # memristor = reference_memristor(**reference_memristor_params)
    patched_model = patch_model(copy.deepcopy(cnn_model),
                                memristor_model=reference_memristor,
                                memristor_model_params=reference_memristor_params,
                                module_parameters_to_patch=[torch.nn.Conv2d],
                                mapping_routine=naive_map,
                                transistor=True,  # true -> 1T1R
                                programming_routine=None,
                                tile_shape=(128, 128),  # size of the crossbar
                                max_input_voltage=0.3,
                                scaling_routine=naive_scale,
                                ADC_resolution=8,
                                ADC_overflow_rate=0.,
                                quant_method='linear')
    patched_model.tune_()
    return patched_model
def add_noise_to_weights( model,device,std=0.5,mean=0):
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
    """
    model = copy.deepcopy(model)
    gassian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_((torch.exp(gassian_kernel.sample(param.size())).to(device)))
    return model


class LinearScheduledSampler:
    def __init__(self, linear_range, start_value, stop_value):
        self.cnt = 0
        self.range = linear_range
        self.start = start_value
        self.stop = stop_value

    def get_prob(self):
        dist_from_start = self.cnt % self.range
        return self.stop + (self.range - dist_from_start) * (self.start - self.stop) / self.range

    def update(self):
        self.cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--bsz", default=256, type=int, help='batch size')
    parser.add_argument("-e", "--epoch", default=10, type=int, help='epoch num')
    parser.add_argument('-l', '--lr', default=0.1, type=float, help='learning rate initial value')
    parser.add_argument('--lr-step', default=2, type=int, help='every this number of epochs, multiply lr by sth')
    parser.add_argument('--lr-coef', default=0.1, type=float, help='every time to multiply lr by when reaching lr_step')
    parser.add_argument('--mem-steps', default=234, type=int, help='interval of steps when we update memristor')
    parser.add_argument('--sch',default=True,type=bool,help = 'use schedule sampling')
    parser.add_argument('--save', default='train_noise2_std1_sch03', type=str, help='save_path')
    parser.add_argument('--std', default=1, type=float, help='gauss std')
    args = parser.parse_args()
    logging.basicConfig(filename=args.save+'.log', level=logging.DEBUG)
    batch_size = args.bsz
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=batch_size, validation=False, num_workers=0)
    cnn_model = CNN().to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr)
    best_accuracy = 0
    logging.info(args)
    learning_rate = args.lr

    update_step_cnt = 0

    # ============= memristor related ===========
    #patched_model = mem_from_cnn(cnn_model)
    patched_model = add_noise_to_weights(cnn_model,device,args.std)
    # ========================================

    # ================ Train =================
    scheduled_sampler = LinearScheduledSampler(args.mem_steps, start_value=0.99, stop_value=.3)
    alpha = 0.99
   
    for epoch in range(args.epoch):
        print(args.epoch)
        print('Epoch: [%d]\t\t' % (epoch + 1), end='')
        if epoch % args.lr_step == 0:
            learning_rate = learning_rate * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        cnn_model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            # if (update_step_cnt % args.mem_steps == 0) and (update_step_cnt != 0):  # NOTE: don't update at 0-step
            #     patched_model = mem_from_cnn(cnn_model)
            # if args.sch:
            #     alpha = scheduled_sampler.get_prob()

            patched_model = add_noise_to_weights(cnn_model,device,1-alpha)

            # print(alpha)

            optimizer.zero_grad()
            cnn_logits = cnn_model(data.to(device))
            cnn_preds = nn.Softmax(dim=-1)(cnn_logits)

            mem_logits = patched_model(data.to(device))
            # mem_logits = cnn_model(data.to(device))
            mem_preds = nn.Softmax(dim=-1)(mem_logits)
            interpolated_preds = 0.5 * cnn_preds + 0.5 * mem_preds

            # ============ make fake loss input =================
            fake_loss_input = interpolated_preds.detach() + cnn_preds - cnn_preds.detach()
            # ===================================================

            loss = nn.NLLLoss()(torch.log(fake_loss_input), target.to(device))  # NOTE: take log at first

            # loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            update_step_cnt += 1
            scheduled_sampler.update()
            pbar.set_postfix(alpha=alpha, batch_loss=loss.item())
            
        
        patched_model = add_noise_to_weights(cnn_model,device,args.std)
        accuracy = test_noise_model(cnn_model, test_loader,args.std)
        accuracy2 = test_acc(cnn_model, test_loader)
        logging.info('acc1 %2.2f%%' % accuracy)
        logging.info('acc2 %2.2f%%' % accuracy2)
        print('%2.2f%%' % accuracy)
        print('%2.2f%%' % accuracy2)
        alpha -= 0.08
        if accuracy > best_accuracy:
            torch.save(cnn_model.state_dict(), args.save+'.pt')
            best_accuracy = accuracy
