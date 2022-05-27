import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
class CNN_1fc(nn.Module):
    def __init__(self):
        super(CNN_1fc, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = (self.fc1(x))
        return x

def copy_weight(layer1,layer2):
    layer1.weight = layer2.weight
    layer1.bias = layer2.bias

class CNN_repeat_conv(nn.Module):
    def __init__(self,model):
        super(CNN_repeat_conv, self).__init__()
        self.conv11 = nn.Conv2d(1, 20, 5, 1)
        copy_weight(self.conv11,model.conv1)
        self.conv12 = nn.Conv2d(1, 20, 5, 1)
        copy_weight(self.conv12, model.conv1)
        self.conv13 = nn.Conv2d(1, 20, 5, 1)
        copy_weight(self.conv13, model.conv1)
        self.conv21 = nn.Conv2d(20, 50, 5, 1)
        copy_weight(self.conv21, model.conv2)
        self.conv22 = nn.Conv2d(20, 50, 5, 1)
        copy_weight(self.conv22, model.conv2)
        self.conv23 = nn.Conv2d(20, 50, 5, 1)
        copy_weight(self.conv23, model.conv2)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        copy_weight(self.fc1, model.fc1)
        self.fc2 = nn.Linear(500, 10)
        copy_weight(self.fc2, model.fc2)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))/3
        x2 = F.relu(self.conv12(x))/3
        x3 = F.relu(self.conv13(x))/3
        x = x1 + x2 + x3
        x = F.max_pool2d(x, 2, 2)
        x1 = F.relu(self.conv21(x))/3
        x2 = F.relu(self.conv22(x))/3
        x3 = F.relu(self.conv23(x))/3
        x = x1 + x2 + x3
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class CNN_repeat(nn.Module):
    def __init__(self,model):
        super(CNN_repeat, self).__init__()
        self.conv11 = nn.Conv2d(1, 20, 5, 1)
        copy_weight(self.conv11,model.conv1)
        self.conv12 = nn.Conv2d(1, 20, 5, 1)
        copy_weight(self.conv12, model.conv1)
        self.conv13 = nn.Conv2d(1, 20, 5, 1)
        copy_weight(self.conv13, model.conv1)
        self.conv21 = nn.Conv2d(20, 50, 5, 1)
        copy_weight(self.conv21, model.conv2)
        self.conv22 = nn.Conv2d(20, 50, 5, 1)
        copy_weight(self.conv22, model.conv2)
        self.conv23 = nn.Conv2d(20, 50, 5, 1)
        copy_weight(self.conv23, model.conv2)
        self.fc11 = nn.Linear(4 * 4 * 50, 500)
        copy_weight(self.fc11, model.fc1)
        self.fc12 = nn.Linear(4 * 4 * 50, 500)
        copy_weight(self.fc12, model.fc1)
        self.fc13 = nn.Linear(4 * 4 * 50, 500)
        copy_weight(self.fc13, model.fc1)
        self.fc21 = nn.Linear(500, 10)
        copy_weight(self.fc21, model.fc2)
        self.fc22 = nn.Linear(500, 10)
        copy_weight(self.fc22, model.fc2)
        self.fc23 = nn.Linear(500, 10)
        copy_weight(self.fc23, model.fc2)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))/3
        x2 = F.relu(self.conv12(x))/3
        x3 = F.relu(self.conv13(x))/3
        x = x1 + x2 + x3
        x = F.max_pool2d(x, 2, 2)
        x1 = F.relu(self.conv21(x))/3
        x2 = F.relu(self.conv22(x))/3
        x3 = F.relu(self.conv23(x))/3
        x = x1 + x2 + x3
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x1 = F.relu(self.fc11(x))/3
        x2 = F.relu(self.fc12(x)) /3
        x3 = F.relu(self.fc13(x)) /3
        x = x1 + x2 + x3
        x1 = self.fc21(x)/3
        x2 = self.fc22(x) / 3
        x3 = self.fc23(x) / 3
        return x1+x2+x3
class CNN_decision(nn.Module):
    def __init__(self,model):
        super(CNN_decision, self).__init__()
        self.conv11 = nn.Conv2d(1, 20, 5, 1)
        copy_weight(self.conv11,model.conv1)
        self.conv12 = nn.Conv2d(1, 20, 5, 1)
        copy_weight(self.conv12, model.conv1)
        self.conv13 = nn.Conv2d(1, 20, 5, 1)
        copy_weight(self.conv13, model.conv1)
        self.conv21 = nn.Conv2d(20, 50, 5, 1)
        copy_weight(self.conv21, model.conv2)
        self.conv22 = nn.Conv2d(20, 50, 5, 1)
        copy_weight(self.conv22, model.conv2)
        self.conv23 = nn.Conv2d(20, 50, 5, 1)
        copy_weight(self.conv23, model.conv2)
        self.fc11 = nn.Linear(4 * 4 * 50, 500)
        copy_weight(self.fc11, model.fc1)
        self.fc12 = nn.Linear(4 * 4 * 50, 500)
        copy_weight(self.fc12, model.fc1)
        self.fc13 = nn.Linear(4 * 4 * 50, 500)
        copy_weight(self.fc13, model.fc1)
        self.fc21 = nn.Linear(500, 10)
        copy_weight(self.fc21, model.fc2)
        self.fc22 = nn.Linear(500, 10)
        copy_weight(self.fc22, model.fc2)
        self.fc23 = nn.Linear(500, 10)
        copy_weight(self.fc23, model.fc2)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x2 = F.relu(self.conv12(x))
        x3 = F.relu(self.conv13(x))

        x1 = F.max_pool2d(x1, 2, 2)
        x2 = F.max_pool2d(x2, 2, 2)
        x3 = F.max_pool2d(x3, 2, 2)
        
        x1 = F.relu(self.conv21(x1))
        x2 = F.relu(self.conv22(x2))
        x3 = F.relu(self.conv23(x3))

        x1 = F.max_pool2d(x1, 2, 2)
        x2 = F.max_pool2d(x2, 2, 2)
        x3= F.max_pool2d(x3, 2, 2)

        x1 = x1.view(-1, 4 * 4 * 50)
        x2 = x2.view(-1, 4 * 4 * 50)
        x3 = x3.view(-1, 4 * 4 * 50)
        x1 = F.relu(self.fc11(x1))
        x2 = F.relu(self.fc12(x2))
        x3 = F.relu(self.fc13(x3))


        x1 = self.fc21(x1)
        x2 = self.fc22(x2)
        x3 = self.fc23(x3)
        return x1+x2+x3