import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet300100(nn.Module):
    """
    Fully-connected LeNet architecture for MNIST
    784 -> 300 -> 100 -> 10
    """
    def __init__(self):
        super(LeNet300100, self).__init__()
        
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def count_parameters(self):
        total = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                total += module.weight.numel()   # ignore biases
        return total

    def count_nonzero_parameters(self):
        nz = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, "weight_mask"):
                    nz += int(module.weight_mask.sum().item())
                else:
                    nz += int((module.weight != 0).sum().item())
        return nz

    def get_sparsity(self):
        total = self.count_parameters()
        nonzero = self.count_nonzero_parameters()
        return 100.0 * (total - nonzero) / total
