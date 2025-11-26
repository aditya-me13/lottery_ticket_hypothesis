import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    Convolutional LeNet-5 architecture for MNIST
    Original LeCun et al. 1998 architecture
    
    Architecture:
    - Conv1: 1x28x28 -> 6x24x24 (5x5 kernel)
    - Pool1: 6x24x24 -> 6x12x12 (2x2 maxpool)
    - Conv2: 6x12x12 -> 16x8x8 (5x5 kernel)
    - Pool2: 16x8x8 -> 16x4x4 (2x2 maxpool)
    - FC1: 256 -> 120
    - FC2: 120 -> 84
    - FC3: 84 -> 10
    
    Total parameters: ~61K
    """
    
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16 channels * 4x4 after pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # Initialize weights
        self._initialize_weights()
    
    def _init_weights(self):
        # kaiming init for relu nets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Conv1 + ReLU + MaxPool
        x = F.relu(self.conv1(x))      # (batch, 6, 24, 24)
        x = F.max_pool2d(x, 2)         # (batch, 6, 12, 12)
        
        # Conv2 + ReLU + MaxPool
        x = F.relu(self.conv2(x))      # (batch, 16, 8, 8)
        x = F.max_pool2d(x, 2)         # (batch, 16, 4, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)      # (batch, 256)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))        # (batch, 120)
        x = F.relu(self.fc2(x))        # (batch, 84)
        x = self.fc3(x)                # (batch, 10)
        
        return x
    
    def count_parameters(self):
        # total prunable weights (biases ignored)
        total = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                total += m.weight.numel()
        return total
    
    def count_nonzero_parameters(self):
        # how many weights are still non-zero
        nz = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if hasattr(m, "weight_mask"):
                    nz += int(m.weight_mask.sum().item())     # mask-based pruning
                else:
                    nz += int((m.weight != 0).sum().item())   # actual zeros
        return nz
    
    def get_sparsity(self):
        """Calculate current sparsity percentage"""
        total = self.count_parameters()
        nonzero = self.count_nonzero_parameters()
        return 100.0 * (total - nonzero) / total
