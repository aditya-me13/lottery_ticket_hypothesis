import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()

        # conv block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # conv block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # conv block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        self._init_weights()

    def _init_weights(self):
        # kaiming init for relu
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)     # 32→16

        # block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)     # 16→8

        # block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)     # 8→4

        # flatten
        x = x.view(x.size(0), -1)

        # fc
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def count_parameters(self):
        total = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                total += m.weight.numel()
        return total

    def count_nonzero_parameters(self):
        nz = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if hasattr(m, "weight_mask"):
                    nz += int(m.weight_mask.sum().item())
                else:
                    nz += int((m.weight != 0).sum().item())
        return nz

    def get_sparsity(self):
        total = self.count_parameters()
        nonzero = self.count_nonzero_parameters()
        return 100.0 * (total - nonzero) / total
