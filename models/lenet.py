import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet300100(nn.Module):
    """
    Fully-connected LeNet architecture for MNIST
    Architecture: 784 -> 300 -> 100 -> 10
    Total parameters: ~266K
    """
    def __init__(self):
        super(LeNet300100, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(784, 300)  # 784*300 + 300 = 235,500
        self.fc2 = nn.Linear(300, 100)  # 300*100 + 100 = 30,100
        self.fc3 = nn.Linear(100, 10)   # 100*10 + 10 = 1,010
        
        # Initialize weights using Glorot (Xavier) initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Gaussian Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            Output logits of shape (batch_size, 10)
        """
        # Flatten input
        x = x.view(x.size(0), -1)  # (batch_size, 784)
        
        # Layer 1: 784 -> 300 with ReLU
        x = F.relu(self.fc1(x))
        
        # Layer 2: 300 -> 100 with ReLU
        x = F.relu(self.fc2(x))
        
        # Layer 3: 100 -> 10 (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_nonzero_parameters(self):
        """Count non-zero parameters (for sparsity calculation)"""
        return sum((p != 0).sum().item() for p in self.parameters())
    
    def get_sparsity(self):
        """Calculate current sparsity percentage"""
        total = self.count_parameters()
        nonzero = self.count_nonzero_parameters()
        return 100.0 * (total - nonzero) / total


# Test the model
if __name__ == "__main__":
    model = LeNet300100()
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    dummy_input = torch.randn(32, 1, 28, 28)  # batch_size=32
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (32, 10)
    print(f"Current sparsity: {model.get_sparsity():.2f}%")