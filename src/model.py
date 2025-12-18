import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, dropout_rate: float = 0.0) -> None:
        super(NeuralNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)