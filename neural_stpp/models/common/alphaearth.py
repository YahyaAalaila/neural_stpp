import torch.nn as nn
import torch

class AlphaEarthProjector(nn.Module):
    """64-D -> k-D bounded projector for AlphaEarth embeddings."""
    def __init__(self, in_dim=64, out_dim=16, hidden=64, layernorm=True):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim) if layernorm else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, e):
        x = self.ln(e)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
