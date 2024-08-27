import torch
import torch.nn as nn

class MyModel(nn.Module, GaussianMixin):
    def __init__(self):
        # GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(37, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12),
            nn.ELU()
        )
    
    def forward(self, x):
        return self.layers(x)
    
model = MyModel()
