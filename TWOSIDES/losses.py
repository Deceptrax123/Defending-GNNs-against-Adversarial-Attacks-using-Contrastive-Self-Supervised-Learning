import torch
import torch.nn.functional as F
from torch.nn import Module


class InfoNCELoss(Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()

    def forward(self, input, target):
        pass
