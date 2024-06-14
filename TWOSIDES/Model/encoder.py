from torch.nn import Module
from torch_geometric.nn import ChebConv
import torch.nn.functional as F


class SpectralMoleculeEncoder(Module):
    def __init__(self, in_features):
        super(SpectralMoleculeEncoder, self).__init__()

        self.gcn1 = ChebConv(in_channels=in_features,
                             out_channels=128, normalization='sym', K=3)
        self.gcn2 = ChebConv(
            in_channels=128, out_channels=256, normalization='sym', K=3)
        self.gcn3 = ChebConv(
            in_channels=256, out_channels=512, normalization='sym', K=3)

        self.conv_mu = ChebConv(in_channels=512, out_channels=256, K=1)
        self.conv_std = ChebConv(in_channels=512, out_channels=256, K=1)

    def forward(self, v, edge_index):
        x = F.relu(self.gcn1(v, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))

        return self.conv_mu(x, edge_index), self.conv_std(x, edge_index)
