import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.utils import to_dense_adj


class InfoNCELoss(Module):
    def __init__(self, reduction=True):
        super(InfoNCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, z, z1, z2, edge_index, max_nodes):
        # Adjacency Matrix for ys
        adj_y = torch.sigmoid(torch.matmul(z, z.t()))
        adj_y1 = torch.sigmoid(torch.matmul(z1, z1.t()))
        adj_y2 = torch.sigmoid(torch.matmul(z2, z2.t()))

        # Adjacency Matrix for x
        adj_x = to_dense_adj(edge_index=edge_index, max_num_nodes=max_nodes)
        adj_x = adj_x[0, :, :]

        # compute similarity scores g(x,y)
        pos_similarity = torch.cosine_similarity(adj_x, adj_y)
        neg_similarity1 = torch.cosine_similarity(adj_x, adj_y1)
        neg_similarity2 = torch.cosine_similarity(adj_x, adj_y2)

        # Expectation
        E_x_y = torch.sum(torch.mul(adj_y, adj_x), dim=1)
        EPSILON = 1e-8

        info_loss = -E_x_y*torch.log(torch.exp(pos_similarity)/(torch.exp(pos_similarity)+torch.exp(neg_similarity1)
                                                                + torch.exp(neg_similarity2)+EPSILON))
        return torch.mean(info_loss) if self.reduction else info_loss
