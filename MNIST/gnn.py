import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
import torch_geometric.transforms as T

from torch_geometric.nn import (
    global_mean_pool,
    graclus,
    max_pool,
    max_pool_x,
)

from conv import NNConv

from torch_geometric.utils import normalized_cut

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class Net(torch.nn.Module):
    def __init__(self,in_dim=32,out_dim=32):
        super().__init__()
        nn1 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, in_dim * 32),
        )
        self.conv1 = NNConv(in_dim, 32, nn1, aggr='max')

        nn2 = Sequential(
            Linear(2, 25),
            ReLU(),
            Linear(25, 32 * 64),
        )
        self.conv2 = NNConv(32, 64, nn2, aggr='max')

        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, out_dim)
        
        self.transform = T.Cartesian(cat=False)

    def forward(self, data,mask=None):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr,edge_mask = mask))
        weight = normalized_cut_2d(data.edge_index, data.pos)  # structure
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=self.transform)
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr,edge_mask = mask))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

