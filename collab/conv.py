import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree
from torch import Tensor
from torch.nn import ModuleList, Sequential
from torch.utils.data import DataLoader
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self,input_dim, emb_dim):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(input_dim, emb_dim)

    def forward(self, x, edge_index, data_mask=None):
        x = self.linear(x)
        row, col = edge_index
        
        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if data_mask is not None:
            norm = deg_inv_sqrt[row] * data_mask * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        return super().reset_parameters()

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self,num_layer,input_dim, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gcn'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim,emb_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        for layer in range(1,num_layer):
            if gnn_type=='gcn':
                self.convs.append(GCNConv(emb_dim,emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data,mask):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        ### computing input node and edge embedding
        h_list = [x]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index,mask)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)
        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
            
        return node_representation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

if __name__ == "__main__":
    pass
