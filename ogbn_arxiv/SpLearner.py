from operator import index
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
eps = 1e-8


class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional


class SpLearner(nn.Module):
    """Sparsification learner"""
    def __init__(self, nlayers, in_dim, hidden, activation, k, weight=True, metric=None, processors=None):
        super().__init__()

        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden))
        for _ in range(nlayers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, 1))

        self.param_init()
        self.activation = activation
        self.k = k
        self.weight = weight

    def param_init(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def internal_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != self.nlayers - 1:
                x = self.activation(x)
        return x

    def gumbel_softmax_sample(self, indices,values, temperature, shape, training):
        """Draw a sample from the Gumbel-Softmax distribution"""
        r = self.sample_gumble(values.shape)
        if training is not None:
            values = torch.log(values) + r.to(indices.device)
        else:
            values = torch.log(values)
        values /= temperature
        y = torch.sparse_coo_tensor(indices=indices, values=values, size=shape,requires_grad=True)
        return torch.sparse.softmax(y, dim=1)

    def sample_gumble(self, shape):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, features, indices,values,shape, temperature,training=None):
        f1_features = torch.index_select(features, 0, indices[0, :])
        f2_features = torch.index_select(features, 0, indices[1, :])
        auv = torch.unsqueeze(values, -1)
        
        temp = torch.cat([f1_features,f2_features,auv],-1)
        temp = self.internal_forward(temp)
        z = torch.reshape(temp, [-1])
        z = F.normalize(z,dim=0)
        
        z = z
        z_matrix = torch.sparse_coo_tensor(indices=indices, values=z, size=shape,requires_grad=True)
        pi = torch.sparse.softmax(z_matrix, dim=1) 
        pi_values = pi.coalesce().values()
        y = self.gumbel_softmax_sample(indices,pi_values, temperature, shape, training) # sparse score
        sparse_indices = y.coalesce().indices()
        sparse_values = y.coalesce().values()
        
        # TODO
        node_idx, num_edges_per_node = sparse_indices[0].unique(return_counts=True)
        k_edges_per_node = (num_edges_per_node.float() * self.k).round().long()
        k_edges_per_node = torch.where(k_edges_per_node>0,k_edges_per_node,torch.ones_like(k_edges_per_node,device=k_edges_per_node.device))
        
        sparse_values,val_sort_idx = sparse_values.sort(descending=True)
        sparse_idx0 = sparse_indices[0].index_select(dim = -1,index = val_sort_idx)
        idx_sort_idx = sparse_idx0.argsort(stable=True,dim=-1,descending = False)
        scores_sorted = sparse_values.index_select(dim=-1,index=idx_sort_idx)

        edge_start_indices = torch.cat((torch.tensor([0],device=y.device), torch.cumsum(num_edges_per_node[:-1], dim=0)))
        edge_end_indices = torch.abs(torch.add(edge_start_indices,k_edges_per_node)-1).long()
        node_keep_thre_cal = torch.index_select(scores_sorted,dim=-1,index=edge_end_indices) 
        node_keep_thre_augmented = node_keep_thre_cal.repeat_interleave(num_edges_per_node)
        mask = BinaryStep.apply(scores_sorted-node_keep_thre_augmented+1e-12) 
        masked_scores = mask*scores_sorted

        idx_resort_idx = idx_sort_idx.argsort()
        val_resort_idx = val_sort_idx.argsort()
        masked_scores = masked_scores.index_select(dim=-1,index = idx_resort_idx)
        masked_scores = masked_scores.index_select(dim=-1,index = val_resort_idx)
        return masked_scores
        
    def write_tensor(self,x,msg):
        with open('temp.txt', "w+") as log_file:
            log_file.write(msg)
            np.savetxt(log_file,x.cpu().detach().numpy())
        
        
        