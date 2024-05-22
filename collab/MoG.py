from MoE import MoE
from gnn import GCN
import torch
import torch.nn as nn


class MoG(nn.Module):
    def __init__(self, in_dim,emb_dim, out_channels,edge_dim, args, device,params=None):
        super(MoG, self).__init__()
        self.args = args
        self.device = device
        self.k_list = torch.tensor(args["k_list"],device = device)
        self.learner = MoE(
                        in_dim = in_dim,
                           emb_dim=emb_dim, 
                           hidden_size=args["hidden_spl"],
                           num_experts=self.k_list.size(0), 
                           nlayers=args["num_layers_spl"], 
                           activation=nn.ReLU(), 
                           k_list=self.k_list,
                           expert_select = args['expert_select'],
                           edge_dim = edge_dim,
                           lam=args['lam'])
        self.gnn = GCN(input_dim=in_dim,
                       gnn_type = 'gcn', 
                       num_class = out_channels, 
                       num_layer = args['num_layers'], 
                       emb_dim = args['emb_dim'], 
                       drop_ratio = args['drop_ratio'], 
                       virtual_node = False)
        
    def reset_parameters(self):
        self.learner.reset_parameters()
        self.gnn.reset_parameters()
        
