from MoE import MoE
from gnn import GNN
import torch
import torch.nn as nn


class MoG(nn.Module):
    def __init__(self, in_channels, out_channels,edge_dim,deg, args, device,params=None):
        super(MoG, self).__init__()
        self.args = args
        self.device = device
        self.k_list = torch.tensor(args["k_list"],device = device)
        self.learner = MoE(input_size=in_channels, 
                           hidden_size=args["hidden_spl"],
                           num_experts=self.k_list.size(0), 
                           nlayers=args["num_layers_spl"], 
                           activation=nn.ReLU(), 
                           k_list=self.k_list,
                           expert_select = args['expert_select'],
                           edge_dim = edge_dim,
                           lam = args['lam'])
        self.gnn = GNN(gnn_type = 'pna', 
                       num_tasks = out_channels, 
                       num_layer = args['num_layers'], 
                       emb_dim = args['emb_dim'], 
                       drop_ratio = args['drop_ratio'], 
                       virtual_node = False,
                       deg=deg)
        
    
