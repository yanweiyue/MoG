from MoE import MoE
from gnn import DeeperGCN
import torch
import torch.nn as nn


class MoG(nn.Module):
    def __init__(self, num_features, num_classes, args, device,params=None):
        super(MoG, self).__init__()
        self.args = args
        self.device = device
        self.k_list = torch.tensor(args.k_list,device = device)
        self.learner = MoE(input_size=num_features, 
                           hidden_size=args.hidden_spl,
                           num_experts=self.k_list.size(0), 
                           nlayers=args.num_layers_spl, 
                           activation=nn.ReLU(), 
                           k_list=self.k_list,
                           expert_select = args.expert_select,
                           lam = args.lam)
        self.gnn = DeeperGCN(args)
