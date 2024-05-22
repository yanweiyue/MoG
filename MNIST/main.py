import os.path as osp
from args import parser_loader
import torch
import torch.nn.functional as F
import numpy as np
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader

from torch_geometric.utils import normalized_cut

from gnn import Net
from MoG import MoG

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


def train(model,optimizer,train_loader,train_dataset,epoch,device,use_topo=True):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    train_loss = 0
    mask_one = 0
    mask_numel = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x, edge_index, edge_attr = data.x, data.edge_index, None
        if use_topo:
            model.learner.get_topo_val(edge_index)        
        mask,add_loss = model.learner(x = x, edge_index = edge_index, 
                                  temp = 1e-3,edge_attr = edge_attr, training = True)  # masks:size(num_edges)
        out = model.gnn(data,mask)
        
        loss = F.nll_loss(out, data.y) + add_loss * 1e-1
        train_loss += loss
        loss.backward()
        optimizer.step()
        mask_one += mask.nonzero().size(0)
        mask_numel+=mask.numel()
    sparsity = (mask_one/mask_numel)
    print("sparsity:",sparsity)     
    return train_loss/len(train_dataset),sparsity

def test(model,test_loader,test_dataset,device,use_topo=True):
    model.eval()
    correct = 0
    
    for data in test_loader:
        data = data.to(device)
        x, edge_index, edge_attr = data.x, data.edge_index, None
        if use_topo:
            model.learner.get_topo_val(edge_index)        
        mask,add_loss = model.learner(x = x, edge_index = edge_index, 
                                  temp = 1e-3,edge_attr = edge_attr, training = False)  # masks:size(num_edges)
        out = model.gnn(data,mask)
        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
           
    return correct / len(test_dataset)

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parser_loader()
    fix_seed(args['seed'])
    device = torch.device("cuda:" + args['device']) if torch.cuda.is_available() else torch.device("cpu")

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
    transform = T.Cartesian(cat=False)
    train_dataset = MNISTSuperpixels(path, True, transform=transform)
    test_dataset = MNISTSuperpixels(path, False, transform=transform)
    
    num_samples = len(train_dataset)
    num_val = num_samples // 10
    valid_dataset = train_dataset[:num_val]
    train_dataset = train_dataset[num_val:]    
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset,batch_size=64,shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = MoG(train_dataset.num_features,args['hidden_spl'], train_dataset.num_classes,0, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    
    best_valid_acc = 0
    best_test_acc = 0
    best_sparsity = 0
    for epoch in range(1, args['epochs']+1):
        loss,sparsity = train(model,optimizer,train_loader,train_dataset,epoch,device,use_topo=args['use_topo'])
        valid_acc = test(model,valid_loader,valid_dataset,device,use_topo=args['use_topo'])
        test_acc = test(model,test_loader,test_dataset,device,use_topo=args['use_topo'])
        if valid_acc>best_valid_acc:
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            best_sparsity = sparsity
        print(f'Epoch: {epoch:02d},Train: {loss:.4f}, Valid: {valid_acc:.4f}, Test: {test_acc:.4f}, Best Valid: {best_valid_acc:.4f}, Best Test: {best_test_acc:.4f}, Best Sparsity:{best_sparsity}')
    


if __name__ == '__main__':
    main()