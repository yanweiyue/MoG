import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam

from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader
from MoG import MoG


def cross_validation_with_val_set(dataset, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay,args, logger=None,device=torch.device("cpu")):

    val_accs, test_accs, durations = [], [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = MoG(dataset.num_features,args['hidden_spl'], dataset.num_classes,dataset.edge_attr.size(1), args, device)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends,
                     'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except ImportError:
                pass

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader,device,args['use_topo'])
            val_accs.append(eval_acc(model, val_loader,device,args['use_topo']))
            test_accs.append(eval_acc(model, test_loader,device,args['use_topo']))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_acc': val_accs[-1],
                'test_acc': test_accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            
            print(eval_info)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends,
                     'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    val_acc, test_acc, duration = tensor(val_accs), tensor(test_accs), tensor(durations)
    val_acc, test_acc = val_acc.view(folds, epochs), test_acc.view(folds, epochs)
    val_acc_best, argmax = val_acc.max(dim=1)
    test_acc_best = test_acc[torch.arange(folds, dtype=torch.long), argmax]

    val_best_mean = val_acc_best.mean().item()
    acc_mean = test_acc_best.mean().item()
    acc_std = test_acc_best.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {val_best_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return val_best_mean, acc_mean, acc_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def row_normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    if isinstance(features, torch.Tensor):
        rowsum = torch.sum(features, dim=1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        features = r_mat_inv @ features
    return features

def train(model, optimizer, loader,device,use_topo=True):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, edge_attr = data.x, data.edge_index, torch.zeros((data.edge_index.size(1),1),device=data.edge_index.device).float()
        if use_topo:
            model.learner.get_topo_val(edge_index)
        mask,add_loss = model.learner(x = x, edge_index = edge_index, 
                                  temp = 0.05,edge_attr = edge_attr, training = True)  # masks:size(num_edges)
        out = model.gnn(data,mask)
        loss = F.nll_loss(out, data.y.view(-1)) + add_loss * 1e-1
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()

    return total_loss / len(loader.dataset)


def eval_acc(model, loader,device,use_topo=True):
    model.eval()

    correct = 0
    mask_one = 0
    mask_numel = 0
    
    for data in loader:
        data = data.to(device)
        x, edge_index, edge_attr = data.x, data.edge_index, torch.zeros((data.edge_index.size(1),1),device=data.edge_index.device).float()      
        with torch.no_grad():
            if use_topo:
                model.learner.get_topo_val(edge_index)
            mask,add_loss = model.learner(x = x, edge_index = edge_index, 
                                    temp = 0.05,edge_attr = edge_attr, training = True)  # masks:size(num_edges)
            pred = model.gnn(data,mask).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        mask_one += mask.sum()
        mask_numel+=mask.numel()
    print("sparsity:",mask_one/mask_numel)
    return correct / len(loader.dataset)


def eval_loss(model, loader,device,use_topo=True):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        x, edge_index, edge_attr = data.x, data.edge_index, torch.zeros((data.edge_index.size(1),1),device=data.edge_index.device).float()
        with torch.no_grad():
            if use_topo:
                model.learner.get_topo_val(edge_index)            
            mask,add_loss = model.learner(x = x, edge_index = edge_index, 
                                    temp = 0.05,edge_attr = edge_attr, training = True)  # masks:size(num_edges)
            out = model.gnn(data,mask)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


@torch.no_grad()
def inference_run(model, loader, bf16,device):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)
