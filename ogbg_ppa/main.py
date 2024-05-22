import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import os.path as osp

from tqdm import tqdm
from args import parser_loader
import numpy as np
### importing OGB
from ogb.graphproppred import Evaluator
from MoG import MoG
from torch_scatter import scatter


multicls_criterion = torch.nn.CrossEntropyLoss()

def add_zeros(data):
    data.x = torch.zeros((data.num_nodes), dtype=torch.long)
    return data


def row_normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    if isinstance(features, torch.Tensor):
        rowsum = torch.sum(features, dim=1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        features = r_mat_inv @ features
    return features

def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                        data.edge_index[0],
                        dim=0,
                        dim_size=data.num_nodes,
                        reduce=reduce)
        data.x = row_normalize_features(data.x)
    else:
        raise Exception('Unknown Aggregation Type')
    return data

def train(model, device, loader, optimizer, task_type,temp,use_topo=True):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            x, edge_index, edge_attr,batch_batch,topo_val = batch.x, batch.edge_index, batch.edge_attr, batch.batch,batch.topo_val
            if use_topo:
                model.learner.topo_val =topo_val
            else:
                model.learner.topo_val = None
            mask,add_loss = model.learner(x = x, edge_index = edge_index, 
                                  temp = temp,edge_attr = edge_attr, training = True)  # masks:size(num_edges)
            
            pred = model.gnn(batch,mask)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            loss+=add_loss*0.1
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator,temp,use_topo=True):
    model.eval()
    y_true = []
    y_pred = []

    masks = []
    importance = torch.zeros_like(model.k_list)
    load = torch.zeros_like(model.k_list)
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        x, edge_index, edge_attr, batch_batch,topo_val = batch.x, batch.edge_index, batch.edge_attr, batch.batch,batch.topo_val

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                if use_topo:
                    model.learner.topo_val =topo_val
                else:
                    model.learner.topo_val = None
                mask,add_loss = model.learner(x = x, edge_index = edge_index, 
                                  temp = temp,edge_attr = edge_attr, training = False)  # masks:size(num_edges)
                pred = model.gnn(batch,None)

            y_true.append(batch.y.view(-1,1).detach().cpu())

            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
            masks.append(mask.detach().cpu())
            load += model.learner.load
            if step ==0:
                importance = model.learner.importance.unsqueeze(0)
            else:
                importance = torch.cat([importance,model.learner.importance.unsqueeze(0)],dim=0)
            
            
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    masks = torch.cat(masks,dim=0)
    spar = masks.sum()/masks.numel()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    print("sparsity:",spar)  # NOTE
    print("load:",load)
    print("importance:",torch.mean(importance,dim=0))
    return evaluator.eval(input_dict),0,0


def main():
    # Training settings
    args = parser_loader()

    device = torch.device("cuda:" + args['device']) if torch.cuda.is_available() else torch.device("cpu")
    ### automatic dataloading and splitting
    path= osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args['dataset'])
    dataset = torch.load(path+'.pt')
    
    split_idx = dataset.get_idx_split()
    
    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args['dataset'])

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args['batch_size'], shuffle=True, num_workers = args['num_workers'])
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args['batch_size'], shuffle=False, num_workers = args['num_workers'])
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args['batch_size'], shuffle=False, num_workers = args['num_workers'])

    model = MoG(args['hidden_spl'], dataset.num_classes,dataset.edge_attr.size(1), args, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    valid_curve = []
    test_curve = []
    train_curve = []
    train_spar_curve = []
    train_load_curve = []

    for epoch in range(1, args['epochs'] + 1):
        # calculate the temperature of sparsity learner
        if (epoch-1) % args["temp_N"] == 0:
            decay_temp = np.exp(-1*args["temp_r"]*epoch)
            temp = max(0.05, decay_temp)
        
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type,temp,args['use_topo'])

        print('Evaluating...')
        train_perf,train_spar,train_load = eval(model, device, train_loader, evaluator,temp,args['use_topo'])
        valid_perf,_,_ = eval(model, device, valid_loader, evaluator,temp,args['use_topo'])
        test_perf,_,_ = eval(model, device, test_loader, evaluator,temp,args['use_topo'])
        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        train_spar_curve.append(train_spar)
        train_load_curve.append(train_load)
        
    
    if 'multiclass' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    print('Best validation train sparsity:{}'.format(train_spar_curve[best_val_epoch]))
    print('Best validation train load:{}'.format(train_load_curve[best_val_epoch]))

    if not args['filename'] == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
