import argparse

def parser_loader():
    parser = argparse.ArgumentParser(description='MoG(arxiv)')
    # experiment settings
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--temp_N',type=int,default=50)
    parser.add_argument('--temp_r',type=float,default=1e-3)
    parser.add_argument('--seed',type=int,default=5)
    
    # args about optim
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay',type=float,default=1e-4)
    
    # args about gnn
    parser.add_argument('--hidden_channels',type=float,default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)    
    
    # args about SpLearner expert
    parser.add_argument('--hidden_spl',type=float,default=128)
    parser.add_argument('--num_layers_spl',type=int,default=2)
    
    # args about MoE
    parser.add_argument('--expert_select',type=int,default=3)
    parser.add_argument('--k_list', nargs='+', type=float)
    parser.add_argument('--lam',type=float,default=1e-1)
    parser.add_argument('--use_topo',default=True,action="store_true")
    
    args = vars(parser.parse_args())
    assert len(args['k_list']),"The sparsity of each sparsifier must be specified"
    
    return args

