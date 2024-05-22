import argparse

def parser_loader():
    parser = argparse.ArgumentParser(description='MoG(mnist)')
    # experiment settings
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--temp_N',type=int,default=50)
    parser.add_argument('--temp_r',type=float,default=1e-3)
    parser.add_argument('--seed',type =int,default=123)
    
    # args about optim
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay',type=float,default=1e-7)
    
    # args about gnn
    
    # args about SpLearner expert
    parser.add_argument('--hidden_spl',type=float,default=128)
    parser.add_argument('--num_layers_spl',type=int,default=2)
    
    # args about MoE
    parser.add_argument('--expert_select',type=int,default=3)
    parser.add_argument('--k_list', nargs='+', type=float)
    parser.add_argument('--lam',type=float,default=0.1)
    parser.add_argument('--use_topo',default=False,action='store_true')
        
    # dataset args
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    
    args = vars(parser.parse_args())
    assert len(args['k_list']),"The sparsity of each sparsifier must be specified"
    
    return args

