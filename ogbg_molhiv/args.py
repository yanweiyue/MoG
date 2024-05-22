import argparse

def parser_loader():
    parser = argparse.ArgumentParser(description='MoG(molhiv)')
    # experiment settings
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--gnn', type=str, default='pna')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--temp_N',type=int,default=50)
    parser.add_argument('--temp_r',type=float,default=1e-3)
    
    # args about optim
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay',type=float,default=1e-7)
    
    # args about gnn
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--drop_ratio', type=float, default=0.3)
    parser.add_argument('--emb_dim',type = int,default=300)    
    
    # args about SpLearner expert
    parser.add_argument('--hidden_spl',type=float,default=128)
    parser.add_argument('--num_layers_spl',type=int,default=2)
    
    # args about MoE
    parser.add_argument('--expert_select',type=int,default=3)
    parser.add_argument('--k_list', nargs='+', type=float)
    parser.add_argument('--lam',type=float,default=1e-1)
    parser.add_argument('--use_topo',default=False,action='store_true')
    
    # dataset args
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    
    args = vars(parser.parse_args())
    assert len(args['k_list']),"The sparsity of each sparsifier must be specified"
    
    return args

