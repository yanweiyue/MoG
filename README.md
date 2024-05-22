# Usage of MoG

## Introduce
Under each folder:

+ gnn.py and conv.py are basic gnn(GCN,SAGE,PNA,Deepergcn...) implementation codes

+ Splearner.py implements a node-granular graph pruner

+ MoE.py implements a model that mixes pruners with different sparsities

+ MoG.py bridges the connection between MoE and GNN

## Usage example

For the ogbn_proteins dataset, please `cd ogbn_proteins` first.
```
python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --num_layers 28 --block res+ --gcn_aggr max --k_list 1 --expert_select 1
python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --num_layers 28 --block res+ --gcn_aggr max --k_list 0.9 0.7 0.5 --expert_select 2
python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --num_layers 28 --block res+ --gcn_aggr max --k_list 0.7 0.5 0.3 --expert_select 2
python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --num_layers 28 --block res+ --gcn_aggr max --k_list 0.4 0.3 0.2 --expert_select 2
```

For the MNIST dataset, please `cd MNIST` first.
```
python main.py --k_list 1 --expert_select 1
python main.py --k_list 0.8 0.5 0.4 --expert_select 2
python main.py --k_list 0.6 0.3 0.2 --expert_select 2
python main.py --k_list 0.35 0.1 0.1 --expert_select 2
```
