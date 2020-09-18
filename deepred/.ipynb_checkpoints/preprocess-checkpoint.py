import utils

import networkx as nx
import numpy as np

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./data/wikipedia/')
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.add_argument('--tr-rate', type=float, default=.6)
    parser.add_argument('--dev-rate', type=float, default=.2)
    parser.add_argument('--temporal', dest='temporal', action='store_true')
    parser.add_argument('--static', dest='temporal', action='store_false')
    parser.add_argument('--log-level', type=int, default=0)
    parser.set_defaults(temporal=False)
    parser.set_defaults(binary=False)
    return parser.parse_args()


def process(args):
    path = os.path.join(args.root_dir, utils.PROCESSED_DIR, 'interaction.txt')
    int_data = utils.interactions_from_text(path=path, compile_nodes=True, temporal=args.temporal)
    utils.log(f'Spliting interactions into training and test interactions with a ratio of {args.tr_rate}')
    train_interactions, dev_interactions, test_interactions, edge_atr = utils.split_interactions(
        interactions=int_data.interaction_frame, train_rate=args.tr_rate, 
        dev_rate=args.dev_rate, temporal=args.temporal)
    tr_g = nx.from_pandas_edgelist(train_interactions, source='Users', target='Items', 
                                   edge_attr=edge_atr, create_using=nx.MultiDiGraph)
    dv_g = nx.from_pandas_edgelist(dev_interactions, source='Users', target='Items', 
                                   edge_attr=edge_atr, create_using=nx.MultiDiGraph)
    te_g = nx.from_pandas_edgelist(test_interactions, source='Users', target='Items', 
                                   edge_attr=edge_atr, create_using=nx.MultiDiGraph)
    return int_data.users, int_data.items, tr_g, dv_g, te_g


def save_graph(graph, root_dir, partition='train', temporal=False, binary=True, state=False):
    if binary:
        path = os.path.join(root_dir, utils.PROCESSED_DIR, f'{partition}.gpickle')
        utils.log(f'Saving interactions to {path}')
        nx.write_gpickle(graph, path)
    else:
        path = os.path.join(root_dir, utils.PROCESSED_DIR, f'{partition}.txt')
        utils.log(f'Saving interactions to {path}')
        edges = list(graph.edges(data=True if temporal else False))
        if temporal:
            attributes = [utils.TIME_ATR]
        else:
            attributes = None
        utils.save_interactions(edges, path, attributes=attributes)
    

def save_nodes(users, items, root_dir, binary=True):
    if binary:
        path = os.path.join(root_dir, utils.PROCESSED_DIR, 'nodes')
        np.savez(path, users=users, items=items)
    else:
        path = os.path.join(root_dir, utils.PROCESSED_DIR, 'nodes.txt')
        with open(path, 'w') as f:
            f.write('begin-users\n')
            for u in users:
                f.write(f'{u}\n')
                
            f.write('begin-items\n')
            for i in items:
                f.write(f'{i}\n')
                
def save_state_label_weights(state_weights, root_dir):
    path = os.path.join(root_dir, utils.PROCESSED_DIR, 'state_label_weights.txt')
    utils.log(f'Saving state label weights to {path}')
    with open(path,  'w') as f:
        for state, weight in state_weights:
            f.write(f'{state} {weight}\n')
        
    
def main():
    args = parse_args()
    utils.LOG_LEVEL = args.log_level
    utils.create_root_sub_dirs(args.root_dir)
    users, items, train_graph, dev_graph, test_graph = process(args)
    save_graph(graph=train_graph, root_dir=args.root_dir, temporal=args.temporal, binary=args.binary, partition='train')
    save_graph(graph=dev_graph, root_dir=args.root_dir, temporal=args.temporal, binary=args.binary, partition='dev')
    save_graph(graph=test_graph, root_dir=args.root_dir, temporal=args.temporal, binary=args.binary, partition='test')
    save_nodes(users=users, items=items, root_dir=args.root_dir, binary=args.binary)
    

if __name__ == '__main__':
    main()
