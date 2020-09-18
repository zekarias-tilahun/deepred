from deepred import DeepRed
from data import load_data
import utils

from collections import namedtuple
from sklearn import metrics

import networkx as nx
import pandas as pd
import numpy as np

import itertools
import argparse
import os


class InteractionPrediction:
    
    def __init__(self, args):
        self.args = args
        
    def load(self):
        args = self.args
        self.dev_data = load_data(args, partition_name='dev')
        self.test_data = load_data(args, partition_name='test')
        self.deepred = DeepRed(num_nodes=self.dev_data.num_nodes, dim=args.dim, dropout=args.dropout, temporal=args.temporal, root_dir=args.root_dir)
        path = os.path.join(args.root_dir, utils.MODEL_DIR, f"deepred_epoch_{args.epoch}.pt")
        self.deepred.load_model(path)
        
    def compute_scores(self, batches):
        scores = None
        for batch in batches:
            self.deepred.fit_batch(batch)
            cur_scores = self.deepred.model.user_rep.mul(self.deepred.model.item_rep).sum(dim=1).detach().cpu().numpy()
            scores = cur_scores if scores is None else np.concatenate([scores, cur_scores])
        return scores
    
    def predict(self, is_dev=True):
        input_data = self.dev_data if is_dev else self.test_data
        utils.log(f"Predicting interaction scores for the {'dev' if is_dev else 'test'} data ...")
        pos_scores = self.compute_scores(input_data.batches())
        neg_scores = self.compute_scores(input_data.negative_batches())
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(pos_scores.shape[0]), np.zeros(neg_scores.shape[0])])
        return scores, labels
        

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root-dir', '-r', type=str, default='../data/sider',
                        help='Path to a root directory of a dataset')
    parser.add_argument('--nbr-size', '-n', type=int, default=200, help='The same nbr_size value used during training')
    parser.add_argument('--dim', '-d', type=int, default=128, help='The same emb_dim value used during training')
    parser.add_argument('--epoch', type=int, default=1, help="The epoch number to be evaluated")
    parser.add_argument('--sfx', type=str, default='', 
                        help='Used to add suffix for specific cases, for example when evaluating a particular training ratio')
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.set_defaults(binary=False)

    return parser.parse_args()


def main():
    args = parse_args()
    args.temporal = False
    args.dropout = 0.
    ip = InteractionPrediction(args)
    ip.load()
    scores, labels = ip.predict()
    auc = metrics.roc_auc_score(y_true=labels, y_score=scores, average="micro")
    utils.log(f"Validation AUC: {auc}")
    
    
if __name__ == '__main__':
    main()