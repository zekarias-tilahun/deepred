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
        self.train_data, self.dev_data = load_data(args)
        self.test_data = load_data(args, partition_name='test')
        self.deepred = DeepRed(num_nodes=self.dev_data.num_nodes, dim=args.dim, dropout=args.dropout, temporal=args.temporal, root_dir=args.root_dir)
        path = os.path.join(args.root_dir, utils.MODEL_DIR, f"deepred_epoch_{args.epoch}.pt")
        self.deepred.load_model(path)
        batches = itertools.chain(self.train_data.batches(), self.dev_data.batches(), self.test_data.batches())
        tb = len(self.train_data.partitions) + len(self.dev_data.partitions) + len(self.test_data.partitions)
        
        embeddings = self.deepred.transform(batches, total_batches=tb)
        self.global_user_emb, self.local_user_emb, self.global_item_emb, self.local_item_emb = embeddings
    
    def predict(self, is_dev=True, use_global=False):
        utils.log(f"Predicting interaction scores for the {'dev' if is_dev else 'test'} data ...")
        def compute(interactions):
            scores = []
            for i, (usr, itm) in enumerate(interactions):
                if usr in user_emb and itm in item_emb:
                    scores.append(user_emb[usr].dot(item_emb[itm].transpose()).max())
                utils.log(f"{i}/{len(interactions)}", cr=True, prefix=utils.PROG)
            utils.log()
            return scores
       
        input_data = self.dev_data if is_dev else self.test_data
        input_data.init_negative_interactions()
        user_emb = self.global_user_emb if use_global else self.local_user_emb
        item_emb = self.global_item_emb if use_global else self.local_item_emb
        pos_scores = compute(input_data.interactions)
        neg_scores = compute(input_data.negative_interactions)
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        return scores, labels
        

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root-dir', '-r', type=str, default='../data/sider',
                        help='Path to a root directory of a dataset')
    parser.add_argument('--nbr-size', '-n', type=int, default=300, help='The same nbr_size value used during training')
    parser.add_argument('--dim', '-d', type=int, default=200, help='The same emb_dim value used during training')
    parser.add_argument('--epoch', type=int, default=1, help="The epoch number to be evaluated")
    parser.add_argument('--sfx', type=str, default='', 
                        help='Used to add suffix for specific cases, for example when evaluating a particular training ratio')
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.set_defaults(binary=False)

    return parser.parse_args()


def main():
    args = utils.parse_eval_args()
    args.temporal = False
    args.dropout = 0.5
    ip = InteractionPrediction(args)
    ip.load()
    scores, labels = ip.predict()
    auc = metrics.roc_auc_score(y_true=labels, y_score=scores, average="micro")
    ap = metrics.average_precision_score(y_true=labels, y_score=scores, average='micro')
    utils.log(f"Validation AUC: {auc}")
    utils.log(f"Validation AP: {ap}")
    scores, labels = ip.predict(is_dev=False)
    auc = metrics.roc_auc_score(y_true=labels, y_score=scores, average="micro")
    ap = metrics.average_precision_score(y_true=labels, y_score=scores, average='micro')
    utils.log(f"Test AUC: {auc}")
    utils.log(f"Test AP: {ap}")
    
if __name__ == '__main__':
    main()