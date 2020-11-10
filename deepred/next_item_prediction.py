from data import load_data, TemporalBatchDataIndices
from deepred import DeepRed
import utils

import argparse
import time
import os

import torch

indexer = TemporalBatchDataIndices


class NextItemPrediction:

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.args = args
        self.ground_truth = []
        self.predicted_items = []

        self.load()

    @property
    def offset(self):
        return self.index_to_item.shape[0] - self.num_past_events_to_remember

    def load(self):
        """
        Loads everything (data, model, ...) required for next item prediction
        :return:
        """
        args = self.args
        """Loading interaction data"""
        self.train_data, self.dev_data = load_data(args)
        self.test_data = load_data(self.args, partition_name='test')
        train_partitions = torch.utils.data.DataLoader(self.train_data, batch_size=256)
        dev_partitions = torch.utils.data.DataLoader(self.dev_data, batch_size=self.batch_size)
        test_partitions = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size)
        self.train_data.init_partitions(partitions=train_partitions)
        self.dev_data.init_partitions(partitions=dev_partitions)
        self.test_data.init_partitions(partitions=test_partitions)

        """Loading a desired model"""
        self.deepred = DeepRed(num_nodes=self.train_data.num_nodes, dim=args.dim, dropout=args.dropout)
        self.deepred.load_model(path=os.path.join(args.root_dir, utils.MODEL_DIR, f'deepred_epoch_{args.epoch}.pt'))

        """Loading (initializing+processing) embeddings from a trained model"""
        embeddings = self.deepred.transform(self.train_data.batches(), total_batches=len(self.train_data.partitions))
        _, self.short_term_user_emb, _, self.short_term_item_emb = embeddings
        self.process_embeddings()

        """ A simple trick used to discard very old embeddings for the purpose of GPU memory usage"""
        self.num_past_events_to_remember = int(self.past_item_embeddings.shape[0] * .15)
        utils.log(f'Number of past events to remember {self.num_past_events_to_remember}')

    def process_embeddings(self):
        """
        Builds embedding tensors from embedding dictionaries and setup a dictionary and tensor
        based indexes for quick access.

        :return:
        """

        def process(emb_dict):
            """
            Executes the desired process by the outer function

            :param emb_dict: The dictionary from which a tensor is going to be constructed
            :return: An embedding tensor, a dictionary index, and a tensor index
            """
            emb_list = []
            node_to_index = {}
            index_to_node = []
            for n, embeddings in emb_dict.items():
                node = int(n)
                size = embeddings.shape[0]
                for i in range(size):
                    index = len(emb_list)
                    emb_list.append(embeddings[i])
                    utils.populate_dict_of_list(node_to_index, node, index)
                    index_to_node.append(node)
            return torch.FloatTensor(emb_list).to(self.deepred.device), node_to_index, torch.LongTensor(index_to_node).to(
                self.deepred.device)

        """"""
        self.last_embeddings = self.deepred.model.encode.embedding.weight.clone().detach()

        utils.log('Processing embeddings for points in the training set')
        self.past_user_embeddings, self.user_to_index, self.index_to_user = process(self.short_term_user_emb)
        self.past_item_embeddings, self.item_to_index, self.index_to_item = process(self.short_term_item_emb)

    def predict(self, dataset):
        """
        Predicts next item interaction

        :param dataset: The dev/test dataset

        :return:
        """

        num_int = dataset.num_interactions
        start = time.time()
        self.past_item_embeddings = self.past_item_embeddings[-self.num_past_events_to_remember:].detach().clone()
        torch.cuda.empty_cache()
        """ batch contains the interaction data between a true user and item at time t """
        for i, batch in enumerate(dataset.batches()):
            """ compute the ids of the true user and item that interact at time t """
            true_user = int(int(batch[indexer.user].flatten()[0]))
            true_item = int(int(batch[indexer.item].flatten()[0]))

            """ The last embedding of the true user before time t """
            true_user_last_emb = self.last_embeddings[true_user]

            """ Computing the pairwise l2 distance between the embeddings of the true user and all items before t """
            l2_dist = torch.nn.PairwiseDistance()
            true_user_dist_to_items = l2_dist(self.past_item_embeddings, true_user_last_emb)

            """ Find the top-k nearest items and append them to next item predictions """
            top_k_items_info = torch.topk(true_user_dist_to_items, k=self.args.k, largest=False)
            k_nearest_item_ixs = top_k_items_info.indices + self.offset
            k_nearest_items = [int(self.index_to_item[int(ix)]) for ix in k_nearest_item_ixs]
            self.ground_truth.append(true_item)
            self.predicted_items.append(k_nearest_items)

            """ Infer the embedding of the true user and item at time t and save it for future prediction"""
            inputs = {
                'users': batch.users[0].to(self.deepred.device), 'user_nh': batch.user_nh[0].to(self.deepred.device),
                'user_nh_dt': batch.user_nh_dt[0].to(self.deepred.device),
                'user_mask': batch.user_mask[0].to(self.deepred.device),
                'items': batch.items[0].to(self.deepred.device), 'item_nh': batch.item_nh[0].to(self.deepred.device),
                'item_nh_dt': batch.item_nh_dt[0].to(self.deepred.device),
                'item_mask': batch.item_mask[0].to(self.deepred.device),
            }
            self.deepred.model(**inputs)
            self.past_item_embeddings = torch.cat(
                [self.past_item_embeddings[1:], self.deepred.model.item_rep.detach().unsqueeze(0)])
            self.last_embeddings[true_user] = self.deepred.model.user_rep.detach().flatten()
            # self.index_to_user = torch.cat([self.index_to_user, torch.LongTensor([true_user]).to(self.deepred.device)])
            self.index_to_item = torch.cat([self.index_to_item, torch.LongTensor([true_item]).to(self.deepred.device)])

            if (i + 1) % 500 == 0:
                delta = time.time() - start
                utils.log("Predicted {}/{} interactions in {:.2f} seconds".format(
                    i + 1, num_int, delta), cr=True, prefix=utils.PROG)
                torch.cuda.empty_cache()

        end = time.time()
        utils.log("")
        utils.log("Finished in {:.2f} seconds".format(end - start))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root-dir', '-r', type=str, default='../data/wikipedia',
                        help='Path to a root directory of a dataset')
    parser.add_argument('--nbr-size', '-n', type=int, default=100, help='The same nbr_size value used during training')
    parser.add_argument('--dim', '-d', type=int, default=128, help='The same embedding dimmension value used during training')
    parser.add_argument('--k', '-k', type=int, default=10, help="The k-value for recall@k")
    parser.add_argument('--epoch', type=int, help="The epoch number to be evaluated")
    parser.add_argument('--sfx', type=str, default='', 
                        help='Used to add suffix for specific cases, for example when evaluating a particular training ratio')
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.set_defaults(binary=False)

    return parser.parse_args()


def main():
    args = parse_args()
    args.batch_size = 1
    args.temporal = True
    args.dropout = 0.
    npi = NextItemPrediction(args=args)

    utils.log("Predicting next item interactions on the dev set")
    npi.predict(npi.dev_data)
    dev_mrr, dev_recall = utils.evaluate_nip(npi.ground_truth, npi.predicted_items)
    utils.log(f'Results on the dev set MRR:{dev_mrr} Recall: {dev_recall}')

    utils.log("Predicting next item interactions on the test set")
    npi.predict(npi.test_data)
    test_mrr, test_recall = utils.evaluate_nip(npi.ground_truth, npi.predicted_items)
    utils.log(f'MRR:{test_mrr} Recall: {test_recall}')
    suffix = '' if args.sfx == '' else f'_{args.sfx}'
    path = os.path.join(args.root_dir, utils.RESULT_DIR, f'next_item_prediction_result_epoch_{args.epoch}{suffix}.txt')
    with open(path, 'w') as f:
        f.write(f"Partition Metrics Score\n")
        f.write(f'dev MRR {dev_mrr}\n')
        f.write(f'dev Recall@10 {dev_recall}\n')
        f.write(f'test MRR {test_mrr}\n')
        f.write(f'test Recall@10 {test_recall}\n')


if __name__ == '__main__':
    main()
