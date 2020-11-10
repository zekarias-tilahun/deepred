from utils import *


from collections import namedtuple, Counter

import networkx as nx
import pandas as pd
import numpy as np

from torch.utils import data
import torch


global TIME_ATR

np.random.seed(7)



class DeepRedDataset(data.Dataset):

    def __init__(self, args, name, users, items):
        super().__init__()
        self.users = users
        self.items = items
        self._args = args
        self._read(name)
        self._init()
        self.init_neighborhood()
        self.interactions = np.array(self.interactions)
        
    @property
    def total_batches(self):
        return len(self.partitions)
    
    @property
    def nodes(self):
        return sorted(set(self.users) | set(self.items))

    def _read(self, name):
        """
        Reads a partition (train, dev, or test) of the dataset specfied by its name
        
        :param name: partition name
        """
        args = self._args
        file_name = f'{name}.gpickle' if args.binary else f'{name}.txt'
        compile_nodes = True if self.users is None or self.items is None else False
        path = os.path.join(args.root_dir, PROCESSED_DIR, file_name)
        data = read(path=path, compile_nodes=compile_nodes, temporal=args.temporal, binary=args.binary)
        self.interaction_graph = data.interaction_graph
        if compile_nodes:
            self.users = data.users
            self.items = data.items
        self.min_item_id = min(self.items)
        # nodes in the connected component
        self.nodes_in_cc = set(self.interaction_graph.nodes())
        self.interaction_graph.add_nodes_from(self.users + self.items)
        if args.temporal:
            self.interactions = sorted(self.interaction_graph.edges(data=True), 
                                       key=lambda l: l[2][TIME_ATR])
        else:
            self.interactions = list(self.interaction_graph.edges())
            
    def is_item(self, id_):
        return id_ >= self.min_item_id

    def _init(self):
        """
        Initializes commonly used instance variables
        
        """
        args = self._args
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        self.num_nodes = self.num_users + self.num_items
        self.num_interactions = len(self.interactions)
        self.cc_nodes = {n for n in self.interaction_graph.nodes() 
                         if self.interaction_graph.degree[n] > 0}
        self.item_ug_dist_75, self.item_ug_dist_75_lst = {}, []
        self.partitions = []

    def init_uni_gram_distributions(self):
        """
        Initializes uni-gram distributions

        :return:
        """

        def init(nodes):
            ug_dist_75, ug_dist_75_lst = {}, []
            for node in nodes:
                degree_75 = self.interaction_graph.degree[node] * .75
                ug_dist_75[node] = degree_75 / len(self.cc_nodes)
                ug_dist_75_lst.extend([node] * int(degree_75))
            return ug_dist_75, ug_dist_75_lst

        self.user_ug_dist_75, self.user_ug_dist_75_lst = init(self.users)
        self.item_ug_dist_75, self.item_ug_dist_75_lst = init(self.items)
        
    def init_negative_interactions(self):
        path = os.path.join(self._args.root_dir, PROCESSED_DIR, 'negative.txt')
        self.negative_interaction_graph = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph)
        self.negative_interactions = np.array(list(self.negative_interaction_graph.edges()))
        
        #size = self.interactions.shape[0]
        #negative_interaction_set = set()
        #nodes = self.nodes
        #np.random.shuffle(nodes)
        #while True:
        #    u, v = np.random.randint(0, len(nodes), 2).tolist()
        #    if (u, v) not in negative_interaction_set and not self.interaction_graph.has_edge(u, v):
        #        negative_interaction_set.add((u, v))
        #        if len(negative_interaction_set) >= size:
        #            break
        #self.negative_interactions = np.array(list(negative_interaction_set))
        #self.negative_interaction_graph = nx.DiGraph(list(negative_interaction_set))
        
    def init_neighborhood(self, nbr_size=None):
        pass

    def neighborhood_sampler(self, **kwargs):
        """
        Samples neighbors/histories of users and items
        
        Possible parameters to be passed
        
        :param node: A node for which neighbors are to be sampled
        :param ug_dist_75: A uni-gram distribution raised to .75
        :param sampling_method: The sampling method
        :return:
        """
        pass

    def init_partitions(self, **kwargs):
        """
        Partitions indices into batches

        :return:
        """
        pass
    
    def batches(self, **kwargs):
        """
        A batch generator 
        
        :return:
        """
        pass


class StaticDataset(DeepRedDataset):

    def __init__(self, args, name, users=None, items=None):
        super().__init__(args, name=name, users=users, items=items)

    def init_neighborhood(self, nbr_size=None):
        ns = self._args.nbr_size if nbr_size is None else nbr_size
        # We add one to the inner dimmension because 0 is used for zero-padding in the model
        self.neighborhood_matrix = np.zeros(shape=(self.num_nodes + 1, ns))
        self.neighborhood_mask = np.zeros(shape=(self.num_nodes + 1, ns))
        
    def _higher_order_sampler(self, node, ug_dist_75, num_walks=10, walk_length=40, sampling_method=UNIFORM_SAMPLING):
        g = self.interaction_graph
        args = self._args
        walks = []
        neighbors = [other for _, other in interaction_edges(node, data=False) if other in self.nodes_in_cc]
        for i in range(num_walks):
            
            walk = [node]
            counter = 0
            while len(walk) < 40:
                counter += 1
                
                cur = walk[-1]
                interaction_edges = g.in_edges if self.is_item(cur) else g.out_edges
                if len(interaction_edges(node)) > 0:
                    # Mask nodes in dev and test set
                    next_nodes = [other for _, other in interaction_edges(cur, data=False)
                                  if other != cur and other in self.nodes_in_cc]
                    if len(next_nodes) > 0:
                        next_node = np.random.choice(next_nodes)
                        walk.append(next_node)
                    else:
                        break
                else:
                    break
            if len(walk) == 1:
                break
            walks += walk
        if 1 < len(walks) < args.nbr_size:
            samples = np.zeros(args.nbr_size)
            samples[:len(walks)] = walks
            return samples
            
        if len(walks) > args.nbr_size:
            neighbors, weights = zip(*Counter(walks).items())
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            return np.random.choice(neighbors, size=args.nbr_size, replace=False, p=weights)
        return np.zeros(args.nbr_size)
    
    def _first_order_sampler(self, node, ug_dist_75, sampling_method=UNIFORM_SAMPLING):
        g = self.interaction_graph
        args = self._args
        samples = np.zeros(args.nbr_size)
        interaction_edges = g.in_edges if self.is_item(node) else g.out_edges
        #node_interactions = list(zip(*list(interaction_edges(node, data=False))))
        #if len(node_interactions) == 0:
        #    return samples
        
        #_, neighbors = node_interactions
        
        #
        neighbors = [other for _, other in interaction_edges(node, data=False) if other in self.nodes_in_cc]
        if len(neighbors) == 0:
            return samples
        neighbors = np.array(neighbors)

        if neighbors.shape[0] < args.nbr_size:
            samples[:neighbors.shape[0]] = neighbors
            return samples
        #
        
        """
        Sampling neighbors according to the uni-gram distribution of node degrees raised to .75
        TODO:
        """
        weights = None
        if sampling_method == BIASED_SAMPLING:
            weights = np.array([ug_dist_75[nbr] for nbr in neighbors])
            weights /= weights.sum()
            #weights = 1 - weights / 0.  # TODO: adjust weights

        """If weights is None sampling is uniform"""
        neighbors = np.random.choice(neighbors, size=args.nbr_size, replace=False, p=weights)
        return neighbors
        

    def neighborhood_sampler(self, node, ug_dist_75, order=FIRST_ORDER_SAMPLING, sampling_method=UNIFORM_SAMPLING):
        if order == FIRST_ORDER_SAMPLING:
            return self._first_order_sampler(node=node, ug_dist_75=ug_dist_75, sampling_method=sampling_method)
        elif order == HIGHER_ORDER_SAMPLING:
            return self._higher_order_sampler(node=node, ug_dist_75=ug_dist_75, sampling_method=sampling_method)

    def build_neighborhood(self):
        """
        Builds node neighborhood matrix along with an associated mask matrix, which will be used to cancel
        the effect of the padded zeros during a softmax computation.
        """
        def build(nodes, ug_dist_75, user_nodes=True):
            log(f'Building {"users" if user_nodes else "items"} interaction history ...')
            counter = 0
            for node in nodes:
                neighbors = self.neighborhood_sampler(node=node, ug_dist_75=ug_dist_75)
                self.neighborhood_matrix[node] = neighbors
                log(f"{counter}/{len(nodes)}", prefix=PROG, cr=True)
                counter += 1
            log()

        
        args = self._args
        build(self.users, self.item_ug_dist_75)
        build(self.items, self.user_ug_dist_75, user_nodes=False)
        self.neighborhood_mask = get_neighborhood_mask(self.neighborhood_matrix)
        self.neighborhood_matrix = self.neighborhood_matrix.astype('int')
        
    def init_partitions(self, **kwargs):
        
        indices = np.random.permutation(self.num_interactions)
        size = indices.shape[0]
        for start in range(0, size, BATCH_SIZE):
            end = start + BATCH_SIZE if size - start > BATCH_SIZE else size
            self.partitions.append(indices[start: end])
            
    def _batches(self, interactions):
        for batch_indices in self.partitions:
            users = interactions[batch_indices, 0]
            user_nh = self.neighborhood_matrix[users]
            user_mask = self.neighborhood_mask[users]
            items = interactions[batch_indices, 1]
            item_nh = self.neighborhood_matrix[items]
            item_mask = self.neighborhood_mask[items]
            yield collate_function((users, user_nh, user_mask, items, item_nh, item_mask))

    def batches(self):
        return self._batches(self.interactions)
            
    def negative_batches(self):
        self.init_negative_interactions()
        return self._batches(self.negative_interactions)
            

class TemporalDataset(DeepRedDataset):

    def __init__(self, args, name, users=None, items=None):
        super().__init__(args=args, name=name, users=users, items=items)
        
    def __len__(self):
        return self.num_interactions
    
    def init_neighborhood(self, nbr_size=None):
        log('Initializing temporally ordered interaction history ...')
        self.neighborhood = [None]  # Index 0 is used for padding
        self.timestamps = [None]  # Index 0 is used for padding
        self.user_states = [None] # Index 0 is used for padding
        ig = self.interaction_graph
        for i in range(1, self.num_nodes + 1):
            interactions = ig.in_edges if self.is_item(i) else ig.out_edges
            i_interactions = sorted(interactions(i, data=True), key=lambda l: l[2][TIME_ATR])
            if len(i_interactions) > 0:
                _, nbrs, edge_data = zip(*i_interactions)
                self.neighborhood.append(np.array(nbrs)) 
                self.timestamps.append(np.array([data[TIME_ATR] for data in edge_data]))
            else:
                self.neighborhood.append(np.array([]))
                self.timestamps.append(np.array([]))

    def neighborhood_sampler(self, node, current_time, nbr_size):
        all_interaction_events_of_node = self.timestamps[node]
        previous_interaction_events = np.argwhere(all_interaction_events_of_node < current_time).flatten()
        previous_interaction_times = all_interaction_events_of_node[previous_interaction_events]
        previously_interacted_neighbors = self.neighborhood[node][previous_interaction_events]
        delta_from_neighborhood = current_time - previous_interaction_times
        delta_samples = np.zeros(nbr_size, dtype='float32')
        samples = np.zeros(nbr_size, dtype='int')
        num_events = previously_interacted_neighbors.shape[0]
        if 0 < num_events <= nbr_size:
            samples[:num_events] = previously_interacted_neighbors
            delta_samples[:num_events] = delta_from_neighborhood
        elif len(previously_interacted_neighbors) > nbr_size:
            samples[:] = previously_interacted_neighbors[-nbr_size:]
            delta_samples[:] = delta_from_neighborhood[-nbr_size:]
            
        mask = get_neighborhood_mask(samples)
        return samples, delta_samples, mask
    
    def __getitem__(self, index):
        args = self._args
        user, item, edge_data = self.interactions[index]
        current_time = edge_data[TIME_ATR]
        user_nh, user_nh_dt, user_mask = self.neighborhood_sampler(user, current_time, args.nbr_size)
        item_nh, item_nh_dt, item_mask = self.neighborhood_sampler(user, current_time, args.nbr_size)
        return user, user_nh, user_nh_dt, user_mask, item, item_nh, item_nh_dt, item_mask
    
    def init_partitions(self, **kwargs):
        assert 'partitions' in kwargs
        self.partitions = kwargs['partitions']
        
    def batches(self):
        for batch in self.partitions:
            yield collate_function(batch, temporal=True)
            

class TemporalBatchDataIndices:
    user, user_nh, user_nh_dt, user_mask, item, item_nh, item_nh_dt, item_mask = range(8)
    
class StaticBatchDataIndices:
    user, user_nh, user_mask, item, item_nh, item_mask = range(6)
            
def collate_function(data, temporal=False):
    batch_size = data[0].shape[0]
    cond = batch_size == 1
    indexer = TemporalBatchDataIndices if temporal else StaticBatchDataIndices
    batch = {
        'users': expand_if(torch.LongTensor(data[indexer.user]), cond),
        'user_nh': expand_if(torch.LongTensor(data[indexer.user_nh]), cond),
        'user_mask': expand_if(torch.FloatTensor(data[indexer.user_mask]), cond),
        'items': expand_if(torch.LongTensor(data[indexer.item]), cond),
        'item_nh': expand_if(torch.LongTensor(data[indexer.item_nh]), cond),
        'item_mask': expand_if(torch.FloatTensor(data[indexer.item_mask]), cond),
    }
    if temporal:
        batch['user_nh_dt'] = expand_if(torch.FloatTensor(data[indexer.user_nh_dt]), cond)
        batch['item_nh_dt'] = expand_if(torch.FloatTensor(data[indexer.item_nh_dt]), cond)
    return namedtuple('Batch', batch.keys())(*batch.values())


def load_data(args, partition_name=None):
    def get_dataset(name):
        if args.temporal:
            return TemporalDataset(args, name=name, users=users, items=items)
        else:
            return StaticDataset(args, name=name, users=users, items=items)
            
    def build(dataset):
        dataset.init_uni_gram_distributions()
        if args.temporal:
            partitions = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
            dataset.init_partitions(partitions=partitions)
        else:
            dataset.build_neighborhood()
            dataset.init_partitions()
        return dataset
    
    node_path = os.path.join(args.root_dir, PROCESSED_DIR, 'nodes.npz' if args.binary else 'nodes.txt')
    users, items = read_nodes(path=node_path, binary=args.binary)
    if partition_name is None:
        train_dataset = get_dataset(name='train')
        dev_dataset = get_dataset(name='dev')
        return build(train_dataset), build(dev_dataset)
    else:
        return build(get_dataset(name=partition_name))
