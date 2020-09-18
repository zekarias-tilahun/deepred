from sklearn.metrics import roc_auc_score
from collections import namedtuple
import networkx as nx
import pandas as pd
import numpy as np

import subprocess
import argparse

import sys
import os

EPSILON = 1e-7
BATCH_SIZE = 256
UNIFORM_SAMPLING = 0
BIASED_SAMPLING = 1

NO_LOG = 4
PROGRESS_LOG = 3
EPOCH_LOG = 2
BATCH_LOG = 1
ALL_LOG = 0

LOG_LEVEL = ALL_LOG

INFO = 'DEEPRED:INFO'
PROG = 'DEEPRED:PROG'

TIME_ATR = 'timestamp'


MODEL_DIR = 'model_dir'
OUTPUT_DIR = 'output_dir'
PROCESSED_DIR = 'processed_dir'
RESULT_DIR = 'results_dir'


np.random.seed(7)


def log(msg='', prefix=INFO, cr=False, caller_level=PROGRESS_LOG):
    """
    Message logger

    :param msg: The message
    :param prefix: A prefix to come before the logger
    :param cr: Whether to enable carriage-return
    :param caller_level: An indicator for the caller's (invoker's) level
    :return:
    """ 
    pfx = f'{prefix}:' if prefix is not None else ''
    if msg == '':
        print()
        return

    if caller_level >= LOG_LEVEL:
        if cr:
            sys.stdout.write(f'\r{pfx} {msg}')
            sys.stdout.flush()
        else:
            print(f'{pfx} {msg}')
            

def compile_node_data(data, users, items):
    data['users'] = users
    data['items'] = items
    num_users = len(users)
    num_items = len(items)
    num_nodes = num_users + num_items
    log(f'Number of users {num_users}')
    log(f'Number of items {num_items}')
    log(f'Number of all nodes {num_nodes}')

        
def interactions_from_text(path, compile_nodes, temporal):
    log(msg=f'Reading interactions from {path}')
    if temporal:
        names = ['Users', 'Items', TIME_ATR]
        edge_atr = TIME_ATR
    else:
        names = ['Users', 'Items']
        edge_atr = None

    interaction_frame = pd.read_csv(path, header=None, sep=r'\s+', comment='#', usecols=range(len(names)), names=names)
    num_interactions = interaction_frame.shape[0]
    interaction_graph = nx.from_pandas_edgelist(
        interaction_frame[names], source='Users', target='Items', 
        create_using=nx.MultiDiGraph, edge_attr=edge_atr if edge_atr else None)
    data = {'interaction_graph': interaction_graph, 'interaction_frame': interaction_frame}
        
    if compile_nodes:
        users = sorted(interaction_frame.Users.unique())
        items = sorted(interaction_frame.Items.unique())
        compile_node_data(data=data, users=users, items=items)
    log(f'Number of iteractions {num_interactions}')
    return namedtuple('InteractionData', data.keys())(*data.values())


def interaction_from_binary(path, compile_nodes):        
    log(f'Reading interactions from {path}')
    interaction_graph = nx.read_gpickle(path)
    data = {'interaction_graph': interaction_graph}
    if compile_nodes:
        users, items = zip(*list(interaction_graph.edges()))
        compile_node_data(data=data, users=sorted(users), items=sorted(items))
    log(f'Number of interactions {interaction_graph.number_of_edges()}')
    return namedtuple('InteractionData', data.keys())(*data.values())
    
    
def read(path, compile_nodes=False, temporal=False, binary=True):
    if binary:
        return interaction_from_binary(path=path, compile_nodes=compile_nodes)
    else:
        return interactions_from_text(path=path, compile_nodes=compile_nodes, temporal=temporal)


def read_nodes(path, binary):
    log(f'Reading nodes from {path}')
    if path != '':
        if binary:
            f = np.load(path)
            users = sorted(f['users'])
            items = sorted(f['items'])
        else:
            users, items = [], []
            with open(path) as f:
                for line in f:
                    if line.startswith('begin'):
                        if line.strip() == 'begin-users':
                            nodes = users
                        elif line.strip() == 'begin-items':
                            nodes = items
                    else:
                        nodes.append(int(line.strip()))
        return users, items
    return None, None


def save_interactions(interactions, path, attributes=None, sep=' '):
    """
    Saves an interaction graph

    :param interactions: A pandas dataframe or networkx graph
    :param attributes: A list containing names of edge attributes
    :param path:
    :param sep:
    :return:
    """
    with open(path, 'w') as f:
        for i in range(len(interactions)):
            user, item = interactions[i][:2]
            data = ''    
            if attributes is not None:
                for j in range(len(attributes)):
                    data += f"{sep}{interactions[i][2][attributes[j]]}"
            f.write(f'{user}{sep}{item}{data}\n')
            
            
def save_embeddings(embeddings, path, binary, short_term=True):
    log(msg=f'Saving embeddings to {path}')
    if binary:
        np.savez(path, **embeddings)
    else:
        with open(path, 'w') as f:
            def write(k, k_emb):
                emb = ' '.join(str(val) for val in k_emb)
                f.write(f'{k} {emb}\n')
                
            for key in embeddings:
                if short_term:
                    for i in range(embeddings[key].shape[0]):
                        write(k=key, k_emb=embeddings[key][i])
                else:
                    write(k=key, k_emb=embeddings[key])

                
def populate_dict_of_list(dict_of_list, key, value):
    """
    used for building a dictionary of lists

    :param dict_of_list: The dictionary of list
    :param key: a key to the dictionary
    :param value: a value to be appended
    :return:
    """
    if key in dict_of_list:
        dict_of_list[key].append(value)
    else:
        dict_of_list[key] = [value]

                
def get_neighborhood_mask(neighborhood):
    neg_inf = -99999999.
    mask = (neighborhood != 0)
    mask = mask.astype(np.float32)
    mask[mask == 0] = neg_inf
    mask[mask != neg_inf] = 0
    return mask


def split_interactions(interactions, train_rate, dev_rate, temporal):
    if temporal:
        interactions.sort_values(TIME_ATR, inplace=True)
        edge_atr=[TIME_ATR] 
    else:
        interactions = interactions.sample(frac=1)
        edge_atr = None
    train_size = int(interactions.shape[0] * train_rate)
    dev_size = int(interactions.shape[0] * dev_rate)
    train_interactions = interactions[:train_size]
    dev_interactions = interactions[train_size: train_size + dev_size]
    test_interactions = interactions[train_size + dev_size:]
    log(f'Number of training interactions {train_interactions.shape[0]}')
    log(f'Number of validations interactions {dev_interactions.shape[0]}')
    log(f'Number of test interactions {test_interactions.shape[0]}')
    return train_interactions, dev_interactions, test_interactions, edge_atr


def random_other_than_this(this, dist, size=1):
    """
    Samples a random item other than this from a given distribution

    :param this:
    :param dist: The distribution (which can be, uniform, uni_gram, uni_gram_75)
    :size: The number of random samples
    :return:
    """
    samples = set()
    while len(samples) < size:
        rnd_ix = np.random.randint(0, len(dist))
        rnd_node = dist[rnd_ix]
        if rnd_node != this and rnd_node not in samples:
            samples.add(rnd_node)
    return list(samples)


def delta_decay(delta):
    return np.log(EPSILON + delta)


def as_numpy_array(tensor, device):
    if device == 'cpu':
        return tensor.data.numpy()
    return tensor.cpu().data.numpy()


def isin(container, obj):
    if isinstance(container, np.ndarray):
        return obj < container.shape[0]
    return obj in container


def expand_if(tensor, cond, dim=0): 
    """
    Expands a given tensor along a given dimension if a condition is satisfied.
    Otherwise returns the tensor as it is
    
    :param tensor: The tensor
    :param cond: The condition
    :param dim: The dimension along which the expansion is to be performed.
    :return: A tensor
    """
    return tensor.unsqueeze(dim) if cond else tensor


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
        
def create_root_sub_dirs(root_dir):
    model_dir = os.path.join(root_dir, MODEL_DIR)
    output_dir = os.path.join(root_dir, OUTPUT_DIR)
    create_dir(model_dir)
    create_dir(output_dir)


def compute_interaction_scores(interactions=None, user_embeddings=None, item_embeddings=None,
                               items_ug_75_dist=None, use_negatives=True):
    """
    Predicts the interaction score between users and items based on their embeddings
    
    """
    scores = []
    if interactions is None:
        dist = np.arange(user_embeddings.shape[0])
        for i in range(user_embeddings.shape[0]):
            rnd_ix = random_other_than_this(this=i, dist=dist)[0]
            user_emb = user_embeddings[i]
            item_emb = item_embeddings[i]
            rnd_emb = item_embeddings[rnd_ix]
            pos_score = np.dot(user_emb, item_emb.transpose())
            neg_score = np.dot(user_emb, rnd_emb.transpose())
            scores.append([pos_score, neg_score])
    else:
        for user, item in interactions:
            if user in user_embeddings and item in item_embeddings:
                user_emb = user_embeddings[user]
                item_emb = item_embeddings[item]
                neg_score = None
                if use_negatives:
                    rnd_item = random_other_than_this(this=item, dist=items_ug_75_dist)[0]
                    if rnd_item in item_embeddings:
                        rnd_emb = item_embeddings[rnd_item]
                        neg_score = user_emb.dot(rnd_emb.transpose()).max()
                    else:
                        continue
                pos_score = user_emb.dot(item_emb.transpose()).max()
                scores.append([pos_score] if neg_score is None else [pos_score, neg_score])
                neg_score = None
    scores = np.array(scores)
    return scores


def AUC(scores):
    hit = scores[:, 0] > scores[:, 1]
    tie = scores[:, 0] == scores[:, 1]
    hit_count = hit[hit].shape[0]
    tie_count = tie[tie].shape[0]
    return (hit_count + tie_count) / scores.shape[0]


def evaluate_nip(ground_truth, predictions):
    """
    Evaluates the quality of next item prediction results
    
    :param ground_truth: The ground truth next items
    :param predictions: The predicted next items
    :return the mean reciprocal rank (mrr) and recall at k (recall)
    """
    mrr = 0
    recall = 0
    for ix in range(len(ground_truth)):
        try:
            y = ground_truth[ix]
            y_hat = predictions[ix]
            pos = y_hat.index(y) + 1
            rr = 1. / pos
            mrr += rr
            recall += 1.
        except ValueError:
            pass
    recall /= len(ground_truth)
    mrr /= len(ground_truth)
    return mrr, recall


def get_device_id(cuda_is_available):
    if not cuda_is_available:
        return -1
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode('utf-8')
    gpu_stats = gpu_stats.strip().split('\n')
    stats = []
    for i in range(1, len(gpu_stats)):
        info = gpu_stats[i].split()
        used = int(info[0])
        free = int(info[2])
        stats.append([used, free])
    stats = np.array(stats)
    gpu_index = stats[:,1].argmax()
    available_mem_on_gpu = stats[gpu_index][1] - stats[gpu_index][0]
    return gpu_index if available_mem_on_gpu > 1000 else -1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./data/wikipedia/', 
                        help='A path to the root (dataset) directory')
    parser.add_argument('--binary', dest='binary', action='store_true', help='Whehter data is and should be stored in binary')
    parser.add_argument('--nbr-size', type=int, default=100, help='Neighborhood size')
    parser.add_argument('--dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--reg-cof', type=float, default=0.78, help='Regularization coefficient')
    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--temporal', dest='temporal', action='store_true', help='Interaction is temporal')
    parser.add_argument('--static', dest='temporal', action='store_false', help='Interaction is static')
    parser.add_argument('--log-level', type=int, default=0, help='Log level, values in [0, 4]')
    parser.set_defaults(binary=False)
    parser.set_defaults(temporal=False)
    return parser.parse_args()
    
