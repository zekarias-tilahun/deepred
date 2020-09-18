from itertools import chain
from model import DeepRedModel
from data import load_data
from utils import *

import numpy as np
import datetime as dt

import torch

import os

torch.manual_seed(7)
np.random.seed(7)


class DeepRed:

    def __init__(self, num_nodes, dim=128, dropout=0.5, temporal=True, lr=0.0001, epochs=1, reg_cof=0.0001, root_dir=None):
        self.num_nodes = num_nodes
        self.dim = dim
        self.dropout = dropout
        self.temporal = temporal
        self.lr = lr
        self.epochs = epochs
        self.reg_cof = reg_cof
        self.root_dir = root_dir
        self.set_device()
        self.model = DeepRedModel(in_dim=self.num_nodes, out_dim=self.dim, dropout=self.dropout,
                               temporal=self.temporal, enc_name='rnn' if temporal else None)
        self.model.to(self.device)
        
    def set_device(self):
        device_id = get_device_id(torch.cuda.is_available())
        self.device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

    def compute_loss(self):
        """
        Computes the loss on the current state of a model

        :return: The loss
        """

        """Model loss"""
        mse_loss = torch.nn.MSELoss()
        interaction_loss = mse_loss(self.model.user_rep.squeeze(), self.model.item_rep.squeeze())

        """Regularization loss"""
        x = torch.cat([self.model.user_rep.squeeze(), self.model.item_rep.squeeze()])
        x = x.matmul(x.transpose(0, 1))
        p = self.reg_cof * (x - torch.eye(x.shape[0]).to(self.device)).norm()

        """Combined loss"""
        return interaction_loss + p

    def fit_batch(self, batch):
        """
        Invokes the forward execution of the model on a given batch

        :param batch: The given batch
        :return:
        """
        inputs = {
            'users': batch.users.to(self.device),
            'user_nh': batch.user_nh.to(self.device),
            'user_mask': batch.user_mask.to(self.device),
            'items': batch.items.to(self.device),
            'item_nh': batch.item_nh.to(self.device),
            'item_mask': batch.item_mask.to(self.device),
        }
        if self.temporal:
            inputs['user_nh_dt'] = batch.user_nh_dt.to(self.device)
            inputs['item_nh_dt'] = batch.item_nh_dt.to(self.device)
        self.model(**inputs)

    def validate(self, dev_dataset, scoring_fun, eval_fun):
        """
        Validates the model on a dev set.

        :param: dev_dataset: The dev set
        :param: scoring_fun: A function to prediction the interaction scores between users and items
        :param: eval_fun: A function to evaluate the quality of the predicted scores
        :return: A dictionary of the expected prediction score and loss along with the std
        """
        losses = []
        predictions = []

        for batch in dev_dataset.batches():
            self.fit_batch(batch=batch)
            error = self.compute_loss()
            losses.append(as_numpy_array(error.data, self.device))
            prediction_scores = scoring_fun(user_embeddings=as_numpy_array(self.model.user_rep, self.device),
                                            item_embeddings=as_numpy_array(self.model.item_rep, self.device))
            predictions.append(eval_fun(prediction_scores))

        result = {'pred': np.mean(predictions), 'pred_std': np.std(predictions),
                  'loss': np.mean(losses), 'loss_std': np.std(losses)}
        return result

    def fit(self, dataset, dev_dataset=None):
        """
        Trains the model. It is trained using MSE loss, and the expected MSE and AUCROC are used on 
        the batched from the dev set to check the quality of predicting interaction links between 
        users and items in the set and to analyse and control overfitting conditions.

        :param dataset: The training set
        :param dev_dataset: The dev set
        :return:
        """
        log("Training ...")
        metric = 'AUC'
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        total_batches = len(dataset.partitions)
        for epoch in range(self.epochs):
            start_time = dt.datetime.now()
            train_loss = 0.
            for bc, b in enumerate(dataset.batches()):
                self.fit_batch(batch=b)
                error = self.compute_loss()
                optimizer.zero_grad()
                error.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.6)
                optimizer.step()
                msg = 'Epoch {}/{} current batch {}/{} training loss {:.6f}'.format(
                    epoch + 1, self.epochs, bc, total_batches, error.data)
                log(msg=msg, cr=True, caller_level=BATCH_LOG)

            msg = ''
            if dev_dataset is not None:
                val_res = self.validate(
                    dev_dataset=dev_dataset, scoring_fun=compute_interaction_scores, eval_fun=AUC)
                msg += 'expected validation loss {:.10f} {} {:.4f} std {:.4f} '.format(
                    val_res['loss'], metric, val_res['pred'], val_res['pred_std'])
            end_time = dt.datetime.now()
            seconds = (end_time - start_time).seconds
            log(msg='{}time {} seconds'.format(msg, seconds), prefix=None, caller_level=EPOCH_LOG)
            self.save_model(epoch + 1)

    def save_model(self, epoch=None):
        file_name = "deepred.mdl" if epoch is None else f"deepred_epoch_{epoch}.pt"
        path = os.path.join(self.root_dir, MODEL_DIR, file_name)
        log(f"Saving the current state of the model to {path}")
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        log(f"Loading model from {path}")
        #self.model = model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def transform(self, batches, total_batches=None):
        """
        Infer the embeddings of users and items based on the current state of the model

        :param batches: A batch of data
        :return: static and dynamic-embeddings of users and items
        """
        log('Inferring user and item long and short term embeddings ...')
        long_term_user_emb = {}
        short_term_user_emb = {}
        long_term_item_emb = {}
        short_term_item_emb = {}
        for bc, b in enumerate(batches):
            self.fit_batch(batch=b)
            users, items = b.users, b.items
            for i in range(users.shape[0]):
                user = int(as_numpy_array(users[i], self.device))
                item = int(as_numpy_array(items[i], self.device))
                populate_dict_of_list(short_term_user_emb, user,
                                      as_numpy_array(self.model.user_rep[i], self.device))
                populate_dict_of_list(short_term_item_emb, item,
                                      as_numpy_array(self.model.item_rep[i], self.device))
                long_term_user_emb[user] = as_numpy_array(self.model.user_emb[i], self.device)
                long_term_item_emb[item] = as_numpy_array(self.model.item_emb[i], self.device)
            if total_batches is not None:
                log(f"Batch {bc + 1}/{total_batches}", cr=True, prefix=PROG)
        log("")

        def embedding_as_type(embedding, as_type=np.array):
            for node in embedding:
                embedding[node] = as_type(embedding[node])

        embedding_as_type(short_term_user_emb)
        embedding_as_type(short_term_item_emb)

        return (long_term_user_emb, short_term_user_emb,
                long_term_item_emb, short_term_item_emb)


def main(args):
    global LOG_LEVEL
    LOG_LEVEL = args.log_level
    train_dataset, dev_dataset = load_data(args)
    deepred = DeepRed(num_nodes=train_dataset.num_nodes, dim=args.dim, dropout=args.dropout,
                temporal=args.temporal, lr=args.lr, reg_cof=args.reg_cof, 
                epochs=args.epochs, root_dir=args.root_dir)
    deepred.fit(dataset=train_dataset, dev_dataset=dev_dataset)


if __name__ == '__main__':
    main(parse_args())
