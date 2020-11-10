import torch.nn.functional as F
from torch import nn
import torch
import math


def get_encoder(enc_name=None):
    valid_names = {'rnn', 'cnn', 'base'}
    invalid_enc_msg = f"{enc_name} is invalid Encoder name. Valid encoder names are 'rnn', 'cnn', or 'base'"
    assert enc_name in valid_names or enc_name is None, invalid_enc_msg
    if enc_name == 'rnn':
        return RNNEncoder
    if enc_name == 'cnn':
        return CNNEncoder
    return BaseEncoder


def check_encoder_validity(temporal=True, enc_name='lstm'):
    import warnings
    if temporal:
        valid_encoders = {'rnn', 'cnn'}
        if enc_name not in valid_encoders:
            warnings.warn("You are using an/a {} encoder for a temporal interaction, "
                          "we recommend an/a {}, {}, or {}".format(get_encoder(enc_name),
                                                                   get_encoder('rnn'),
                                                                   get_encoder('cnn')))
    else:
        valid_encoders = {'base', None}
        if enc_name not in valid_encoders:
            warnings.warn("You are using {} encoder for non-temporal ineteraction, we recommend using "
                          "{} encoder".format(get_encoder(enc_name), get_encoder()))


class DeepRedModel(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.5, temporal=False, enc_name=None, **kwargs):
        super().__init__()
        self.temporal = temporal
        check_encoder_validity(temporal=temporal, enc_name=enc_name)
        encoder_model = get_encoder(enc_name=enc_name)
        self.encode = encoder_model(in_dim=in_dim, out_dim=out_dim, dropout=dropout, **kwargs)

    def forward(self, users, user_nh, user_mask, items, item_nh, item_mask, user_dt=None, user_nh_dt=None, 
                item_dt=None, item_nh_dt=None):
        self.user_emb = self.encode(users)
        self.user_nh_emb = self.encode(user_nh, delta_t=user_nh_dt, mask=user_mask)
        self.item_emb = self.encode(items)
        self.item_nh_emb = self.encode(item_nh, delta_t=item_nh_dt, mask=item_mask)
        
        self.user_item_alignment = torch.tanh(
            self.user_nh_emb.transpose(-1, -2).matmul(self.item_nh_emb))
        
        self.user_atn_coef = self.user_item_alignment.mean(dim=-1, keepdims=True)
        self.item_atn_coef = self.user_item_alignment.mean(dim=-2, keepdims=True).transpose(-1, -2)
        
        #self.user_atn_coef = self.user_item_alignment.max(dim=-1, keepdims=True).values
        #self.item_atn_coef = self.user_item_alignment.max(dim=-2, keepdims=True).values.transpose(-1, -2)
        
        self.user_atn_vec = torch.softmax(self.user_atn_coef + user_mask.unsqueeze(-1), dim=1)
        self.item_atn_vec = torch.softmax(self.item_atn_coef + item_mask.unsqueeze(-1), dim=1)

        self.user_rep = self.user_nh_emb.matmul(self.user_atn_vec).squeeze()
        self.item_rep = self.item_nh_emb.matmul(self.item_atn_vec).squeeze()


class BaseEncoder(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = out_dim
        self.feature_dim, self.seq_len_dim, self.batch_dim = -2, -1, 0
        self.embedding = nn.Embedding(in_dim + 1, out_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.__init_weight()

    def __init_weight(self):
        torch.manual_seed(7)
        nn.init.xavier_uniform_(self.embedding.weight)

    def embed(self, x):
        """
        :param x: A 1d [batch dimension] or 2d [batch dimension, sequence length dimension] tensor
        :return : An embedding of the input tensor
        for a 1d input the output is shaped as 
            [batch dimension, feature dimension]
        for a 2d input the output is shaped as 
            [batch dimension, feature dimension, sequence length dimension]
        """
        emb = self.dropout(self.embedding(x))
        return emb.transpose(-1, -2) if len(emb.shape) > 2 else emb

    def forward(self, x, **kwargs):
        return self.embed(x)


class RNNEncoder(BaseEncoder):

    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__(in_dim, out_dim, dropout)
        torch.manual_seed(7)
        self.rnn = nn.GRU(out_dim + 1, out_dim // 2, bidirectional=True)

    def forward(self, x, delta_t=None, **kwargs):
        x = self.embed(x)
        output_shape = x.shape
        if delta_t is not None:
            """x.shape = (batch_size, emb_dim, seq_len)
               delta_t.shape = (batch_size, seq_len)"""
            x = torch.cat([x, delta_t.unsqueeze(1)], dim=1)
            
        if len(output_shape) > 2:
            rnn_input_shape = x.shape[self.seq_len_dim], -1, x.shape[self.feature_dim]
            x, _ = self.rnn(x.reshape(rnn_input_shape))
            x = x.reshape(output_shape)
        return x
        

class CNNEncoder(BaseEncoder):

    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dropout=0.5):
        super().__init__(in_dim, out_dim, dropout)
        self.cnn = nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.embed(x)
        output_shape = x.shape
        if len(output_shape) > 2:
            cnn_input_shape = -1, output_shape[self.feature_dim], output_shape[self.seq_len_dim]
            x = self.cnn(x.reshape(cnn_input_shape)).reshape(output_shape)
        return x
