# -*- coding:utf-8 -*-
"""
Author: 
    Fengtong Xiao (fengtong.xiao@alibaba-inc.com),
    Lin Li (boolean.ll@alibaba-inc.com),
    Weinan Xu (stella.xu@lazada.com)
Slightly mofification based on original DIN model to fit multi-GPU usage for DMBGN work.
References:
    DMBGN: Deep Multi-Behavior Graph Networks for Voucher Redemption Rate Prediction
"""

"""
Original Author:
    Yuef Zhang
Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""

import torch.nn as nn
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import *
from deepctr_torch.layers import *

from .util import HistAttentionSeqPoolingLayer

class DIN(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list, to indicate sequence sparse field
    :param target_emb_dim_aft: int, to indicate number of embedding dimension corresponding to post-collection phase, which need s to be removed from targeted current voucher as it includes the targeted voucher label information
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: 'cpu' or 'torch:cuda', training device of CPU/GPU
    
    :return:  A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, target_emb_dim_aft=0,
                 dnn_use_bn=False, dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, 
                 l2_reg_dnn=0.0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu'):
        super(DIN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device)
        
        dnn_feature_columns_new = list()
        for feat in dnn_feature_columns: # remove keys_length from dense feature list
            if feat.name != 'keys_length':
                dnn_feature_columns_new.append(feat)
                
        self.dnn_feature_columns= dnn_feature_columns_new      
        self.sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)
                
        att_emb_dim = self._compute_interest_dim()
        self.emb_dim_here=att_emb_dim

        self.attention = HistAttentionSeqPoolingLayer(att_hidden_units=att_hidden_size,
                                                      embedding_dim=att_emb_dim,
                                                      target_emb_dim_aft=target_emb_dim_aft,
                                                      att_activation=att_activation,
                                                      return_score=False,
                                                      supports_masking=False,
                                                      weight_normalization=att_weight_normalization)
        
        self.target_emb_dim_aft=target_emb_dim_aft

        self.dnn = DNN(inputs_dim=int(self.compute_input_dim(self.dnn_feature_columns) - target_emb_dim_aft),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.device = device
        self.to(device)

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,)).to(self.device)
        for weight_list, weight_decay, p in self.regularization_weight:
            weight_reg_loss = torch.zeros((1,)).to(self.device)
            for w in weight_list:
                if isinstance(w, tuple):
                    l2_reg = torch.norm(w[1], p=p, )
                else:
                    l2_reg = torch.norm(w, p=p, )
                weight_reg_loss = weight_reg_loss + l2_reg.to(self.device)
            reg_loss = weight_decay * weight_reg_loss
            total_reg_loss += reg_loss
        return total_reg_loss
    
    def get_column_from_input(self, X, feat_name) :
        return X[:, self.feature_index[feat_name][0]:self.feature_index[feat_name][1]].long()
        
        
    def forward(self, X, step = 0):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        
        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          self.history_feature_list, self.history_feature_list, to_list=True)
            
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         self.history_fc_names, self.history_fc_names, to_list=True)
        
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns, mask_feat_list=self.history_feature_list, to_list=True)
        
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)
        
        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)
        
        dnn_input_emb_list += sequence_embed_list
        
        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)             # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)               # [B, T, E]
        
        keys_length = self.get_column_from_input(X,'keys_length') # [B, 1]
        
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)
        hist = self.attention(query_emb, keys_emb, keys_length)   # [B, 1, E]
            
        # deep part
        deep_input_emb = torch.cat((hist,deep_input_emb), dim=-1) # [B, 1, D]
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1) #[B, D]

        if self.target_emb_dim_aft > 0:
            deep_input_emb = deep_input_emb[:,:-self.target_emb_dim_aft]
        
        dnn_input = combined_dnn_input([deep_input_emb],dense_value_list)
        
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        
        y_pred = self.out(dnn_logit)
        
        reg_loss = self.get_regularization_loss()
        aux_loss = self.aux_loss
        return y_pred, reg_loss, aux_loss

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim  
