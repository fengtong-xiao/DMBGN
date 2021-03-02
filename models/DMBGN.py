# -*- coding:utf-8 -*-

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import *
from deepctr_torch.layers import *

from .util import HistAttentionSeqPoolingLayer

default_gnn_dim = 16


class DMBGN(BaseModel):
    """ Instantiates the DMBGN network

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list, to indicate sequence sparse field
    :param target_emb_dim_aft: int, to indicate number of embedding dimension corresponding to post-collection phase,
    which need s to be removed from targeted current voucher as it includes the targeted voucher label information
    :param sequence_size: int, historical UVG sequence size
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: 'cpu' or 'torch:cuda', training device of CPU/GPU
    :param gnet_tune: boolean, whether finetune the DMBGN UVG network parameters, if true, the following parameters are
    required: gnet, gnet_before
    :param hist_gnn_dropout: dropout rate for UVG training, the probability of UVG for not being trained, this is used
    to save training time as well as prevent overfitting
    :param gnet: pretrained gnet for historical UVG sequence
    :param gnet_before: pretrained gnet for historical UVG sequence which only takes 'bef' user behaviors, used for
    target UVG network
    :param hash_sid_uvg_graphs_dic: dictionary, key = hashed session id, value = UVG

    :return:  A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, history_feature_list, target_emb_dim_aft=0, sequence_size=0,
                 dnn_use_bn=False, dnn_hidden_units=(256, 128), dnn_activation='relu', dnn_dropout=0,
                 att_hidden_size=(64, 16), att_activation='Dice', att_weight_normalization=False,
                 l2_reg_dnn=0.0, l2_reg_embedding=1e-6, init_std=0.0001,
                 seed=1024, task='binary', device='cpu',
                 gnet_tune=False, hist_gnn_dropout=0.6, gnet=None, gnet_before=None, hash_sid_uvg_graphs_dic=None):
        super(DMBGN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                    init_std=init_std, seed=seed, task=task, device=device)

        dnn_feature_columns_new = list()
        for feat in dnn_feature_columns:  # remove keys_length from dense feature list
            if feat.name != 'keys_length':
                dnn_feature_columns_new.append(feat)
        self.dnn_feature_columns = dnn_feature_columns_new
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

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
        self.emb_dim_here = att_emb_dim

        self.attention = HistAttentionSeqPoolingLayer(att_hidden_units=att_hidden_size,
                                                      embedding_dim=att_emb_dim,
                                                      target_emb_dim_aft=target_emb_dim_aft,
                                                      att_activation=att_activation,
                                                      return_score=False,
                                                      supports_masking=False,
                                                      weight_normalization=att_weight_normalization)
        self.target_emb_dim_aft = target_emb_dim_aft

        self.dnn = DNN(
            inputs_dim=int(self.compute_input_dim(self.dnn_feature_columns) - target_emb_dim_aft + sequence_size + 3),
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            l2_reg=l2_reg_dnn,
            use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

        if gnet_tune:
            if gnet is None or gnet_before is None:
                raise Exception("Fine tune GNN version, but gnet or gnet_before is None, please check..")
        self.gnet_tune = gnet_tune
        self.hist_gnn_dropout = hist_gnn_dropout
        self.gnet = gnet
        self.gnet_before = gnet_before
        self.hash_sid_uvg_graphs_dic = hash_sid_uvg_graphs_dic
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

    def get_column_from_input(self, X, feat_name):
        """ get the feature value from input dataframe, based on the generated feature_index

        :param X: input dataframe containing all feature values
        :param feat_name: string, feature name to be retrieved

        :return: corresponding feature values
        """
        return X[:, self.feature_index[feat_name][0]:self.feature_index[feat_name][1]].long()

    def set_hash_id(self, dat, column, hash_size):
        """ hash feature columns

        :param X: input dataframe containing all feature values
        :param feat_name: string, feature name to be retrieved

        :return: corresponding feature values
        """
        cols = self.feature_index[column]
        dat[:, cols[0]:cols[1]] = dat[:, cols[0]:cols[1]] % hash_size

    def get_gnn_emb(self, session_id, gnet, gprefix=['b_', 'a_'], gactions=['atc', 'ord']):
        """ get UVG embedding from UVG network

        :param session_id: session id
        :param gnet: pretrained UVG network, could be gnet or gnet_before
        :param gprefix: list of prefix string, i.e. ['b_', 'a_']
        :param gactions: list of action type string, i.e. ['atc_', 'ord_'], the value should match emb_dict keys

        :return: tuple(s_{UVG} UVG Score, original promotion id, promotion embedding, original session id,
        session id embedding, boolean value indicating whether the UVG exists)
        """
        graph = None
        if session_id in self.hash_sid_uvg_graphs_dic.keys():
            gg = self.hash_sid_uvg_graphs_dic[session_id]
            graph = {}
            for p in gprefix:
                for a in gactions:
                    k = p + a
                    if k in gg.keys():
                        graph[k] = gg[k]
            if len(graph) > 0:
                gnn_score, raw_promotion_id, promotion_emb, raw_session_id, sess_emb = gnet(graph)
                return gnn_score, raw_promotion_id, promotion_emb.unsqueeze(0).unsqueeze(0), sess_emb.unsqueeze(
                    0).unsqueeze(0), True

        sess_emb = torch.zeros(default_gnn_dim).unsqueeze(0).unsqueeze(0).to(self.device)
        promotion_emb = torch.zeros(default_gnn_dim).unsqueeze(0).unsqueeze(0).to(self.device)
        return torch.tensor(0).to(self.device), torch.tensor(0).to(self.device), promotion_emb, sess_emb, False

    def forward(self, X, step=0):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          self.history_feature_list, self.history_feature_list, to_list=True)

        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         self.history_fc_names, self.history_fc_names, to_list=True)
        keys_length = self.get_column_from_input(X, 'keys_length')

        if self.gnet_tune:
            hist_session_ids = self.get_column_from_input(X, 'hist_sid').cpu().detach().numpy()
            target_session_ids = self.get_column_from_input(X, 'session_id').cpu().detach().numpy()

            if ((step % 10) / 10 < (1 - self.hist_gnn_dropout)) is True:
                target_sess_emb_agg_all = None
                target_promotion_emb_agg_all = None
                for sids in target_session_ids:
                    target_sess_emb_agg = None
                    target_promotion_emb_agg = None
                    for session_id in sids:
                        gnn_score, raw_promotion_id, promotion_emb, sess_emb, flag = self.get_gnn_emb(session_id,
                                                                                                      gnet=self.gnet_before,
                                                                                                      gprefix=['b_'])
                        target_sess_emb_agg = sess_emb if target_sess_emb_agg is None else torch.cat(
                            [target_sess_emb_agg, sess_emb], dim=1)
                        target_promotion_emb_agg = promotion_emb if target_promotion_emb_agg is None else torch.cat(
                            [target_promotion_emb_agg, promotion_emb], dim=1)

                    target_sess_emb_agg_all = target_sess_emb_agg if target_sess_emb_agg_all is None else torch.cat(
                        [target_sess_emb_agg_all, target_sess_emb_agg], dim=0)
                    target_promotion_emb_agg_all = target_promotion_emb_agg if target_promotion_emb_agg_all is None else torch.cat(
                        [target_promotion_emb_agg_all, target_promotion_emb_agg], dim=0)

                for ii in range(len(self.history_feature_list)):
                    target_feat = self.sparse_feature_columns[ii].name
                    if target_feat == 'promotion_id':
                        query_emb_list[ii] = target_promotion_emb_agg_all
                    elif target_feat == 'sid':
                        query_emb_list[ii] = target_sess_emb_agg_all

                hist_sess_emb_agg_all = None
                hist_promotion_emb_agg_all = None
                for sids in hist_session_ids:
                    hist_sess_emb_agg = None
                    hist_promotion_emb_agg = None
                    for session_id in sids:
                        gnn_score, raw_promotion_id, promotion_emb, sess_emb, flag = self.get_gnn_emb(session_id,
                                                                                                      gnet=self.gnet)
                        hist_sess_emb_agg = sess_emb if hist_sess_emb_agg is None else torch.cat(
                            [hist_sess_emb_agg, sess_emb], dim=1)
                        hist_promotion_emb_agg = promotion_emb if hist_promotion_emb_agg is None else torch.cat(
                            [hist_promotion_emb_agg, promotion_emb], dim=1)

                    hist_sess_emb_agg_all = hist_sess_emb_agg if hist_sess_emb_agg_all is None else torch.cat(
                        [hist_sess_emb_agg_all, hist_sess_emb_agg], dim=0)
                    hist_promotion_emb_agg_all = hist_promotion_emb_agg if hist_promotion_emb_agg_all is None else torch.cat(
                        [hist_promotion_emb_agg_all, hist_promotion_emb_agg], dim=0)

                for ii in range(len(self.history_feature_columns)):
                    hist_feat = self.history_feature_columns[ii].name
                    if hist_feat == 'hist_promotion_id':
                        keys_emb_list[ii] = hist_promotion_emb_agg_all
                    elif hist_feat == 'hist_sid':
                        keys_emb_list[ii] = hist_sess_emb_agg_all

        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              mask_feat_list=self.history_feature_list, to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)  # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)  # [B, T, E]

        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)
        hist = self.attention(query_emb, keys_emb, keys_length)  # [B, 1, E]

        # hist embedding score
        if self.target_emb_dim_aft > 0:
            keys_emb_list_sid = keys_emb_list[1][:, :, :-self.target_emb_dim_aft].unsqueeze(3)
        else:
            keys_emb_list_sid = keys_emb_list[1].unsqueeze(3)

        hist_score = torch.matmul(keys_emb_list[0].unsqueeze(2), keys_emb_list_sid).squeeze(1).squeeze(2)  # [B, T, 1]
        hist_score = hist_score.view(hist_score.size(0), -1)  # [B, T]
        hist_score = torch.sigmoid(hist_score)

        batch_size, max_length, dim = keys_emb.size()
        keys_masks = torch.arange(max_length, device=self.device, dtype=keys_length.dtype).repeat(batch_size,
                                                                                                  1)  # [B, T]
        keys_masks = keys_masks < keys_length.view(-1, 1)  # [B, T] True False mask

        paddings_zero = torch.zeros_like(hist_score, device=self.device)

        hist_score = torch.where(keys_masks, hist_score, paddings_zero)  # [B, T]

        hist_score_max, _ = torch.max(hist_score, dim=1)
        hist_score_max = hist_score_max.unsqueeze(1)

        hist_score_avg = torch.mean(hist_score, dim=1)
        hist_score_avg = hist_score_avg.unsqueeze(1)

        hist_score_min, _ = torch.min(hist_score, dim=1)
        hist_score_min = hist_score_min.unsqueeze(1)

        # deep part
        deep_input_emb = torch.cat((hist, deep_input_emb), dim=-1)  # [B, 1, D]
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)  # [B, D]

        if self.target_emb_dim_aft > 0:
            deep_input_emb = deep_input_emb[:, :-self.target_emb_dim_aft]

        deep_input_emb = torch.cat((deep_input_emb, hist_score), dim=-1)
        deep_input_emb = torch.cat((deep_input_emb, hist_score_max), dim=-1)
        deep_input_emb = torch.cat((deep_input_emb, hist_score_avg), dim=-1)
        deep_input_emb = torch.cat((deep_input_emb, hist_score_min), dim=-1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)

        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        reg_loss = self.get_regularization_loss()
        aux_loss = 1.0 / torch.mean(hist_score_avg, dim=0)
        return y_pred, reg_loss, aux_loss

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim