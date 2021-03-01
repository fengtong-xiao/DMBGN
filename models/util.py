# -*- coding:utf-8 -*-
"""
Author: 
    Fengtong Xiao (fengtong.xiao@alibaba-inc.com),
    Lin Li (boolean.ll@alibaba-inc.com),
    Weinan Xu (stella.xu@lazada.com)
References:
    DMBGN: Deep Multi-Behavior Graph Networks for Voucher Redemption Rate Prediction
"""


from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import *
from deepctr_torch.layers import *

class AttentionUnit(nn.Module):
    """
        The LocalActivationUnit used in DIN with which the representation of
        user interests varies adaptively given different candidate items.

    Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

    Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

    Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """
    def __init__(self, 
                 hidden_units=(64, 32), 
                 sparse_feature_nums=0,
                 embedding_dim=4, 
                 activation='sigmoid', 
                 dropout_rate=0, 
                 dice_dim=3, 
                 l2_reg=0, 
                 use_bn=False):
        super(AttentionUnit, self).__init__()

        self.dnn = DNN(inputs_dim=4*embedding_dim,
                        hidden_units=hidden_units,
                        activation=activation,
                        l2_reg=l2_reg,
                        dropout_rate=dropout_rate,
                        dice_dim=dice_dim,
                        use_bn=use_bn,
                        init_std=1)

        self.dense = nn.Linear(hidden_units[-1], 1)
        self.embedding_dimension=embedding_dim
        

    def forward(self, query, user_behavior):
        # query ad       # [B, 1, E]
        # user behavior  # [B, T, E]
        user_behavior_len = user_behavior.size(1)
        queries = query.expand(-1, user_behavior_len, -1) # [B, T, E]
        
        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior], dim=-1)
        
        attention_output = self.dnn(attention_input)
        attention_score = self.dense(attention_output)    # [B, T, 1]

        return attention_score

class HistAttentionSeqPoolingLayer(nn.Module):
    """
        Slightly mofification based on original DIN model to fit multi-GPU usage for DMBGN work.
        The Attentional sequence pooling operation used in DIN & DIEN.

        Arguments
          - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.

          - **att_activation**: Activation function to use in attention net.

          - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.

          - **supports_masking**:If True,the input need to support masking.

        References
          - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
      """
    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', 
                 weight_normalization=False,
                 return_score=False, 
                 supports_masking=False, 
                 embedding_dim=4,
                 target_emb_dim_aft=0,
                 **kwargs):
        super(HistAttentionSeqPoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking
        self.target_emb_dim_aft = target_emb_dim_aft

        self.local_att = AttentionUnit(hidden_units=att_hidden_units, 
                                       embedding_dim=embedding_dim - target_emb_dim_aft,
                                       activation=att_activation,
                                       dropout_rate=0, use_bn=False)

    def forward(self, query, keys, keys_length, mask=None):
        batch_size, max_length, dim = keys.size()

        # Mask
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True,input must support masking")
            keys_masks = mask.unsqueeze(1)
        else:
            keys_masks = torch.arange(max_length, device=keys_length.device, dtype=keys_length.dtype).repeat(batch_size, 1)  # [B, T]
            keys_masks = keys_masks < keys_length.view(-1, 1)  # 0, 1 mask
            keys_masks = keys_masks.unsqueeze(1)  # [B, 1, T]

        if self.target_emb_dim_aft > 0:
            # remove the aft emb part from the end of input
            attention_score = self.local_att(query[:,:,:-self.target_emb_dim_aft], 
                                             keys[:,:,:-self.target_emb_dim_aft])  # [B, T, 1]
        else:
            attention_score = self.local_att(query, keys)  # [B, T, 1]
        
        outputs = torch.transpose(attention_score, 1, 2)   # [B, 1, T]

        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = torch.zeros_like(outputs)
       
        outputs = torch.where(keys_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        # outputs = outputs / (keys.shape[-1] ** 0.05)

        if self.weight_normalization:
            outputs = F.softmax(outputs, dim=-1)  # [B, 1, T]
            print("attention outputs aft normalize ", outputs)
        if not self.return_score:
            # Weighted sum
            outputs = torch.matmul(outputs, keys)  # [B, 1, E]

        return outputs

import time
import numpy as np
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def hash_func(x, size) :
    """Calculate hashed value based on dictionary size
    
    :param x: input value
    :param size: dictionary size
    
    :return: hashed value based on dictionary size
    
    """
    return x % size

def init_emb(emb_dict, emb_size=16, hash_factor=1, lbe=None, 
             sparse=True, requires_grad=False, cast_to_str=False, emb_index=None):
    """ Initialize embedding based on value from dictionary
    
    :param emb_dict: embedding dictionary, key=id, value=np.array
    :param emb_size: embedding dimension
    :param hash_factor: hash factor of output embedding
    :param lbe: LabelEncoderExt class, non-empty
    :param sparse: whether returned embedding is a sparse tensor
    :param requires_grad: whether returned embedding requires back propagation
    :param cast_to_str: whether cast dictionary id to string
    :param emb_index: index from input value if dict value is a tuple i.e. value=(id2, np.array)
    
    :return: embedding tensor
    
    """
    if lbe == None: raise Exception("Encoder is empty")
        
    if cast_to_str:
        indices = lbe.transform(np.array(list(emb_dict.keys())).astype(str))
    else:
        indices = lbe.transform(emb_dict.keys())

    session_size = int(max(len(emb_dict), len(lbe.classes_)) * hash_factor)
    ts_emb = torch.rand(session_size, emb_size, dtype = torch.float)
    for i, (key, emb) in tqdm(enumerate(emb_dict.items())):
        if emb_index == None:
            ts_emb[indices[i]] = torch.FloatTensor(emb)
        else:
            ts_emb[indices[i]] = torch.FloatTensor(emb[emb_index])
    emb_ts = torch.nn.Embedding.from_pretrained(ts_emb, sparse=sparse)
    emb_ts.weight.requires_grad = requires_grad
    return emb_ts

def gen_dmbgn_input_data(feature_names, raw_features, target, label_encoder={}, sequence_size=6, 
                         sparse_feature=[], hist_list_features=[]):
    """ Process DMBGN input data
    
    :param feature_names: list of all feature names (string)
    :param raw_features: input dataframe
    :param target: label column name, string
    :param label_encoder: dictionary of LabelEncoderExt for each sparse feature
    :param sequence_size: historical UVG sequence length
    :param sparse_feature: list sparse features names (string)
    :param hist_list_features: list historical features names (string)
    
    :return: tuple of processed data, (y, x) representing label and input features
    
    """
    model_input = {}
    for name in feature_names :
        if name in sparse_feature :
            model_input[name] = raw_features[name].values
        elif name == 'keys_length' :
            model_input[name] = raw_features[name].fillna(0).astype(np.float32).values
        elif name in hist_list_features :
            np_a = raw_features[name].fillna(0).values.copy()
            print('handling hist_list_features Feature: ' + name) 
            for i in tqdm(range(np_a.shape[0])):
                if np_a[i] == 0:
                    np_a[i] = np.zeros(sequence_size, dtype=np.int32)
                else:
                    np_a[i] = np.array(np_a[i].split(","))
                    if np_a[i].shape[0] >= sequence_size:
                        np_a[i] = np_a[i][-sequence_size:]
                    else:
                        np_a[i] = np.concatenate([np_a[i], np.zeros(sequence_size - np_a[i].shape[0], dtype=np.int32)], axis=-1)
            a = np.vstack(np_a)
            res = np.zeros_like(a, dtype=np.int32)
            le_dict = dict(zip(label_encoder[name[5:]].classes_,
                               label_encoder[name[5:]].transform(label_encoder[name[5:]].classes_))) # removed 'hist_'
            mmax = 0
            for i in range(len(a)) :
                for j in range(len(a[i])) :
                    res[i][j] = int(le_dict.get(str(a[i][j]), 0))
                    mmax = max(res[i][j], mmax)
            model_input[name] = res
        else :
            model_input[name] = raw_features[name].fillna(0).astype(np.float32).values
    return raw_features[target].values, model_input

def predict(model, x, batch_size=256, use_double=False, feature_index=None, device='cpu'):
    """ Modified predict method compatible of multi-GPU usage,
        get model predicted value based on input feature value x

    :param model: The trained model, a PyTorch model instance.
    :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
    :param batch_size: Integer. If unspecified, it will default to 256.
    :param use_double: Boolean, whether convert the predicted output to double type
    :param feature_index: OrderedDIct, output from baseModel.build_input_features method. representing feature column start and end indexes from the input dataframe {feature_name:(start, start+dimension)}
    :param device: 'cpu' or 'torch:cuda', training device of CPU/GPU
    
    :return: Numpy array(s) of predictions.
    """
    model = model.eval()
    if isinstance(x, dict):
        x = [x[feature] for feature in feature_index]
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    tensor_data = Data.TensorDataset(
        torch.from_numpy(np.concatenate(x, axis=-1)))
    test_loader = DataLoader(
        dataset=tensor_data, shuffle=False, batch_size=batch_size)

    pred_ans = []
    with torch.no_grad():
        for index, x_test in enumerate(test_loader):
            x = x_test[0].to(device).float()
            # y = y_test.to(self.device).float()

            y_pred, reg_loss, aux_loss = model(x)
            y_pred = y_pred.cpu().data.numpy()
            pred_ans.append(y_pred)

    if use_double:
        return np.concatenate(pred_ans).astype("float64")
    else:
        return np.concatenate(pred_ans)


def evaluate(model, metrics, x, y, batch_size=256, device='cpu'):
    """ Modified evaluate method compatible of multi-GPU usage.

    :param model: The trained model, a PyTorch model instance.
    :param metrics: dictionary, key=metric name, value = corresponding calculation class. Field from output of baseModel._get_metrics method, representing evaluation metrics
    :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
    :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
    :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
    :param device: 'cpu' or 'torch:cuda', training device of CPU/GPU
    
    :return: 
    """
    pred_ans = predict(model, x, batch_size, device=device)
    eval_result = {}
    for name, metric_fun in metrics.items():
        eval_result[name] = metric_fun(y, pred_ans)
    return eval_result, pred_ans


def fit(model, feature_index, optim, metrics, loss_func, x=None, y=None, batch_size=None, epochs=1, verbose=1,
        retain_graph=True, initial_epoch=0, 
        validation_split=0., validation_data=None, shuffle=True, callbacks=None, writer=None, 
        device='cpu', device_count=0):
    """ Modified evaluate method compatible of multi-GPU usage.
    
    :param model: The trained model, a PyTorch model instance.
    :param feature_index: OrderedDIct, field from output of baseModel.build_input_features method. representing feature column start and end indexes from the input dataframe {feature_name:(start, start+dimension)}
    :param optim: torch.optim class, field from output of baseModel._get_optim method. representing optimizer for model training
    :param metrics: dictionary, key=metric name, value = corresponding calculation class. Field from output of baseModel._get_metrics method, representing evaluation metrics
    :param loss_func: torch.nn.functional class. Field from output of baseModel._get_loss_func method, representing loss function for model training
    
    :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
        dictionary mapping input names to Numpy arrays.
    :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
    :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
    :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
    :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    :param retain_graph: Boolean. whether retain graph for loss.backward() operation
    :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
    :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
    :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
    :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
    :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`
    :param writer: Tensorboard writer, if needed
    """

    if isinstance(x, dict):
        x = [x[feature] for feature in feature_index]

    do_validation = False
    validation_result = {}
    pred_y = None
    if validation_data:
        do_validation = True
        if len(validation_data) == 2:
            val_x, val_y = validation_data
            val_sample_weight = None
        elif len(validation_data) == 3:
            val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
        else:
            raise ValueError(
                'When passing a `validation_data` argument, '
                'it must contain either 2 items (x_val, y_val), '
                'or 3 items (x_val, y_val, val_sample_weights), '
                'or alternatively it could be a dataset or a '
                'dataset or a dataset iterator. '
                'However we received `validation_data=%s`' % validation_data)
        if isinstance(val_x, dict):
            val_x = [val_x[feature] for feature in feature_index]

    elif validation_split and 0. < validation_split < 1.:
        do_validation = True
        if hasattr(x[0], 'shape'):
            split_at = int(x[0].shape[0] * (1. - validation_split))
        else:
            split_at = int(len(x[0]) * (1. - validation_split))
        x, val_x = (slice_arrays(x, 0, split_at),
                    slice_arrays(x, split_at))
        y, val_y = (slice_arrays(y, 0, split_at),
                    slice_arrays(y, split_at))

    else:
        val_x = []
        val_y = []
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    train_tensor_data = Data.TensorDataset(
        torch.from_numpy(np.concatenate(x, axis=-1)),
        torch.from_numpy(y))
    if batch_size is None:
        batch_size = 256

    train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

    model = model.train()

    sample_num = len(train_tensor_data)
    steps_per_epoch = (sample_num - 1) // batch_size + 1

    # Train
    print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
        len(train_tensor_data), len(val_y), steps_per_epoch))
    gstep = 0
    for epoch in range(initial_epoch, epochs):
        epoch_logs = {}
        start_time = time.time()
        loss_epoch = 0
        total_loss_epoch = 0
        train_result = {}
        try:
            with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                for index, (x_train, y_train) in t:
                    x = x_train.float()
                    y = y_train.to(device).float()
                    y_pred, reg_loss, aux_loss = model(x, gstep)
                    y_pred = y_pred.squeeze()
                    optim.zero_grad()
                    loss = loss_func(y_pred, y.squeeze(), reduction='sum')

                    if device_count <= 1 :
                        total_loss = loss + reg_loss + aux_loss  
                    else:
                        total_loss = loss
                        for iii in range(device_count) :
                            total_loss += reg_loss[iii] + aux_loss[iii]

                    loss_epoch += loss.item()
                    total_loss_epoch += total_loss.item()
                    total_loss.backward(retain_graph=retain_graph)
                    optim.step()
                    if verbose > 0:
                        gstep += 1
                        for name, metric_fun in metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            metric = metric_fun(y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"))
                            train_result[name].append(metric)
                            if writer is not None: writer.add_scalar(name, metric, global_step=gstep)

        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

        # Add epoch_logs
        epoch_logs["loss"] = total_loss_epoch / sample_num
        for name, result in train_result.items():
            epoch_logs[name] = np.sum(result) / steps_per_epoch
            
        if do_validation:
            eval_result, pred_ans = evaluate(model, metrics, val_x, val_y, batch_size, device=device)
            for name, result in eval_result.items():
                key = "eval_" + name
                update = False
                if key not in validation_result.keys():
                    validation_result[key] = result
                    pred_y = pred_ans
                else:
                    if name not in ['auc', 'logloss']:
                        raise ValueError("Evaluation Metric not supported")
                    tmp = max(validation_result[key], result) if name == 'auc' else min(validation_result[key], result)
                    if validation_result[key] != tmp:
                        validation_result[key] = tmp
                        pred_y = pred_ans
                    
                epoch_logs[key] = result
                if writer is not None: writer.add_scalar("evaluation_" + name, result, global_step=epoch)

        # verbose
        if verbose > 0:
            epoch_time = int(time.time() - start_time)
            print('Epoch {0}/{1}'.format(epoch + 1, epochs))
            print("Epoch Time : {}".format(epoch_time))
            eval_str = "{0}s - loss: {1: .4f}".format(epoch_time, epoch_logs["loss"])

            for name in metrics:
                eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in metrics:
                    eval_str += " - " + "val_" + name + ": {0: .4f}".format(epoch_logs["eval_" + name])
            print(eval_str)
    return validation_result, pred_y