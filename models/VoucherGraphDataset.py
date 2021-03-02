import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np

item_features = ['item_id', 'item_category_id', 'item_brand_id', 'item_price_level']
promotion_features = ['promotion_id', 'session_id', 'voucher_min_spend', 'voucher_discount_amount']
all_features = item_features + promotion_features + ['action_type', 'label']

after_prefix = 'a_'
before_prefix = 'b_'


class GraphInfo():
    """ Instantiates GraphInfo object represents a UVG instance.

    :param x: torch tensor, input features from UVG
    :param edge_index: torch.tensor([source_nodes, target_nodes], dtype=torch.long), indication edge direction from
    source_nodes to target_nodes

    :return:  A GraphInfo object.
    """
    def __init__(self, x = None, edge_index = None):
        self.x = x
        self.edge_index = edge_index


class MultipleGraphData(Data):
    """ Instantiates MultipleGraphData object storing all graphInfo with corresponding redemption label for pre-training

    :param graph_dict: dictionary, key = graph name, value = GraphInfo(x, edge_index)
    :param label: torch tensor, label of the corresponding UVG

    :return:  A MultipleGraphData object.
    """
    def __init__(self, graph_dict = None, label = None):
        super(MultipleGraphData, self).__init__()
        self.graph_dict = graph_dict
        self.label = label
        
    def __inc__(self, key, value):
        if key in self.graph_dict.keys():
            return self.graph_dict[key].x.size(0)
        else:
            return super(MultipleGraphData, self).__inc__(key, value)


class VoucherGraphDataset(InMemoryDataset):
    """ Voucher graph dataset to be fitted in CPU memory

    :param root: Root directory where the dataset should be saved.
    :param processed_file_name: torch tensor, label of the corresponding UVG
    :param gnn_session_df: full session dataframe contains all related feature data of a session_id
    :param transform: None, did not use
    :param pre_transform: None, did not use

    :return: A VoucherGraphDataset object.
    """
    def __init__(self,
                 root,
                 processed_file_name,
                 gnn_session_df,
                 transform=None, 
                 pre_transform=None):
        self.processed_file_name = processed_file_name
        self.gnn_session_df = gnn_session_df
        super(VoucherGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return [self.processed_file_name]

    def download(self):
        pass
    
    def process(self):
        data_list = []

        # process by session_id
        grouped = self.gnn_session_df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            group = group.sort_values('action_time')
            group = group[group.rk <= 10]
            group_before = group[group.type == 'bef'][all_features + ['rk']]
            group_after = group[group.type == 'aft'][all_features + ['rk']]

            b = get_various_action_graphs(group_before, before_prefix)
            a = get_various_action_graphs(group_after, after_prefix)

            graph_dict = {}
            for k, g in ({**a, **b}).items() :
                if len(g) > 0 :
                    g = g.sort_values('rk')
                    gdat = get_ego_network(g, 
                                           promotion_features=promotion_features,
                                           item_features=item_features, 
                                           before_flag=True if before_prefix in k else False)
                    graph_dict[k] = gdat

            y = torch.FloatTensor([group['label'].values[0]])
            graph_data = MultipleGraphData(graph_dict, y)
            data_list.append(graph_data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# To help understand UVG:
# voucher collect => (before, after)
# action => ipv, cart, order
# node 0 => promotion id
# node 1 => item id 1
# node 2 => item id 2
# node 3 => item id 3
# node 4 => item id 4
# ....
# Edge 1 - {item id 1} => {item id 2}
# Edge 2 - {item id 2} => {item id 3}
# Edge 3 - {item id 3} => {item id 4}
# ...
#
# before collection graph
#     Edge' 1 - {item id n-1} => {promotion id}
#     Edge' 2 - {item id n-2} => {promotion id}
#     Edge' 3 - {item id n-3} => {promotion id}

# after collection graph
#     Edge' 1 - {item id 1} => {promotion id}
#     Edge' 2 - {item id 2} => {promotion id}
#     Edge' 3 - {item id 3} => {promotion id}

def get_ego_network(group,
                    promotion_features=['promotion_id', 'session_id', 'voucher_min_spend', 'voucher_discount_amount'],
                    item_features=['item_id', 'item_category_id', 'item_brand_id', 'item_price_level'],
                    before_flag=True):
    """ construct GraphInfo based on the input features for UVG, for detail explanation, please refer to
    section 3.2 UVG construction

    :param group: dataframe, which contains all required feature values
    :param promotion_features: list of feature names to be used as promotion feature
    :param item_features: list of feature names to be used as item feature
    :param before_flag: boolean, indication whether the related (item) graph node is before or after voucher collection

    :return:  A MultipleGraphData object.
    """
    le = LabelEncoder()
    sess_ids = le.fit_transform(group.item_id.values)
    # save index 0 for promotion id
    promotion_id_idx = 0
    sess_item_id = [x + 1 for x in sess_ids]
    group['sess_item_id'] = sess_item_id

    source_nodes = []
    target_nodes = []

    for ii in range(len(sess_item_id)):
        idx = sess_item_id[ii]
        if ii > 0:
            target_nodes.append(idx)
        if ii + 1 < len(sess_item_id):
            source_nodes.append(idx)

    n_items = 5
    start = 0
    end = min(n_items, len(sess_item_id))
    if before_flag is True:
        start = max(0, len(sess_item_id) - n_items)
        end = len(sess_item_id)

    for ii in range(start, end):
        idx = sess_item_id[ii]
        source_nodes.append(idx if before_flag is True else promotion_id_idx)
        target_nodes.append(promotion_id_idx if before_flag is True else idx)

    # promotion id as node 0
    promotion_feature = np.expand_dims(group[promotion_features].values[0], axis=0).astype(np.long)
    node_features = group.sort_values('sess_item_id')[item_features].drop_duplicates().values.astype(np.long)

    x_features = np.concatenate((promotion_feature, node_features), axis=0)
    x = torch.LongTensor(x_features).unsqueeze(1)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    return GraphInfo(x=x, edge_index=edge_index)


def get_various_action_graphs(group, prefix):
    group_map = {prefix + 'atc': group[group.action_type == 'cart'],
                 prefix + 'ord': group[group.action_type == 'order']}
    return group_map
