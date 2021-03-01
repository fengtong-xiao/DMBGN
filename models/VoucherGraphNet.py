import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import remove_self_loops, add_self_loops
from .util import hash_func
import numpy as np

class VoucherGraphNet(torch.nn.Module):
    def __init__(self, item_features,  promotion_features, emb_info, emb_dict,
                 gprefix = ['b_', 'a_'], gactions = ['atc', 'ord'], 
                 conv_method = GraphConv,  ratio = 0.9, gnn_layers = (2, 16), linear_layers = [], device='cpu') :
        super(VoucherGraphNet, self).__init__()
        
        self.emb_dic = torch.nn.ModuleDict({
            'atc_emb' : emb_dict['atc'],
            'ord_emb' : emb_dict['ord'],
        })
        self.item_features = item_features
        self.promotion_features = promotion_features
        self.emb_info = emb_info
        
        for key, (item_size, emb_size) in emb_info.items() :
            self.emb_dic[key] = torch.nn.Embedding(num_embeddings=item_size, embedding_dim=emb_size)   
        
        embed_dim = 0
        for feat in promotion_features :
            if feat in emb_info.keys() :
                _, emb_size = emb_info[feat]
                embed_dim += emb_size
        
        self.gprefix = gprefix
        self.gactions = gactions
        
        num_layers, emb_dim = gnn_layers
        gnn_layers = [embed_dim] + [emb_dim] * num_layers
        
        # GNN Layer
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for idx in range(0, len(gnn_layers) - 1):
            conv = conv_method(gnn_layers[idx], gnn_layers[idx + 1])
            self.convs.append(conv)
            pool = TopKPooling(gnn_layers[idx + 1], ratio=ratio)
            self.pools.append(pool)
        
        # Linear Layer
        self.emb_dim = emb_dim
        hidden_units = [emb_dim * 2 * len(gprefix) * len(gactions)] + linear_layers + [self.emb_dic['promotion_id'].weight.shape[1]]
        
        if hidden_units[0] == self.emb_dic['promotion_id'].weight.shape[1] :
            hidden_units = []
        
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_units[i - 1], hidden_units[i]) for i in range(1, len(hidden_units))])      
        self.hidden_units = hidden_units
        self.device = device

    def gnn(self, gname, x, edge_index) :
        edge_index, _ = remove_self_loops(edge_index)
        batch_size = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=batch_size)
        x = x.squeeze(1)
        
        promotion_id = hash_func(x[0][0].unsqueeze(0), int(self.emb_dic[self.promotion_features[0]].weight.shape[0]))
        emb_promotion = self.emb_dic[self.promotion_features[0]](promotion_id).squeeze(1)
        for ii in range(1, len(self.promotion_features)) :
            feat = self.promotion_features[ii]
            if feat not in self.emb_info.keys() :
                continue
            promotion_val = hash_func(x[0][ii].unsqueeze(0), int(self.emb_dic[feat].weight.shape[0]))
            emb_pro = self.emb_dic[feat](promotion_val)
            if 1 < emb_pro.shape[1] :
                emb_pro = emb_pro.squeeze(1)
            emb_promotion = torch.cat([emb_promotion, emb_pro], dim=1) 
            
        if 'atc' in gname: 
            dic_key = 'atc_emb'
        elif 'ord' in gname: 
            dic_key = 'ord_emb'
        else :
            dic_key = 'clk_emb'
            
        item_ids = hash_func(x[1:][:,0], int(self.emb_dic[dic_key].weight.shape[0]))
        emb_item = self.emb_dic[dic_key](item_ids).squeeze(1)
        
        for ii in range(1, len(self.item_features)):
            feat = self.item_features[ii]
            if feat not in self.emb_info.keys() :
                continue
            item_val = hash_func(x[1:][:,ii], int(self.emb_dic[feat].weight.shape[0]))
            emb_pro = self.emb_dic[feat](item_val)
            if 1 < emb_pro.shape[1] :
                emb_pro = emb_pro.squeeze(1)
            emb_item = torch.cat([emb_item, emb_pro], dim=1)  

        x = torch.cat([emb_promotion, emb_item], dim=0)
        xx = None
        for idx in range(0, len(self.convs)):
            conv = self.convs[idx]
            pool = self.pools[idx]
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, None)
            xn = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            xx = xx + xn if xx is not None else xn
        return xx, promotion_id
            
    def forward(self, data):
        xx = None
        promotion_id = None
        session_id = None 
        
        for prefix in self.gprefix :
            for actions in self.gactions :
                gname = prefix + actions
                if gname in data.keys() :
                    graph_info = data[gname]
                    x, p_id = self.gnn(gname, graph_info.x.to(self.device), 
                                       graph_info.edge_index.to(self.device))
                    promotion_id = p_id if promotion_id is None else promotion_id
                    raw_promotion_id = graph_info.x.squeeze(1)[0][0]
                    session_id = graph_info.x.squeeze(1)[0][1]
                else :
                    x = torch.zeros(1, self.emb_dim * 2, dtype = torch.float).to(self.device)
                xx = x if xx is None else torch.cat([xx, x], dim=1) 
        if promotion_id is None :
            return None, None, None, None, None
        fc = xx
        for i in range(len(self.linears)):
            fc = self.linears[i](fc) 
        emb_promotion = self.emb_dic['promotion_id'](promotion_id).squeeze(1)
        output = torch.matmul(emb_promotion.squeeze(0), fc.squeeze(0))
        return torch.sigmoid(output.unsqueeze(0)), raw_promotion_id, emb_promotion.squeeze(0), session_id, fc.squeeze(0)