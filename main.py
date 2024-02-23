from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import os
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import json

import dgl
import dgl.data
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv, MaxPooling, GlobalAttentionPooling
from torch.utils.data import Dataset
import argparse

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from dgl.data.utils import save_graphs, load_graphs
from torch.utils.tensorboard import SummaryWriter
from shapely.geometry import mapping
from dgl.nn import SumPooling, GlobalAttentionPooling, AvgPooling, MaxPooling, SortPooling, WeightAndSum
from dgl.nn.pytorch.conv import NNConv, EGATConv, GATConv
from sklearn.model_selection import train_test_split
import time
import dgl.nn.pytorch as dglnn
import setproctitle
import warnings

os.environ['DGLBACKEND'] = 'pytorch'
setproctitle.setproctitle('pred')

parser = argparse.ArgumentParser(description='Carbon_Prediction')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--coef', type=float, default=1)
parser.add_argument('--countyOD', type=int, default=0)
parser.add_argument('--rncount', type=int, default=2)
parser.add_argument('--cbgcount', type=int, default=2)
parser.add_argument('--countycount', type=int, default=2)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--pretrainfull', type=int, default=1)
parser.add_argument('--attention', type=int, default=0)
parser.add_argument('--ablation', type=int, default=0)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--hidden_n_feats', type=int, default=8)
parser.add_argument('--hidden_e_feats', type=int, default=8)
parser.add_argument('--pretrain_epoch', type=int, default=100)
parser.add_argument('--mapping_dim', type=int, default=8)

args = parser.parse_args()

writer = SummaryWriter('training_record/scale_pred_inter_community/ablationis{}_num_headsis{}_lr_{}_bs_{}_pe_{}'.format(str(args.ablation), str(args.num_heads), str(args.lr), str(args.batch_size), str(args.pretrain_epoch)))

writer.add_scalar('lr', args.lr)
writer.add_scalar('batch_size', args.batch_size)
writer.add_scalar('pretrain_epochs', args.pretrain_epoch)
writer.add_scalar('patience', args.patience)

seed = 78
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)


class RoadNetworkDataset(Dataset):
    def __init__(self):
        graph_list, graph_labels = load_graphs('data/gnn_multigraph_insider_OD.bin')
        label_list = graph_labels['glabel']
        cbg2node_list = np.load('data/cbg2nodes_multi_layer.npy', allow_pickle=True).tolist()
        cbg2edge_list = np.load('data/cbg2edges_multi_layer.npy', allow_pickle=True).tolist()
        cbgedge2edge_list = np.load('data/cbgedge2edges_multi_layer.npy', allow_pickle=True).tolist()

        print('graph count: ', len(graph_list) / 2)

        cbg_ad_graph_list = graph_list[1::3]
        cbg_od_graph_list = graph_list[2::3]
        graph_list = graph_list[::3]

        county_graph_list, graph_labels = load_graphs(
            'data/gnn_dataset_usa_counties_three_layers_cross_county_filtered.bin')
        print('graph count: ', len(county_graph_list) / 2)
        county_od_graph_list = county_graph_list[::2]
        county_ad_graph_list = county_graph_list[1::2]

        state2county_list = np.load('data/state2county_list.npy', allow_pickle=True).item()

        bg = dgl.batch(graph_list)
        max_evalue = torch.max(bg.edata['feats'], 0).values
        min_evalue = torch.min(bg.edata['feats'], 0).values
        max_nvalue = torch.max(bg.ndata['feats'], 0).values
        min_nvalue = torch.min(bg.ndata['feats'], 0).values

        for m in range(len(graph_list)):
            graph_list[m].edata['feats'] = (graph_list[m].edata['feats'] - min_evalue) / (max_evalue - min_evalue)
            graph_list[m].ndata['feats'] = (graph_list[m].ndata['feats'] - min_nvalue) / (max_nvalue - min_nvalue)

        bg = dgl.batch(cbg_od_graph_list)
        max_evalue = torch.max(bg.edata['feats'], 0).values
        min_evalue = torch.min(bg.edata['feats'], 0).values

        for m in range(len(cbg_od_graph_list)):
            cbg_od_graph_list[m].edata['feats'] = (cbg_od_graph_list[m].edata['feats'] - min_evalue) / (
                        max_evalue - min_evalue)

        for m in range(len(county_ad_graph_list)):
            county_ad_graph_list[m].edata['feats'] = county_ad_graph_list[m].edata['feats'].float()

        for m in range(len(cbg_ad_graph_list)):
            cbg_ad_graph_list[m].edata['feats'] = torch.tensor(
                np.zeros((cbg_ad_graph_list[m].num_edges(), args.hidden_e_feats * args.num_heads))).float()

        bg = dgl.batch(county_ad_graph_list)
        max_evalue = torch.max(bg.edata['feats'], 0).values
        min_evalue = torch.min(bg.edata['feats'], 0).values

        for m in range(len(county_ad_graph_list)):
            county_ad_graph_list[m].edata['feats'] = (county_ad_graph_list[m].edata['feats'] - min_evalue) / (
                        max_evalue - min_evalue)

        bg = dgl.batch(county_od_graph_list)
        max_evalue = torch.max(bg.edata['feats'], 0).values
        min_evalue = torch.min(bg.edata['feats'], 0).values

        for m in range(len(county_od_graph_list)):
            county_od_graph_list[m].edata['feats'] = (county_od_graph_list[m].edata['feats'] - min_evalue) / (
                        max_evalue - min_evalue)

        state_fip_list = [str(m).zfill(2) for m in graph_labels['gfip'].numpy()]
        fips_list_final = np.load('data/fip_list_final.npy', allow_pickle=True)
        fip2gid = dict(zip(fips_list_final, range(len(fips_list_final))))
        gid2id = {}
        cid2gid = {}
        gid2cid = {}
        gid2cgid = {}
        count = 0
        state_count = 0
        for statefip in state_fip_list:
            for fip in state2county_list[statefip]:
                cid2gid[count + state2county_list[statefip][fip]] = fip2gid[fip]
                gid2cid[fip2gid[fip]] = count + state2county_list[statefip][fip]
                gid2id[fip2gid[fip]] = state2county_list[statefip][fip]
                gid2cgid[fip2gid[fip]] = state_count
            count += len(state2county_list[statefip])
            state_count += 1

        cid2gid_idxs = [cid2gid[idx] for idx in range(len(cid2gid))]

        self.cid2gid = cid2gid
        self.gid2cid = gid2cid
        self.cid2gid_idxs = cid2gid_idxs
        self.gid2id = gid2id
        self.gid2cgid = gid2cgid

        num_nodes_list = np.array([g.num_nodes() for g in graph_list])
        num_edges_list = np.array([g.num_edges() for g in graph_list])
        batched_graphs = dgl.batch(graph_list)
        edge_embedding_list = dgl.sum_edges(batched_graphs, 'feats').numpy()
        edge_embedding_list = np.concatenate(
            [edge_embedding_list, num_nodes_list.squeeze().reshape(-1, 1), num_edges_list.squeeze().reshape(-1, 1)],
            axis=1).tolist()
        self.embedding_dim = len(edge_embedding_list[0])
        edge_embedding_list = (edge_embedding_list - np.min(edge_embedding_list, axis=0)) / (np.max(edge_embedding_list, axis=0) - np.min(edge_embedding_list, axis=0))
        self.embedding_list = edge_embedding_list

        batched_od_county_graph = dgl.batch(county_od_graph_list)
        batched_od_county_graph.ndata['feats'] = torch.tensor(np.array(edge_embedding_list)[self.cid2gid_idxs]).float()
        county_od_graph_list = dgl.unbatch(batched_od_county_graph)

        batched_ad_county_graph = dgl.batch(county_ad_graph_list)
        batched_ad_county_graph.ndata['feats'] = torch.tensor(np.array(edge_embedding_list)[self.cid2gid_idxs]).float()
        county_ad_graph_list = dgl.unbatch(batched_ad_county_graph)

        self.label_list = label_list
        self.fip2idx = fip2gid
        self.state_list = state_fip_list
        self.graph_list = graph_list
        self.cbg_od_graph_list = cbg_od_graph_list
        self.cbg_ad_graph_list = cbg_ad_graph_list
        self.county_od_graph_list = county_od_graph_list
        self.county_ad_graph_list = county_ad_graph_list
        self.state2county = state2county_list
        self.dim_nfeats = graph_list[0].ndata['feats'].shape[1]
        self.dim_efeats = graph_list[0].edata['feats'].shape[1]
        self.dim_cbgodfeats = cbg_od_graph_list[0].edata['feats'].shape[1]
        self.dim_cbgadfeats = cbg_ad_graph_list[0].edata['feats'].shape[1]
        self.dim_countyodfeats = county_od_graph_list[0].edata['feats'].shape[1]
        self.dim_countyadfeats = county_ad_graph_list[0].edata['feats'].shape[1]

        cbg2node_matrix_list = {}
        for i in range(len(cbg2node_list)):
            cbg2node = cbg2node_list[i]
            indices = []
            for cbg in cbg2node:
                indices += [[cbg, value] for value in cbg2node[cbg]]
            values = np.ones(len(indices)).tolist()
            indices = [np.array(indices)[:, 0].tolist(), np.array(indices)[:, 1].tolist()]
            pool_matrix = torch.sparse_coo_tensor(indices, values, size=(len(cbg2node), len(graph_list[i].nodes())))
            cbg2node_matrix_list[i] = pool_matrix

        cbg2edge_matrix_list = {}
        for i in range(len(cbg2edge_list)):
            cbg2edge = cbg2edge_list[i]
            indices = []
            for cbg in cbg2edge:
                indices += [[cbg, value] for value in cbg2edge[cbg]]
            if len(indices) == 0:
                pool_matrix = torch.sparse_coo_tensor([[0], [0]], [0],
                                                      size=(len(cbg2edge), len(graph_list[i].edges()[0])))
            else:
                values = np.ones(len(indices)).tolist()
                indices = [np.array(indices)[:, 0].tolist(), np.array(indices)[:, 1].tolist()]
                pool_matrix = torch.sparse_coo_tensor(indices, values,
                                                      size=(len(cbg2edge), len(graph_list[i].edges()[0])))
            cbg2edge_matrix_list[i] = pool_matrix

        cbgedge2edge_matrix_list = {}
        for i in range(len(cbgedge2edge_list)):
            cbgedge2edge = cbgedge2edge_list[i]
            indices = []
            if len(cbgedge2edge) == 0:
                continue
            for cbgedge in cbgedge2edge:
                indices += [[cbgedge, value] for value in cbgedge2edge[cbgedge]]
            if len(indices) == 0:
                pool_matrix = torch.sparse_coo_tensor([[0], [0]], [0],
                                                      size=(
                                                          cbg_ad_graph_list[i].num_edges(),
                                                          len(graph_list[i].edges()[0])))
            else:
                values = np.ones(len(indices)).tolist()
                indices = [np.array(indices)[:, 0].tolist(), np.array(indices)[:, 1].tolist()]
                pool_matrix = torch.sparse_coo_tensor(indices, values,
                                                      size=(
                                                          cbg_ad_graph_list[i].num_edges(),
                                                          len(graph_list[i].edges()[0])))
            cbgedge2edge_matrix_list[i] = pool_matrix

        self.cbg2node_matrix_list = cbg2node_matrix_list
        self.cbg2edge_matrix_list = cbg2edge_matrix_list
        self.cbgedge2edge_matrix_list = cbgedge2edge_matrix_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        graph = self.graph_list[idx]
        cbg_od_graph = self.cbg_od_graph_list[idx]
        cbg_ad_graph = self.cbg_ad_graph_list[idx]
        county_od_graph = self.county_od_graph_list[self.gid2cgid[idx]]
        county_ad_graph = self.county_ad_graph_list[self.gid2cgid[idx]]
        label = self.label_list[idx]
        county_idx = self.gid2id[idx]
        return graph, cbg_od_graph, cbg_ad_graph, county_od_graph, county_ad_graph, label, idx, county_idx

    def update_nh(self, countynh):
        county_od_graph_list = self.county_od_graph_list
        batched_county_od_graph = dgl.batch(county_od_graph_list).to(args.device)
        batched_county_od_graph.ndata['embedding_feats'] = torch.tensor(np.array(self.embedding_list)[self.cid2gid_idxs]).float().to(args.device)
        batched_county_od_graph.ndata['feats'] = torch.cat([countynh[self.cid2gid_idxs], torch.tensor(np.array(self.embedding_list)[self.cid2gid_idxs]).float().to(args.device)], axis=1)
        county_od_graph_list = dgl.unbatch(batched_county_od_graph)
        self.county_od_graph_list = county_od_graph_list

        county_ad_graph_list = self.county_ad_graph_list
        batched_county_ad_graph = dgl.batch(county_ad_graph_list).to(args.device)
        batched_county_ad_graph.ndata['embedding_feats'] = torch.tensor(np.array(self.embedding_list)[self.cid2gid_idxs]).float().to(args.device)
        batched_county_ad_graph.ndata['feats'] = torch.cat([countynh[self.cid2gid_idxs], torch.tensor(np.array(self.embedding_list)[self.cid2gid_idxs]).float().to(args.device)], axis=1)
        county_ad_graph_list = dgl.unbatch(batched_county_ad_graph)
        self.county_ad_graph_list = county_ad_graph_list

    def update_best_nh(self):
        county_od_graph_list = self.county_od_graph_list
        batched_county_od_graph = dgl.batch(county_od_graph_list).to(args.device)
        batched_county_od_graph.ndata['best_feats'] = batched_county_od_graph.ndata['feats']
        county_od_graph_list = dgl.unbatch(batched_county_od_graph)
        self.county_od_graph_list = county_od_graph_list

        county_ad_graph_list = self.county_ad_graph_list
        batched_county_ad_graph = dgl.batch(county_ad_graph_list).to(args.device)
        batched_county_ad_graph.ndata['best_feats'] = batched_county_ad_graph.ndata['feats']
        county_ad_graph_list = dgl.unbatch(batched_county_ad_graph)
        self.county_ad_graph_list = county_ad_graph_list
    
    def load_best_nh(self):
        county_od_graph_list = self.county_od_graph_list
        batched_county_od_graph = dgl.batch(county_od_graph_list).to(args.device)
        batched_county_od_graph.ndata['feats'] = batched_county_od_graph.ndata['best_feats']
        county_od_graph_list = dgl.unbatch(batched_county_od_graph)
        self.county_od_graph_list = county_od_graph_list

        county_ad_graph_list = self.county_ad_graph_list
        batched_county_ad_graph = dgl.batch(county_ad_graph_list).to(args.device)
        batched_county_ad_graph.ndata['feats'] = batched_county_ad_graph.ndata['best_feats']
        county_ad_graph_list = dgl.unbatch(batched_county_ad_graph)
        self.county_ad_graph_list = county_ad_graph_list

dataset = RoadNetworkDataset()
cbg2node_matrix_list = dataset.cbg2node_matrix_list
cbg2edge_matrix_list = dataset.cbg2edge_matrix_list
cbgedge2edge_matrix_list = dataset.cbgedge2edge_matrix_list

cid2gid_idxs = dataset.cid2gid_idxs
graph_list = dataset.graph_list
cbg_od_graph_list = dataset.cbg_od_graph_list
cbg_ad_graph_list = dataset.cbg_ad_graph_list
batched_all_graphs = dgl.batch(graph_list).to(args.device)
batched_all_cbg_od_graphs = dgl.batch(cbg_od_graph_list).to(args.device)
batched_all_cbg_ad_graphs = dgl.batch(cbg_ad_graph_list).to(args.device)

labels = dataset.label_list
all_idxs = np.array(list(range(len(labels))))
valid_idxs = all_idxs[labels > 0]
num_examples = len(valid_idxs)
print('Num of Samples:', num_examples)

fips_list_final = np.load('data/fip_list_final.npy', allow_pickle=True)
train_idxs, test_idxs = train_test_split(list(range(num_examples)), train_size=0.8, random_state=seed)
train_idxs, val_idxs = train_test_split(train_idxs, train_size=0.75, random_state=seed)

train_idxs = valid_idxs[train_idxs]
val_idxs = valid_idxs[val_idxs]
test_idxs = valid_idxs[test_idxs]

train_sampler = SubsetRandomSampler(train_idxs)
val_sampler = SubsetRandomSampler(val_idxs)
test_sampler = SubsetRandomSampler(test_idxs)

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=False
)
val_dataloader = GraphDataLoader(
    dataset, sampler=val_sampler, batch_size=args.batch_size, drop_last=False
)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=args.batch_size, drop_last=False
)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()
        self.l1 = torch.nn.Linear(in_size, hidden_size, bias=True)
        self.ac = nn.Tanh()
        self.l2 = torch.nn.Linear(int(hidden_size), 1, bias=False)

        weights_init_2(self.l1)
        weights_init_1(self.l2)

    def forward(self, z):
        w = self.l1(z)

        w = self.ac(w)
        w = self.l2(w)

        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)


class Aggregate(nn.Module):
    def __init__(self, intra_nfeats, inter_nfeats, hidden_nfeats=32):
        super(Aggregate, self).__init__()
        self.linear1 = nn.Linear(intra_nfeats, hidden_nfeats)
        self.linear2 = nn.Linear(inter_nfeats, hidden_nfeats)
        self.attention = Attention(hidden_nfeats)

    def forward(self, intra_feats, inter_feats):
        intra_feats = self.linear1(intra_feats)
        inter_feats = self.linear2(inter_feats)
        rep = torch.stack([intra_feats, inter_feats], dim=1)
        out = self.attention(rep).squeeze()
        return out


class Aggregate_samedim(nn.Module):
    def __init__(self, intra_nfeats):
        super(Aggregate_samedim, self).__init__()
        self.attention = Attention(intra_nfeats)

    def forward(self, intra_feats, inter_feats):
        rep = torch.stack([intra_feats, inter_feats], dim=1)
        out = self.attention(rep).squeeze()
        return out

class MPNNNet(nn.Module):
    def __init__(self, in_nfeats, hidden_nfeats, in_efeats, hidden_efeats, od_efeats, ad_efeats, countynfeats,
                 countyodfeats, countyadfeats, num_heads):
        super(MPNNNet, self).__init__()
        self.hidden_feats = hidden_nfeats
        self.conv1 = EGATConv(in_node_feats=in_nfeats, in_edge_feats=in_efeats, out_node_feats=hidden_nfeats,
                              out_edge_feats=hidden_efeats, num_heads=num_heads)
        self.conv2 = EGATConv(in_node_feats=hidden_nfeats * num_heads, in_edge_feats=hidden_efeats * num_heads,
                              out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)

        self.aggregate_cbg = Aggregate_samedim(hidden_nfeats * num_heads)
        self.aggregate_county = Aggregate_samedim(hidden_nfeats * num_heads)

        self.ad_conv1 = EGATConv(in_node_feats=hidden_nfeats * num_heads + hidden_efeats * num_heads,
                                 in_edge_feats=hidden_efeats * num_heads, out_node_feats=hidden_nfeats,
                                 out_edge_feats=hidden_efeats, num_heads=num_heads)
        self.ad_conv2 = EGATConv(in_node_feats=hidden_nfeats * num_heads, in_edge_feats=hidden_efeats * num_heads,
                                 out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)

        self.od_conv1 = EGATConv(in_node_feats=hidden_nfeats * num_heads + hidden_efeats * num_heads,
                                 in_edge_feats=od_efeats, out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats,
                                 num_heads=num_heads)
        self.od_conv2 = EGATConv(in_node_feats=hidden_nfeats * num_heads, in_edge_feats=hidden_efeats * num_heads,
                                 out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)

        self.ad_conv3 = EGATConv(in_node_feats=countynfeats, in_edge_feats=countyadfeats, out_node_feats=hidden_nfeats,
                                 out_edge_feats=hidden_efeats, num_heads=num_heads)
        self.ad_conv4 = EGATConv(in_node_feats=hidden_nfeats * num_heads, in_edge_feats=hidden_efeats * num_heads,
                                 out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)

        self.od_conv3 = EGATConv(in_node_feats=countynfeats, in_edge_feats=countyodfeats, out_node_feats=hidden_nfeats,
                                 out_edge_feats=hidden_efeats, num_heads=num_heads)
        self.od_conv4 = EGATConv(in_node_feats=hidden_nfeats * num_heads, in_edge_feats=hidden_efeats * num_heads,
                                 out_node_feats=hidden_nfeats, out_edge_feats=hidden_efeats, num_heads=num_heads)

        self.map_layer = nn.Sequential(nn.Linear(hidden_nfeats * num_heads * 3, hidden_nfeats * num_heads),
                                       nn.ReLU(),
                                       nn.Linear(hidden_nfeats * num_heads, args.mapping_dim))

    def forward(self, g, cbg_od_g, cbg_ad_g, county_od_g, county_ad_g, g_idxs, cg_idxs):
        nfeat, efeat, odfeat = g.ndata['feats'], g.edata['feats'], cbg_od_g.edata['feats']
        
        nh, eh = self.conv1(g, nfeat, efeat)
        nh = F.relu(torch.flatten(nh,1))
        eh = F.relu(torch.flatten(eh,1))
        nh, eh = self.conv2(g, nh, eh)
        g.ndata['h'] = torch.flatten(nh,1)
        g.edata['h'] = torch.flatten(eh,1)
        
        graphs = dgl.unbatch(g)
        ad_graphs = dgl.unbatch(cbg_ad_g)
        od_graphs = dgl.unbatch(cbg_od_g)
        
        def get_cbg2node(cbgnode_matrix, gnh):
            pool_matrix = cbgnode_matrix.float().to(args.device)
            nhcbg = torch.sparse.mm(pool_matrix, gnh)
            count = torch.sparse.sum(pool_matrix, 1).unsqueeze(1).to_dense()
        
            nonzero_row_indices = torch.where(count != 0)[0]
            nhcbg[nonzero_row_indices, :] /= count[nonzero_row_indices]
            return nhcbg
        
        def get_cbg2edge(cbgedge_matrix, geh):
            pool_matrix = cbgedge_matrix.float().to(args.device)
            ehcbg = torch.sparse.mm(pool_matrix, geh)
            count = torch.sparse.sum(pool_matrix, 1).unsqueeze(1).to_dense()
            nonzero_row_indices = torch.where(count != 0)[0]
            ehcbg[nonzero_row_indices, :] /= count[nonzero_row_indices]
            return ehcbg
        
        for graph_id in range(len(graphs)):
            ad_graphs[graph_id].ndata['nodes'] = get_cbg2node(cbg2node_matrix_list[g_idxs.numpy().tolist()[graph_id]],
                                                              graphs[graph_id].ndata['h'])
            ad_graphs[graph_id].ndata['edges'] = get_cbg2edge(cbg2edge_matrix_list[g_idxs.numpy().tolist()[graph_id]],
                                                              graphs[graph_id].edata['h'])
            ad_graphs[graph_id].ndata['h'] = torch.cat(
                (ad_graphs[graph_id].ndata['nodes'], ad_graphs[graph_id].ndata['edges']), 1)
            od_graphs[graph_id].ndata['h'] = ad_graphs[graph_id].ndata['h']
        
        def get_cbg2edge2edge(cbgedge2edge_matrix, geh):
            pool_matrix = cbgedge2edge_matrix.float().to(args.device)
            ehcbg = torch.sparse.mm(pool_matrix, geh)
            count = torch.sparse.sum(pool_matrix, 1).unsqueeze(1).to_dense()
        
            nonzero_row_indices = torch.where(count != 0)[0]
            ehcbg[nonzero_row_indices, :] /= count[nonzero_row_indices]
            return ehcbg
        
        for graph_id in range(len(graphs)):
            if g_idxs.numpy().tolist()[graph_id] not in cbgedge2edge_matrix_list:
                continue
            ad_graphs[graph_id].edata['feats'] = get_cbg2edge2edge(cbgedge2edge_matrix_list[g_idxs.numpy().tolist()[graph_id]], graphs[graph_id].edata['h'])
        cbg_ad_g = dgl.batch(ad_graphs)
        cbg_od_g = dgl.batch(od_graphs)
        
        nh, eh = self.ad_conv1(cbg_ad_g, cbg_ad_g.ndata['h'], cbg_ad_g.edata['feats'])
        nh = F.relu(torch.flatten(nh, 1))
        eh = F.relu(torch.flatten(eh, 1))
        nh1, eh1 = self.od_conv1(cbg_od_g, cbg_od_g.ndata['h'], odfeat)
        nh1 = F.relu(torch.flatten(nh1, 1))
        eh1 = F.relu(torch.flatten(eh1, 1))
        
        if args.ablation == 0:
            nh = self.aggregate_cbg(nh, nh1)
        if args.ablation == 1:
            nh = nh
        if args.ablation == 2:
            nh = nh1
        
        nh, eh = self.ad_conv2(cbg_ad_g, nh, eh)
        nh = F.relu(torch.flatten(nh, 1))
        eh = F.relu(torch.flatten(eh, 1))
        nh1, eh1 = self.od_conv2(cbg_od_g, nh, eh1)
        nh1 = F.relu(torch.flatten(nh1, 1))
        
        if args.ablation == 0:
            nh = self.aggregate_cbg(nh, nh1)
        if args.ablation == 1:
            nh = nh
        if args.ablation == 2:
            nh = nh1

        cbg_ad_g.ndata['h'] = torch.flatten(nh, 1)
        
        # county level
        nhg = dgl.mean_nodes(g, 'h')
        ehg = dgl.mean_edges(g, 'h')
        nhcbg = dgl.mean_nodes(cbg_ad_g, 'h')
        county_ad_nh_ori = torch.cat([nhg, ehg, nhcbg], 1)
        county_ad_nh_ori= self.map_layer(county_ad_nh_ori)

        county_ad_graphs = dgl.unbatch(county_ad_g)
        em_nh = torch.stack([county_ad_graphs[i].ndata['embedding_feats'][cg_idxs[i]] for i in range(len(county_ad_graphs))],0)
        county_ad_nh_ori = torch.cat([county_ad_nh_ori, em_nh], axis=1)

        county_nh = county_ad_g.ndata['feats']

        nh, eh = self.ad_conv3(county_ad_g, county_nh, county_ad_g.edata['feats'])
        nh = F.relu(torch.flatten(nh, 1))
        eh = F.relu(torch.flatten(eh, 1))

        nh1, eh1 = self.od_conv3(county_od_g, county_nh, county_od_g.edata['feats'])
        nh1 = F.relu(torch.flatten(nh1, 1))
        eh1 = F.relu(torch.flatten(eh1, 1))

        if args.ablation == 0:
            nh = self.aggregate_county(nh, nh1)
        if args.ablation == 1:
            nh = nh
        if args.ablation == 2:
            nh = nh1

        nh, eh = self.ad_conv4(county_ad_g, nh, eh)
        nh = F.relu(torch.flatten(nh, 1))
        nh1, eh1 = self.od_conv4(county_od_g, nh, eh1)
        nh1 = F.relu(torch.flatten(nh1, 1))

        if args.ablation == 0:
            nh = self.aggregate_county(nh, nh1)
        if args.ablation == 1:
            nh = nh
        if args.ablation == 2:
            nh = nh1

        county_ad_g.ndata['h'] = torch.flatten(nh, 1)
        county_ad_g.edata['h'] = torch.flatten(eh, 1)

        county_ad_graphs = dgl.unbatch(county_ad_g)
        county_ad_nh = torch.stack([county_ad_graphs[i].ndata['h'][cg_idxs[i]] for i in range(len(county_ad_graphs))],0)
        return county_ad_nh, county_ad_nh_ori

    def pretrain(self, g, cbg_od_g, cbg_ad_g, county_od_g, county_ad_g, g_idxs, cg_idxs):
        county_nh = county_ad_g.ndata['feats']

        nh, eh = self.ad_conv3(county_ad_g, county_nh, county_ad_g.edata['feats'])
        nh = F.relu(torch.flatten(nh, 1))
        eh = F.relu(torch.flatten(eh, 1))

        nh1, eh1 = self.od_conv3(county_od_g, county_nh, county_od_g.edata['feats'])
        nh1 = F.relu(torch.flatten(nh1, 1))
        eh1 = F.relu(torch.flatten(eh1, 1))

        if args.ablation == 0:
            nh = self.aggregate_county(nh, nh1)
        if args.ablation == 1:
            nh = nh
        if args.ablation == 2:
            nh = nh1

        nh, eh = self.ad_conv4(county_ad_g, nh, eh)
        nh = F.relu(torch.flatten(nh, 1))
        nh1, eh1 = self.od_conv4(county_od_g, nh, eh1)
        nh1 = F.relu(torch.flatten(nh1, 1))

        if args.ablation == 0:
            nh = self.aggregate_county(nh, nh1)
        if args.ablation == 1:
            nh = nh
        if args.ablation == 2:
            nh = nh1

        county_ad_g.ndata['h'] = torch.flatten(nh, 1)
        county_ad_g.edata['h'] = torch.flatten(eh, 1)

        county_ad_graphs = dgl.unbatch(county_ad_g)
        county_ad_nh = torch.stack([county_ad_graphs[i].ndata['h'][cg_idxs[i]] for i in range(len(county_ad_graphs))],0)
        county_ad_nh_ori = torch.stack([county_ad_graphs[i].ndata['feats'][cg_idxs[i]] for i in range(len(county_ad_graphs))], 0)
        return county_ad_nh, county_ad_nh_ori

    def update_nh(self, batched_g, batched_cbg_ad_g, batched_cbg_od_g):
        with torch.no_grad():
            nh, eh = self.conv1(batched_g, batched_g.ndata['feats'], batched_g.edata['feats'])
            nh = F.relu(torch.flatten(nh, 1))
            eh = F.relu(torch.flatten(eh, 1))
            nh, eh = self.conv2(batched_g, nh, eh)
            batched_g.ndata['h'] = torch.flatten(nh, 1)
            batched_g.edata['h'] = torch.flatten(eh, 1)

            graphs = dgl.unbatch(batched_g)
            ad_graphs = dgl.unbatch(batched_cbg_ad_g)
            od_graphs = dgl.unbatch(batched_cbg_od_g)

            def get_cbg2node(cbgnode_matrix, gnh):
                pool_matrix = cbgnode_matrix.float().to(args.device)
                nhcbg = torch.sparse.mm(pool_matrix, gnh)
                count = torch.sparse.sum(pool_matrix, 1).unsqueeze(1).to_dense()

                nonzero_row_indices = torch.where(count != 0)[0]
                nhcbg[nonzero_row_indices, :] /= count[nonzero_row_indices]
                return nhcbg

            def get_cbg2edge(cbgedge_matrix, geh):
                pool_matrix = cbgedge_matrix.float().to(args.device)
                ehcbg = torch.sparse.mm(pool_matrix, geh)
                count = torch.sparse.sum(pool_matrix, 1).unsqueeze(1).to_dense()
                nonzero_row_indices = torch.where(count != 0)[0]
                ehcbg[nonzero_row_indices, :] /= count[nonzero_row_indices]
                return ehcbg

            for graph_id in range(len(graphs)):
                ad_graphs[graph_id].ndata['nodes'] = get_cbg2node(cbg2node_matrix_list[graph_id],
                                                                  graphs[graph_id].ndata['h'])
                ad_graphs[graph_id].ndata['edges'] = get_cbg2edge(cbg2edge_matrix_list[graph_id],
                                                                  graphs[graph_id].edata['h'])
                ad_graphs[graph_id].ndata['h'] = torch.cat(
                    (ad_graphs[graph_id].ndata['nodes'], ad_graphs[graph_id].ndata['edges']), 1)
                od_graphs[graph_id].ndata['h'] = ad_graphs[graph_id].ndata['h']

            def get_cbg2edge2edge(cbgedge2edge_matrix, geh):
                pool_matrix = cbgedge2edge_matrix.float().to(args.device)
                ehcbg = torch.sparse.mm(pool_matrix, geh)
                count = torch.sparse.sum(pool_matrix, 1).unsqueeze(1).to_dense()

                nonzero_row_indices = torch.where(count != 0)[0]
                ehcbg[nonzero_row_indices, :] /= count[nonzero_row_indices]
                return ehcbg

            for graph_id in range(len(graphs)):
                if graph_id not in cbgedge2edge_matrix_list:
                    continue
                ad_graphs[graph_id].edata['feats'] = get_cbg2edge2edge(cbgedge2edge_matrix_list[graph_id],
                                                                       graphs[graph_id].edata['h'])
            batched_cbg_ad_g = dgl.batch(ad_graphs)
            batched_cbg_od_g = dgl.batch(od_graphs)

            nh, eh = self.ad_conv1(batched_cbg_ad_g, batched_cbg_ad_g.ndata['h'], batched_cbg_ad_g.edata['feats'])
            nh = F.relu(torch.flatten(nh, 1))
            eh = F.relu(torch.flatten(eh, 1))
            nh1, eh1 = self.od_conv1(batched_cbg_od_g, batched_cbg_od_g.ndata['h'], batched_cbg_od_g.edata['feats'])
            nh1 = F.relu(torch.flatten(nh1, 1))
            eh1 = F.relu(torch.flatten(eh1, 1))
            if args.ablation == 0:
                nh = self.aggregate_cbg(nh, nh1)
            if args.ablation == 1:
                nh = nh
            if args.ablation == 2:
                nh = nh1

            nh, eh = self.ad_conv2(batched_cbg_ad_g, nh, eh)
            nh = F.relu(torch.flatten(nh, 1))
            nh1, eh1 = self.od_conv2(batched_cbg_od_g, nh, eh1)
            nh1 = F.relu(torch.flatten(nh1, 1))
            if args.ablation == 0:
                nh = self.aggregate_cbg(nh, nh1)
            if args.ablation == 1:
                nh = nh
            if args.ablation == 2:
                nh = nh1

            batched_cbg_ad_g.ndata['h'] = torch.flatten(nh, 1)

            nhg = dgl.mean_nodes(batched_g, 'h')
            ehg = dgl.mean_edges(batched_g, 'h')
            nhcbg = dgl.mean_nodes(batched_cbg_ad_g, 'h')
            countynh = torch.cat([nhg, ehg, nhcbg], 1)
            countynh = self.map_layer(countynh)
        return countynh


def weights_init_1(m):
    torch.nn.init.xavier_uniform_(m.weight, gain=1)

def weights_init_2(m):
    torch.nn.init.xavier_uniform_(m.weight, gain=1)
    torch.nn.init.constant_(m.bias, 0)

num_heads = args.num_heads
args.intra_dim = args.mapping_dim + dataset.embedding_dim

class Regressor(nn.Module):
    def __init__(self, in_nfeats, h_feats, in_efeats, h_efeats, od_efeats, ad_efeats, countynfeats, countyodfeats,
                 countyadfeats, num_heads=3):
        super(Regressor, self).__init__()
        self.gcn = MPNNNet(in_nfeats, h_feats, in_efeats, h_efeats, od_efeats, ad_efeats, countynfeats, countyodfeats,
                           countyadfeats, num_heads)
        if args.attention == 0:
            self.regressor = nn.Sequential(nn.Linear(h_feats * num_heads+args.intra_dim, h_feats * 2 * 2),
                                           nn.ReLU(),
                                           nn.Linear(h_feats * 2 * 2, h_feats * 2),
                                           nn.ReLU(),
                                           nn.Linear(h_feats * 2, h_feats),
                                           nn.ReLU(),
                                           nn.Linear(h_feats, h_feats),
                                           nn.ReLU(),
                                           nn.Linear(h_feats, 1))
        else:
            self.regressor = nn.Sequential(nn.Linear(h_feats * num_heads, h_feats * 2 * 2),
                                           nn.ReLU(),
                                           nn.Linear(h_feats * 2 * 2, h_feats * 2),
                                           nn.ReLU(),
                                           nn.Linear(h_feats * 2, h_feats),
                                           nn.ReLU(),
                                           nn.Linear(h_feats, h_feats),
                                           nn.ReLU(),
                                           nn.Linear(h_feats, 1))
        self.aggregate = Aggregate(args.intra_dim, h_feats * num_heads, h_feats * num_heads)

    def forward(self, g, cbg_od_g, cbg_ad_g, county_od_g, county_ad_g, g_idxs, cg_idxs):
        county_ad_nh, county_ad_nh_ori = self.gcn(g, cbg_od_g, cbg_ad_g, county_od_g, county_ad_g, g_idxs, cg_idxs)
        if args.attention == 0:
            pred = self.regressor(torch.cat([county_ad_nh, county_ad_nh_ori], 1))
        else:
            nh = self.aggregate(county_ad_nh_ori, county_ad_nh)
            pred = self.regressor(nh)
        return pred

    def pretrain(self, g, cbg_od_g, cbg_ad_g, county_od_g, county_ad_g, g_idxs, cg_idxs):
        county_ad_nh, county_ad_nh_ori = self.gcn.pretrain(g, cbg_od_g, cbg_ad_g, county_od_g, county_ad_g, g_idxs, cg_idxs)
        if args.attention == 0:
            pred = self.regressor(torch.cat([county_ad_nh, county_ad_nh_ori], 1))
        else:
            nh = self.aggregate(county_ad_nh_ori, county_ad_nh)
            pred = self.regressor(nh)
        return pred

    def update_nh(self, batched_g, batched_cbg_od_g, batched_cbg_ad_g):
        return self.gcn.update_nh(batched_g, batched_cbg_ad_g, batched_cbg_od_g)

model = Regressor(dataset.dim_nfeats, args.hidden_n_feats, dataset.dim_efeats, args.hidden_e_feats,
                  dataset.dim_cbgodfeats, dataset.dim_cbgadfeats, args.intra_dim, dataset.dim_countyodfeats,
                  dataset.dim_countyadfeats, num_heads).to(args.device)
if args.pretrainfull:
    pretrained_model = torch.load('data/models/pretrained_model_0.01_8_8_8.pt', map_location=args.device)

selected_keys = ['gcn.conv1', 'gcn.conv2', 'gcn.ad_conv1', 'gcn.ad_conv2', 'gcn.od_conv1', 'gcn.od_conv2', 'gcn.aggregate_cbg']
selected_dict = {}
for key in selected_keys:
    selected_dict.update({k: v for k, v in pretrained_model['model_state_dict'].items() if key in k})
model.load_state_dict(selected_dict, strict=False)

print('Lr:', args.lr)
print('Batch Size:', args.batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss = torch.nn.MSELoss(reduce='sum')
scale_loss = torch.nn.MSELoss(reduce='sum')


def train(epoch, n_step):
    model.train()
    train_preds, train_labels = [], []
    losses = 0
    for batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs, batched_county_ad_graph, labels, idxs, cidxs in train_dataloader:
        batched_graphs = batched_graphs.to(args.device)
        batched_cbg_od_graphs = batched_cbg_od_graphs.to(args.device)
        batched_cbg_ad_graphs = batched_cbg_ad_graphs.to(args.device)
        batched_county_od_graphs = batched_county_od_graphs.to(args.device)
        batched_county_ad_graph = batched_county_ad_graph.to(args.device)
        labels = labels.to(args.device)
        if epoch < args.pretrain_epoch:
            pred = model.pretrain(batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs,
                         batched_county_ad_graph, idxs, cidxs)
        else:
            pred = model(batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs,
                        batched_county_ad_graph, idxs, cidxs)
        loss_epoch = loss(pred.squeeze().float(), labels.float())
        loss_total = loss_epoch
        losses += loss_epoch.item()
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        train_preds += list(pred.cpu().detach().numpy())
        train_labels += list(labels.cpu().detach().numpy())
    mae = mean_absolute_error(train_preds, train_labels)
    rmse = np.sqrt(mean_squared_error(train_preds, train_labels))
    r_2 = r2_score(train_labels, train_preds)
    print('Training Epoch {} Pred Loss:{:.3f} MAE:{:.3f}, RMSE:{:.3f}, R2: {:.2f}'.format(epoch,
                                                                                          losses / len(train_preds),
                                                                                          mae, rmse, r_2))
    writer.add_scalar('train_pred_loss', losses / len(train_preds), n_step)
    writer.add_scalar('train_mae', mae, n_step)
    writer.add_scalar('train_rmse', rmse, n_step)
    writer.add_scalar('train_r2', r_2, n_step)


def val(epoch, t_step):
    model.eval()
    val_losses = 0
    val_preds, val_labels = [], []
    for batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs, batched_county_ad_graph, labels, idxs, cidxs in val_dataloader:
        batched_graphs = batched_graphs.to(args.device)
        batched_cbg_od_graphs = batched_cbg_od_graphs.to(args.device)
        batched_cbg_ad_graphs = batched_cbg_ad_graphs.to(args.device)
        batched_county_od_graphs = batched_county_od_graphs.to(args.device)
        batched_county_ad_graph = batched_county_ad_graph.to(args.device)
        labels = labels.to(args.device)
        if epoch < args.pretrain_epoch:
            pred = model.pretrain(batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs,
                         batched_county_ad_graph, idxs, cidxs)
        else:
            pred = model(batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs,
                        batched_county_ad_graph, idxs, cidxs)       
        loss_epoch = loss(pred.squeeze().float(), labels.float())
        val_losses += loss_epoch.item()
        val_preds += list(pred.cpu().detach().numpy())
        val_labels += list(labels.cpu().detach().numpy())
    mae = mean_absolute_error(val_preds, val_labels)
    rmse = np.sqrt(mean_squared_error(val_preds, val_labels))
    r_2 = r2_score(val_labels, val_preds)
    print('Validation Epoch {} Pred Loss:{:.3f}, MAE:{:.3f}, RMSE:{:.3f}, R2: {:.2f}'.format(epoch,
                                                                                             val_losses / len(
                                                                                                 val_preds), mae,
                                                                                             rmse, r_2))
    writer.add_scalar('val_pred_loss', val_losses / len(val_preds), t_step)
    writer.add_scalar('val_mae', mae, t_step)
    writer.add_scalar('val_rmse', rmse, t_step)
    writer.add_scalar('val_R2', r_2, t_step)
    return r_2


def test(epoch, t_step):
    model.eval()
    test_losses = 0
    test_preds, test_labels = [], []
    for batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs, batched_county_ad_graph, labels, idxs, cidxs in test_dataloader:
        batched_graphs = batched_graphs.to(args.device)
        batched_cbg_od_graphs = batched_cbg_od_graphs.to(args.device)
        batched_cbg_ad_graphs = batched_cbg_ad_graphs.to(args.device)
        batched_county_od_graphs = batched_county_od_graphs.to(args.device)
        batched_county_ad_graph = batched_county_ad_graph.to(args.device)
        labels = labels.to(args.device)
        pred = model(batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs,
                     batched_county_ad_graph, idxs, cidxs)
        loss_epoch = loss(pred.squeeze().float(), labels.float())
        test_losses += loss_epoch.item()
        test_preds += list(pred.cpu().detach().numpy())
        test_labels += list(labels.cpu().detach().numpy())
    mae = mean_absolute_error(test_preds, test_labels)
    rmse = np.sqrt(mean_squared_error(test_preds, test_labels))
    r_2 = r2_score(test_labels, test_preds)
    print('Test Epoch {} Pred Loss:{:.3f}, MAE:{:.3f}, RMSE:{:.3f}, R2: {:.2f}'.format(epoch,
                                                                                       test_losses / len(
                                                                                           test_preds), mae,
                                                                                       rmse, r_2))
    writer.add_scalar('test_pred_loss', test_losses / len(test_preds), t_step)
    writer.add_scalar('test_mae', mae, t_step)
    writer.add_scalar('test_rmse', rmse, t_step)
    writer.add_scalar('test_R2', r_2, t_step)


n_step, t_step = 0, 0
patience = 0
min_loss, min_loss_intra = -10, -10
best_epoch = 0
model = model.eval()
countynh = model.update_nh(batched_all_graphs, batched_all_cbg_od_graphs, batched_all_cbg_ad_graphs)
dataset.update_nh(countynh)

for epoch in range(args.epochs):
    # training
    train(epoch, n_step)
    # validation
    if epoch % 5 == 0:
        val_r2 = val(epoch, t_step)
        test(epoch, t_step)
        if val_r2 > min_loss:
            dataset.update_best_nh()           
            min_loss = val_r2
            patience = 0
            fname = 'training_record/county_numheadsis{}_{}_{}_{}_{}_attentionis{}_pretrainepochsis{}_mappinglayeris{}.pt'.format(str(args.num_heads),
                                                                                             str(args.lr), str(args.batch_size), str(args.hidden_e_feats), args.hidden_n_feats,  str(args.attention), str(args.pretrain_epoch), str(args.mapping_dim))
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, fname, _use_new_zipfile_serialization=False)
        t_step += 1
        patience += 1
        if patience > args.patience and epoch > args.pretrain_epoch:
            print('Early Stopping at Epoch:{}'.format(epoch))
            break
    n_step += 1
    if epoch > args.pretrain_epoch:
        countynh = model.update_nh(batched_all_graphs, batched_all_cbg_od_graphs, batched_all_cbg_ad_graphs)
        dataset.update_nh(countynh)

# test
fname = 'training_record/county_numheadsis{}_{}_{}_{}_{}_attentionis{}_pretrainepochsis{}_mappinglayeris{}.pt'.format(str(args.num_heads),
                                                                                             str(args.lr), str(args.batch_size), str(args.hidden_e_feats), args.hidden_n_feats,  str(args.attention), str(args.pretrain_epoch), str(args.mapping_dim))
checkpoint = torch.load(fname)
model.load_state_dict(checkpoint['model_state_dict'])
dataset.load_best_nh()

def test_final(epoch):
    model.eval()
    test_losses = 0
    test_preds, test_labels = [], []
    for batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs, batched_county_ad_graph, labels, idxs, cidxs in test_dataloader:
        batched_graphs = batched_graphs.to(args.device)
        batched_cbg_od_graphs = batched_cbg_od_graphs.to(args.device)
        batched_cbg_ad_graphs = batched_cbg_ad_graphs.to(args.device)
        batched_county_od_graphs = batched_county_od_graphs.to(args.device)
        batched_county_ad_graph = batched_county_ad_graph.to(args.device)
        labels = labels.to(args.device)
        pred = model(batched_graphs, batched_cbg_od_graphs, batched_cbg_ad_graphs, batched_county_od_graphs,
                     batched_county_ad_graph, idxs, cidxs)
        loss_epoch = loss(pred.squeeze().float(), labels.float())
        test_losses += loss_epoch.item()
        test_preds += list(pred.cpu().detach().numpy())
        test_labels += list(labels.cpu().detach().numpy())
    mae = mean_absolute_error(test_preds, test_labels)
    rmse = np.sqrt(mean_squared_error(test_preds, test_labels))
    r_2 = r2_score(test_labels, test_preds)
    print('Test Epoch {} Pred Loss:{:.3f}, MAE:{:.3f}, RMSE:{:.3f}, R2: {:.2f}'.format(epoch,
                                                                                       test_losses / len(
                                                                                           test_preds), mae,
                                                                                       rmse, r_2))
    writer.add_scalar('test_pred_loss_final', test_losses / len(test_preds))
    writer.add_scalar('test_mae_final', mae)
    writer.add_scalar('test_rmse_final', rmse)
    writer.add_scalar('test_R2_final', r_2)
    writer.close()


test_final(epoch)

