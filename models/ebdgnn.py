import time

from .gcn import GCN
from .gcn2 import GCN2
import torch
import torch.nn
from torch.nn import ModuleList,BatchNorm1d
from torch_geometric.nn import MessagePassing, Linear, MLP
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix,find
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor, coalesce
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from torch_sparse import SparseTensor

def get_edge(csr, i):
    row_start = 0
    for row_end in csr.indptr[1:]:
        if row_start <= i < row_end:
            break
        row_start = row_end
    else:
        raise IndexError("Index is out of bounds for the number of edges.")
    col = csr.indices[i]
    return row_start, col
def get_edge_data(i, j, csr):

    if i >= csr.shape[0] or j >= csr.shape[1]:
        raise ValueError("Source or target node index out of bounds.")

    col_indices = csr.indices[csr.indptr[i]:csr.indptr[i+1]]

    edge_positions = np.where(col_indices == j)[0]

    if edge_positions.size > 0:
        data_index = csr.indptr[i] + edge_positions[0]
        return csr.data[data_index]
    else:

        return 0
class MyLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(MyLinear, self).__init__()
        self.dropout = dropout
        self.linear = Linear(in_channels = in_channels, out_channels = out_channels)

    def forward(self, x, adj_t):
        x = self.linear(x)
        x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
        return x
    
class EbdGNN(torch.nn.Module):
    def __init__(self, in_channels, in_channels3, hidden_channels,
                 out_channels, dropout, num_layers, fi_type='ori', si_type='se', gnn_type='gcn', gamma=0.2, device='cuda:0',drop_input = True, batch_norm= False, residual = False,use_linear=False,shared_weights=True,alpha=0.0, theta=0.0, gnn_norm=True, gnn_self_loops=False):
        super(EbdGNN, self).__init__()

        self.fi_type = fi_type
        self.si_type = si_type
        self.gnn_type = gnn_type
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.gamma = gamma
        self.fw = 1. - gamma
        self.device = device
        self.convs = ModuleList()
        self.batch_norm=batch_norm
        if self.batch_norm:
            self.bns = ModuleList()
            for i in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)

        self.lin1 = MyLinear(in_channels, hidden_channels, dropout)
        if si_type == 'se':
            self.lin2 = MyLinear(in_channels3, hidden_channels, dropout)
        self.similarity_head = MyLinear(hidden_channels, out_channels, dropout)

        # gnn layers
        if gnn_type == 'gcn':
            self.backbone = GCN(in_channels=hidden_channels,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                num_layers=num_layers,
                                dropout=dropout,drop_input = drop_input, batch_norm= batch_norm, residual = residual, gnn_norm=gnn_norm, gnn_self_loops=gnn_self_loops)
        elif gnn_type == 'gcn2':
            self.backbone = GCN2(in_channels=hidden_channels,
                                  hidden_channels=hidden_channels, out_channels=out_channels,
                                  num_layers=num_layers,alpha=alpha, theta=theta,shared_weights= shared_weights,dropout=dropout,drop_input= drop_input,
                 batch_norm= batch_norm, residual= residual, gnn_norm=gnn_norm, gnn_self_loops=gnn_self_loops)
            

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def node_similarity(self, pred, data):

        pred = F.softmax(pred, dim=-1)
        edge_index = data.adj_t.coo()[:2]
        edge = torch.stack(edge_index)

        nnodes = data.x.size()[0]
        src = edge[0]
        dst=edge[1]
        batch_size = 4000
        similarity = []
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        pred_cpu = pred.cpu()
        del pred  
        torch.cuda.empty_cache()

        for i in range(0, edge.size(1), batch_size):
            start_index = i
            end_index = min(i + batch_size, edge.size(1))
            batch_edge = edge[:, start_index:end_index].cpu() 
            batch_src_pred = pred_cpu[batch_edge[0]]
            batch_dst_pred = pred_cpu[batch_edge[1]]
            batch_src_pred = torch.tensor(batch_src_pred).to(self.device)
            batch_dst_pred = torch.tensor(batch_dst_pred).to(self.device)
            batch_similarity = F.cosine_similarity(batch_src_pred, batch_dst_pred, dim=-1)
            similarity.append(batch_similarity)
            del batch_src_pred
            del batch_dst_pred
            torch.cuda.empty_cache()

        similarity = torch.cat(similarity)
        similarity_coo = coo_matrix((similarity.cpu().numpy(), (src_np, dst_np)),
                                    shape=(nnodes, nnodes))
        similarity_sum_list = similarity_coo.sum(axis=1).transpose() + similarity_coo.sum(axis=0)
        similarity_sum = torch.tensor(similarity_sum_list).to(self.device).view(-1)
        similarity = similarity * (1. / similarity_sum[edge[0]] + 1. / similarity_sum[edge[1]]) / 2

        return similarity
    
    def graph_sparse(self, similairity,graph,args):
        edge_index = graph.adj_t.coo()[:2]
        value = graph.adj_t.coo()[2]
        edge=torch.stack(edge_index)
        edges_num = edge.size(1)
        sample_rate = 1. - args.rho
        sample_edges_num = int(edges_num * sample_rate)
        degree_norm_sim = similairity
        sorted_dns = torch.sort(degree_norm_sim, descending=True)
        idx = sorted_dns.indices
        sample_edge_idx = idx[: sample_edges_num]
        edge = edge[:, sample_edge_idx]
        graph.edge_index = edge

        if value != None:
            adj_t = torch.sparse_coo_tensor(edge, value[sample_edge_idx], size=(graph.x.size(0), graph.x.size(0))).to(graph.x.device)
            graph.adj_t = SparseTensor.from_torch_sparse_coo_tensor(adj_t)
        else:
            graph = T.ToSparseTensor()(graph)

        return graph

    
    def forward(self, f,s, adj_t, state='pre'):
        febd = self.lin1(f, adj_t)
        sebd = self.lin2(s, adj_t)
        ebd = self.fw * febd + self.gamma * sebd
        ebd = F.relu(ebd)
        if state == 'pre':
            output = self.similarity_head(ebd, adj_t)
            return output
        output = self.backbone(ebd, adj_t)
        return output
