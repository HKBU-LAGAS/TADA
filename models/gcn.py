import torch
from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing, Linear, MLP
import torch.nn.functional as F

class MyLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(MyLinear, self).__init__()
        self.dropout = dropout
        self.linear = Linear(in_channels = in_channels, out_channels = out_channels)

    def forward(self, x, adj_t):
        x = self.linear(x)
        x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 drop_input: bool = False, batch_norm: bool = False, residual: bool = False, gnn_norm=True, gnn_self_loops=False):
        super(GCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.lin = MyLinear(in_channels,hidden_channels,dropout)
        self.weight = Parameter(torch.Tensor(in_channels, self.out_channels))
        self.dropout = torch.nn.Dropout(p=dropout)
        self.drop_input = drop_input
        if drop_input:
            self.input_dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()
        self.batch_norm = batch_norm
        self.residual = residual
        self.num_layers = num_layers
        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
            conv = GCNConv(in_dim, out_dim, normalize=gnn_norm, add_self_loops=gnn_self_loops) 
            self.convs.append(conv)

        if self.batch_norm:
            self.bns = ModuleList()
            for i in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)


    def forward(self, x: Tensor, adj_t: SparseTensor,*args) -> Tensor:
        
        if self.drop_input:
            x = self.input_dropout(x)
        for idx, conv in enumerate(self.convs[:-1]):
            h = conv(x, adj_t)
            if self.batch_norm:
                h = self.bns[idx](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = self.activation(h)
            x = self.dropout(x)

        x = self.convs[-1](x, adj_t)

        return x

     
    
