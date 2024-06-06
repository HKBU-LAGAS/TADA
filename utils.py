
from typing import Optional, Tuple
import random

from scipy.sparse import csr_matrix, diags
from scipy.linalg import clarkson_woodruff_transform
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import TruncatedSVD

import numpy as np 

import torch
import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor

from ogb.nodeproppred import Evaluator
from torch_geometric.utils import subgraph, to_undirected
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric.transforms as T
import dgl 

######################################################################

ARXIV_EVAL = Evaluator(name='ogbn-arxiv')
PRODUCTS_EVAL = Evaluator(name='ogbn-products')

def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

def ogb_get_acc(out, y, train_mask, val_mask, test_mask, dataset):
    y = y.unsqueeze(1)
    if dataset == 'arxiv':
        evaluator = ARXIV_EVAL
    elif dataset in ['products-cluster', 'products-saint']:
        evaluator = PRODUCTS_EVAL
    else:
        raise NotImplementedError
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': y[train_mask],
        'y_pred': y_pred[train_mask],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[val_mask],
        'y_pred': y_pred[val_mask],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[test_mask],
        'y_pred': y_pred[test_mask],
    })['acc']

    return train_acc, valid_acc, test_acc

def compute_micro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)

    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.

def get_optimizer(model_config, model):
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'], weight_decay = model_config['wd'])
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model_config['lr'], weight_decay = model_config['wd'])
    else:
        raise NotImplementedError
    
    return optimizer

def to_inductive(data):
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def rwr_clustering(data, arg_list, k=256):
    alpha = 0.5 
    t = int(1 / alpha)
    x =0.5 
    y = 1. - x
    t1 = arg_list[0]
    t2 = 1
    init_range = 5 * k

    nnodes = data.shape[0]
    ones_vector = np.ones(nnodes, dtype=float)
    degrees = data.dot(ones_vector)
    degrees_inv = diags((1. / (degrees + 1e-10)).tolist())

    # clustering: random walk with restart
    topk_deg_nodes = np.argpartition(degrees, -init_range)[-init_range:]
    P = degrees_inv.dot(data)

    # init k clustering centers
    PC = P[:, topk_deg_nodes]
    M = PC

    for i in range(t):
        M = (1 - alpha) * P.dot(M) + PC

    cluster_sum = M.sum(axis=0).flatten().tolist()[0]
    newcandidates = np.argpartition(cluster_sum, -k)[-k:]
    M = M[:, newcandidates]

    column_sqrt = diags((1. / (np.squeeze(np.asarray(M.sum(axis=-1))) ** x + 1e-10)).tolist())
    row_sqrt = diags((1. / (np.squeeze(np.asarray(M.sum(axis=0))) ** y + 1e-10)).tolist())
    prob = column_sqrt.dot(M).dot(row_sqrt)

    center_idx = np.squeeze(np.asarray(prob.argmax(axis=-1)))

    cluster_center = csr_matrix(([1.] * nnodes, (np.array([i for i in range(nnodes)]), center_idx
                                                    )),
                                shape=(nnodes, k))

    random_flip = diags(np.where(np.random.rand(nnodes) > 0.5, 1., -1.).tolist())
    sketching = csr_matrix(
        ([1.] * nnodes, (np.array([i for i in range(nnodes)]), np.random.randint(0, k, nnodes))),
        shape=(nnodes, k))
    sketching = random_flip.dot(sketching)

    ebd = data.dot((t1 * random_flip.dot(cluster_center) + t2 * sketching))

    return ebd
  

def ebding_function(ebd_source, ebd_type, ebd_dim, arg_list=[]):
    if ebd_type == 'ori':
        ebd = ebd_source
    # ours embedding method, using beta to control the weight of count sketch and RWR 
    elif ebd_type == 'rwr': 
        ebd = rwr_clustering(ebd_source, arg_list, k=ebd_dim)
    elif ebd_type == 'cwt':
        print('cwt')
        ebd = clarkson_woodruff_transform(ebd_source.transpose(), ebd_dim).transpose()
    return ebd

def se(data, args):
    se_type = args.se_type
    se_dim = args.se_dim
    edges = data.edge_index.numpy()
    rows = edges[0]
    cols = edges[1]
    Feat = data.x.numpy()
    nnodes = Feat.shape[0]
    A = csr_matrix(([1.0] * len(rows), (rows, cols)), (nnodes, nnodes))  

    Feat = csr_matrix(Feat)
    if se_type == 'rwr':
        se_arg_list = [args.beta]
    else:
        se_arg_list = []
    sebd = ebding_function(A, se_type, se_dim, se_arg_list)
    if not isinstance(sebd, np.ndarray):
        sebd = sebd.toarray()
    data.sebd = sebd

    return data


