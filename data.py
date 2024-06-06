import os
from typing import Tuple

import scipy
from sklearn.preprocessing import label_binarize
import numpy as np
import  torch
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import scatter,to_undirected, is_undirected 
from torch_geometric.datasets import  Amazon,WikipediaNetwork, WikiCS,Reddit2
from ogb.nodeproppred import PygNodePropPredDataset
from utils import index2mask

script_dir = os.path.dirname(os.path.realpath(__file__))

def split_data(data):
    num_nodes = data.x.shape[0]
    num_train = int(num_nodes * 0.6)
    num_val = int(num_nodes * 0.2)
    num_test = num_nodes - (num_train + num_val)
    indices = torch.randperm(num_nodes)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:(num_train + num_val)]
    test_indices = indices[(num_train + num_val):]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask

def build_adj_t(data):
    (row, col), N = data.edge_index, data.num_nodes
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]
    value = None
    for key in ['edge_weight', 'edge_attr', 'edge_type']:
        if data[key] is not None:
            value = data[key][perm]
            break
    adj_t = SparseTensor(row=col, col=row, value=value,
                                  sparse_sizes=(N, N), is_sorted=True)
    adj_t.storage.rowptr()
    adj_t.storage.csc()
    return adj_t

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.data = {}
        self.label = None
        self.num_classes=0

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.data, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}

        return split_idx
    
def rand_train_test_idx(data, train_prop=.6, valid_prop=.2, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(data.y != -1)[0]
    else:
        labeled_nodes = data.y
    
    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]
    return train_idx, valid_idx, test_idx

def load_fb100():
    filepath = os.path.join(script_dir, 'data', 'Penn94.mat')
    mat = scipy.io.loadmat(filepath)
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata

def load_penn_dataset():
    A, metadata = load_fb100()
    filename='penn'
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = np.round(metadata).astype(np.int_)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled
    dataset.num_classes = 2
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(label)

    data = Data(x=x, edge_index=edge_index, y=y)
    dataset.data = data
    dataset.num_features=features.shape[1]
    return dataset

def load_pokec_mat():
    """ requires pokec.mat
    """
    file_path=os.path.join(script_dir,'data','pokec.mat')
    fulldata = scipy.io.loadmat(file_path)
    dataset = NCDataset('pokec')
    dataset.num_classes = 2
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    x = torch.tensor(
        fulldata['node_feat'], dtype=torch.float)
    label = fulldata['label'].flatten()
    y = torch.tensor(label, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    dataset.data = data
    dataset.num_features=x.shape[1]
    return dataset

def get_photo(root: str) -> Tuple[Data, int, int]:
    dataset = Amazon(root=root, name="Photo")
    data=dataset[0]
    train_mask_list = []
    val_mask_list = [] 
    test_mask_list = [ ] 
    for i  in range(10):
        train_mask, val_mask, test_mask =split_data(data)
        train_mask_list.append(train_mask)
        val_mask_list.append(val_mask) 
        test_mask_list.append(test_mask) 
    data.train_mask = torch.stack(train_mask_list,dim=0)
    data.val_mask = torch.stack(val_mask_list,dim=0)
    data.test_mask = torch.stack(test_mask_list,dim=0)
    return data, dataset.num_features, dataset.num_classes

def get_wikics(root: str) -> Tuple[Data, int, int]:
    dataset = WikiCS(root = root, is_undirected = True)
    data=dataset[0]
    
    train_mask_list = []
    val_mask_list = [] 
    test_mask_list = [ ] 
    for i  in range(10):
        train_mask, val_mask, test_mask =split_data(data)
        train_mask_list.append(train_mask)
        val_mask_list.append(val_mask) 
        test_mask_list.append(test_mask) 
    data.train_mask = torch.stack(train_mask_list,dim=0)
    data.val_mask = torch.stack(val_mask_list,dim=0)
    data.test_mask = torch.stack(test_mask_list,dim=0)
    return data, dataset.num_features, dataset.num_classes

def get_reddit2(root: str) -> Tuple[Data, int, int]:
    dataset = Reddit2(f'{root}/Reddit2')
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)

    train_mask_list = []
    val_mask_list = [] 
    test_mask_list = [ ] 
    for i  in range(10):
        train_mask= data.train_mask 
        val_mask= data.val_mask
        test_mask =data.test_mask
        train_mask_list.append(train_mask)
        val_mask_list.append(val_mask) 
        test_mask_list.append(test_mask) 
    data.train_mask = torch.stack(train_mask_list,dim=0)
    data.val_mask = torch.stack(val_mask_list,dim=0)
    data.test_mask = torch.stack(test_mask_list,dim=0)

    return data, dataset.num_features, dataset.num_classes

def get_products(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-products', f'{root}/OGB')
    data = dataset[0]
    data.x = data.x.contiguous()
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, dataset.num_features, dataset.num_classes

def get_squirrel(root: str) -> Tuple[Data, int, int]:
    dataset = WikipediaNetwork(root = root, name = 'squirrel')
    data = dataset[0]
    train_mask_list = []
    val_mask_list = [] 
    test_mask_list = [ ] 
    for i  in range(10):
        train_mask, val_mask, test_mask =split_data(data)
        train_mask_list.append(train_mask)
        val_mask_list.append(val_mask) 
        test_mask_list.append(test_mask) 
        
    data.train_mask = torch.stack(train_mask_list,dim=0)
    data.val_mask = torch.stack(val_mask_list,dim=0)
    data.test_mask = torch.stack(test_mask_list,dim=0)

    return data, dataset.num_features, dataset.num_classes

def get_penn(root: str) -> Tuple[Data, int, int]:
    dataset=load_penn_dataset()
    data=dataset.data
    return data, dataset.num_features, dataset.num_classes

def get_proteins(root: str) -> Tuple[Data, int, int]:
    dataset = PygNodePropPredDataset('ogbn-proteins', f'{root}/OGB', transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    col = data.adj_t.storage.col()
    data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum') # add node features from edge features
    data.adj_t = None
    split_idx = dataset.get_idx_split()
    data.train_mask = index2mask(split_idx['train'], data.num_nodes)
    data.val_mask = index2mask(split_idx['valid'], data.num_nodes)
    data.test_mask = index2mask(split_idx['test'], data.num_nodes)
    return data, data.num_features, 112 # the number of node classes, from ogb examples

def get_pokec(root: str) -> Tuple[Data, int, int]:
    dataset=load_pokec_mat()
    data=dataset.data
    return data, dataset.num_features, dataset.num_classes

def get_data(root: str, name: str) -> Tuple[Data, int, int]:
    if name.lower() == 'photo':
        return  get_photo(root)
    elif name.lower() == 'wikics':
        return  get_wikics (root)
    elif name.lower() == 'reddit2':
        return get_reddit2(root)
    elif name.lower() == 'products':
        return get_products(root)
    elif name.lower()=='squirrel':
        return get_squirrel(root)
    elif name.lower() == 'penn':
        return get_penn(root)
    elif name.lower() == 'proteins':
        return get_proteins(root)
    elif name.lower() == 'pokec':
        return get_pokec(root)
    else:
        raise NotImplementedError
        return