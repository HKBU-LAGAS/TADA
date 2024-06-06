import sys
import os
sys.path.append(os.getcwd())
import argparse
import random
import time
import warnings
import yaml
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix, diags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.autograd import grad
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor
import models
from data import get_data
from logger import Logger
import copy
from utils import compute_micro_f1,get_optimizer, se, to_inductive
from ogb.nodeproppred import Evaluator
from torch_geometric.utils import scatter,to_undirected, is_undirected 
##################################################################

MB = 1024 ** 2
GB = 1024 ** 3
OGB = ['proteins','products']

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True,
                    help='the path to the configuration file')
parser.add_argument('--dataset', type=str, required=True,
                    help='the name of the applied dataset')
parser.add_argument('--model', type=str, required=True,
                    help='the name of the applied dataset')
parser.add_argument('--root', type=str, default='~/data')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--grad_norm', type=float, default=None)
parser.add_argument('--inductive', action='store_true')
parser.add_argument('--debug_mem', action='store_true')
parser.add_argument('--test_speed', action='store_true')
parser.add_argument('--weight_dir', help='the weights of each model', default='gnn_reddit.pth')
parser.add_argument('--se_type', type=str, default='rwr',help='the type of structural embedding, ori,rwr,etc,...')
parser.add_argument('--se_dim', type=int, default=128, help='the dimension of structural embedding, the hyper parameter k in paper')
parser.add_argument('--gamma', type=float, default=0.2, help='a factor to weigh the feature and structure embedding, feature weight coe: fw = 1.- gamma')
parser.add_argument('--rho', type=float, default=0.5, help='the sample rate of sparsifier')
parser.add_argument('--beta', type=float, default=1.0, help='the rate of rwr embedding')

""" GNN+TADA is called EbdGNN in the source code"""

def train(model, model_name, optimizer, data, loss_op, grad_norm,state='pre',run=0):
    data.y = data.y.float()
    t1 = time.time()
    model.train()
    optimizer.zero_grad()
    if model_name== 'EbdGNN':
        out = model(data.x, data.sebd, data.adj_t, state)
    else:
        out  = model(data.x, data.adj_t)
    
    if data.train_mask.dim() >1:
        data.y = data.y.long()
        loss = loss_op(out[data.train_mask[run]], data.y[data.train_mask[run]])
    else:
        loss = loss_op(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    if grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
    optimizer.step()
    t2 = time.time()    
    train_time = t2 - t1 
    del data
    return loss.item(), train_time

@torch.no_grad()
def test(model, model_name, data,data_name, evaluator, state='pre',run=0):

    data.y = data.y.float()
    start1 = time.time()
    model.eval()
    if model_name== 'EbdGNN':
        out = model(data.x, data.sebd, data.adj_t, state)
    else:
        out = model(data.x, data.adj_t)
    end = time.time() - start1
    y_true = data.y

    if data_name not in OGB:
        train_acc =  compute_micro_f1(out, y_true, data.train_mask[run])
        valid_acc = compute_micro_f1(out, y_true, data.val_mask[run])
        test_acc = compute_micro_f1(out, y_true, data.test_mask[run])
    else:
        train_acc = evaluator.eval({
            'y_true': data.y[data.train_mask],
            'y_pred': out[data.train_mask],
        })['rocauc']
        valid_acc = evaluator.eval({
            'y_true': data.y[data.val_mask],
            'y_pred': out[data.val_mask],
        })['rocauc']
        test_acc = evaluator.eval({
            'y_true': data.y[data.test_mask],
            'y_pred': out[data.test_mask],
        })['rocauc']

    del data
    out1 = out.clone()

    return train_acc, valid_acc, test_acc, out1, end


def main():
    global args

    args = parser.parse_args()

    with open(args.cfg, 'r') as fp:
        model_config = yaml.load(fp, Loader=yaml.FullLoader)
        name = model_config['name']
        model_config = model_config['params'][args.dataset]
        model_config['name'] = name
        model_config['device']='cuda:'+str(args.gpu)

    print(f'model config: {model_config}')
    print(f'clipping grad norm: {args.grad_norm}')

    if model_config['name'] not in ['GAT', 'SGC', 'APPNP']:
        args.model = model_config['arch_name']
    else:
        args.model = model_config['name']
    assert model_config['name'] in ['GCN', 'GCN2', 'EbdGNN', 'GAT', 'SGC', 'APPNP']

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print("Use GPU {} for training".format(args.gpu))
    
    torch.cuda.set_device(args.gpu)
    data, num_features, num_classes = get_data(args.root, args.dataset)

    if model_config['name'] == 'EbdGNN':
        data = se(data, args)
        data.sebd = torch.tensor(data.sebd.copy()).float().to(args.gpu)
    
    if args.dataset == 'squirrel' or args.dataset == 'photo':
        data.edge_index = to_undirected(data.edge_index)

    data = T.ToSparseTensor()(data.to('cuda'))
    data = data.to('cuda')
    
    GNN = getattr(models, args.model)

    loss_op = F.cross_entropy
    if args.dataset == 'proteins':
        loss_op = torch.nn.BCEWithLogitsLoss()

    if model_config['loop']:
        t = time.perf_counter()
        print('Adding self-loops...', end=' ', flush=True)
        data.adj_t = data.adj_t.set_diag()

    if model_config['norm']:
        t = time.perf_counter()
        print('Normalizing data...', end=' ', flush=True)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    if args.inductive:
        print('inductive learning mode')
        data = to_inductive(data)
    
    logger = Logger(args.runs, args)
    infer_time_list = []
    test_acc_list = []
    train_time_list = []

    evaluator = None 
    if args.dataset in OGB:
        evaluator = Evaluator(name= 'ogbn-'+args.dataset )

    adj_t = copy.deepcopy(data.adj_t)
    for run in range(args.runs):
        data.adj_t = adj_t
        
        if args.model == 'EbdGNN':
            model = GNN(in_channels=num_features, in_channels3=args.se_dim,
                        out_channels=num_classes, device=model_config['device'], gamma=args.gamma,
                        gnn_type=model_config['gnn_type'], **model_config['architecture']).cuda(args.gpu)
        else:
            model = GNN(in_channels=num_features, out_channels=num_classes, **model_config['architecture']).cuda(args.gpu)
        optimizer = get_optimizer(model_config, model)
        best_val_acc = 0.0
        patience = 0

        state = 'pre'

        dir = args.weight_dir
        if os.path.exists(dir):
            checkpoint = torch.load(dir)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_epoch = 0 
        for epoch in range(1, 1 + model_config['epochs']):
            
            loss, train_time = train(model, model_config['name'], optimizer, data, loss_op, args.grad_norm, state=state,run=run)
            result = test(model,  model_config['name'], data, args.dataset, evaluator=evaluator, state=state,run=run)
            train_acc, valid_acc, test_acc, pred, infer_time = result
            result = (train_acc, valid_acc, test_acc)

            if valid_acc > best_val_acc:
                patience = 0
                best_val_acc = valid_acc
                result_t = test(model,  model_config['name'], data, args.dataset, evaluator, state=state,run=run)
                _, _, test_acc, best_pred, infer_time = result_t
                best_epoch = epoch
            else:
                patience += 1
                if patience > 100:
                    # reddit2 400; other 100
                    if model_config['name'] == 'EbdGNN':
                        if epoch > model_config['pepochs'] + 128:
                            break
                    else:
                        break
            
            logger.add_result(run, result)
            if model_config['name'] == 'EbdGNN' and epoch == model_config['pepochs']:
                state = 'train'
                print("start sparse")
                similarity = model.node_similarity(best_pred, data)
                data = model.graph_sparse(similarity, data, args)
                torch.cuda.empty_cache()
            
            if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or  model_config['name'] != 'EbdGNN':
                train_time_list.append(train_time)
                infer_time_list.append(infer_time)

            if (epoch % 2) == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss}'
                      f'Train f1: {100 * train_acc:.2f}%, '
                      f'Valid f1: {100 * valid_acc:.2f}% '
                      f'Test f1: {100 * test_acc:.2f}%')
        
        test_acc_list.append(test_acc) 
        logger.add_result(run, result) 
        logger.print_statistics(run) 
        print("best epoch is {}".format(best_epoch))
    
    logger.print_statistics()
    test_acc_mean = np.array(test_acc_list).mean()
    test_acc_std = np.array(test_acc_list).std()
    train_time_mean = np.array(train_time_list).mean()
    infer_time_mean = np.array(infer_time_list).mean() 

    if torch.cuda.is_available():
        print("Max GPU memory usage: {:.5f} GB, max GPU memory cache {:.5f} GB".format(
            torch.cuda.max_memory_allocated(args.gpu) / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)))
    

if __name__ == '__main__':
    main()
