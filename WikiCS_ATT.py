import os.path as osp
import scipy.sparse as sp
import argparse
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets.wikics import WikiCS
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric.transforms as T
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'WikiCS')
from torch_geometric.utils import to_scipy_sparse_matrix
from utils import normalize_adjacency_matrix,normalizemx
from utils import normalize_adjacency_matrix,accuracy,scattering1st,sparse_mx_to_torch_sparse_tensor
from layers import GC_withres
import torch.optim as optim
import numpy as np
import time
from torch_geometric.transforms import TargetIndegree
from models import SCT_ATTEN,SCT_GAT_wikics



### use gcn
from torch_geometric.nn import GCNConv, ChebConv  # noqa


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1', type=float, default=0.0,
                    help='Weight decay (L1 loss on parameters).')
parser.add_argument('--hid', type=int, default=60,
                    help='Number of hidden units.')
parser.add_argument('--nheads', type=int, default=10,
                    help='Number of heads in attention mechism.')
parser.add_argument('--data_spilt', type=int, default=0)
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['AugNormAdj'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='Smooth for Res layer')
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    print('Traning on Cuda,yeah!')
    torch.cuda.manual_seed(args.seed)


#from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import MultiStepLR,StepLR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = WikiCS(path,transform=T.TargetIndegree())
data = dataset[0]
# Num of feat:1639
adj = to_scipy_sparse_matrix(edge_index = data.edge_index)
adj = adj+ adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#A_tilde = sparse_mx_to_torch_sparse_tensor(normalize_adjacency_matrix(adj,sp.eye(adj.shape[0]))).to(device)
A_tilde = normalize_adjacency_matrix(adj,sp.eye(adj.shape[0]))
adj_p = normalizemx(adj)
#adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
features = data.x
features = torch.FloatTensor(np.array(features))
labels = data.y 
#print(labels)
#print('Max')
#print(labels.max()+1)
labels = torch.LongTensor(np.array(labels))
#print(features.shape)
#print("========================")
#print('Loading')
adj_sct1 = scattering1st(adj_p,1)
adj_sct2 = scattering1st(adj_p,2)
adj_sct4 = scattering1st(adj_p,4)
adj_p = sparse_mx_to_torch_sparse_tensor(adj_p)
A_tilde = sparse_mx_to_torch_sparse_tensor(A_tilde)
idx_train = data.train_mask[:,args.data_spilt] ## there are 20 different splits, here we select the first/default one
#print(idx_train)
idx_val = data.val_mask[:,args.data_spilt]
idx_test = data.test_mask
np.savetxt('Attention_dir/data_spilt.txt',idx_val, fmt="%5i")
model = SCT_GAT_wikics(features.shape[1],args.hid,10,dropout=args.dropout,nheads=args.nheads,smoo=args.smoo)
# run homophily
adj = sparse_mx_to_torch_sparse_tensor(adj)
#from utils import new_homophily
#homo_list = new_homophily(adj,labels)
#print('Total Same classes:--------------')
#print(homo_list)
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data.shape)
if args.cuda:
    model = model.cuda()
    features = features.cuda()
    A_tilde = A_tilde.cuda()
    adj_p = adj_p.cuda()
    labels = labels.cuda()
#    idx_train = idx_train.cuda()
#    idx_val = idx_val.cuda()
#    idx_test = idx_test.cuda()




#from utils import homophily
#homo_list = homophily(adj,labels)
#with open("HOMO_DIR/wikics", "w") as output:
#    for item in homo_list:
#        output.write("%.4f\n" % item)

optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=50, gamma=0.8)
acc_val_list = []
def train(epoch):
    global valid_error
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,adj_p,A_tilde,adj_sct1,adj_sct2,adj_sct4)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features,adj_p,A_tilde,adj_sct1,adj_sct2,adj_sct4)
#    model.eval()
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))
    acc_val_list.append(acc_val.item())
    valid_error = 1.0 - acc_val.item()
    return loss_val.data.item()
def test():
    model.eval()
    output = model(features,adj_p,A_tilde,adj_sct1,adj_sct2,adj_sct4)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print("Vali set results:",
          "loss= {:.4f}".format(loss_val.item()),
          "accuracy= {:.4f}".format(acc_val.item()))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    acc_val_list.append(acc_test.item())
from pytorchtools import EarlyStopping
patience = args.patience
early_stopping = EarlyStopping(patience=patience, verbose=True)
# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
    scheduler.step()
#    early_stopping(valid_error, model)
#    if early_stopping.early_stop:
#       print("Early stopping")
#       break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print('Smooth is %.5f'%(args.smoo))
model.eval()
# Testing
test()
