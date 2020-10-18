from __future__ import division
from __future__ import print_function
from utils import load_citation, accuracy
import time
import argparse
import numpy as np
from scipy import sparse
from atten_sct_model import ScattterAttentionLayer 
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import SCT_GAT
from torch.optim.lr_scheduler import StepLR
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora",help='Dataset to use.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-6,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1', type=float, default=0.0,
                    help='Weight decay (L1 loss on parameters).')
parser.add_argument('--hid', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--nheads', type=int, default=8,
                    help='Number of heads in attention mechism.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['AugNormAdj'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='Smooth for Res layer')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj,adj_p,A_tilde,adj_sct1,adj_sct2,adj_sct4,features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization,args.cuda)
# Model and optimizer

with open("HOMO_DIR/%s_feature"%args.dataset, "w") as output:
    for item in labels.cpu():
        output.write("%.4f\n" % item)

#from torch.optim.lr_scheduler import StepLR
model = SCT_GAT(features.shape[1],args.hid,labels.max().item()+1,dropout=args.dropout,nheads=args.nheads,smoo=args.smoo)


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)

if args.cuda:
    model = model.cuda()
    features = features.cuda()
    A_tilde = A_tilde.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
#scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
# import EarlyStopping
from pytorchtools import EarlyStopping

acc_val_list = []
def train(epoch):
    global valid_error
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,adj,A_tilde,adj_sct1,adj_sct2,adj_sct4)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features,adj,A_tilde,adj_sct1,adj_sct2,adj_sct4)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    acc_val_list.append(acc_val.item())
    valid_error = 1.0 - acc_val.item()
    return loss_val.data.item()

    # record

def test():
    model.eval()
    output = model(features,adj,A_tilde,adj_sct1,adj_sct2,adj_sct4)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    acc_val_list.append(acc_test.item())
# Train model
t_total = time.time()

patience = 100
early_stopping = EarlyStopping(patience=patience, verbose=True)

for epoch in range(args.epochs):
    train(epoch)
    scheduler.step()
#    print('Epoch:', epoch,'LR:', scheduler.get_lr())
#    print(valid_error)
#    early_stopping(valid_error, model)
#    if early_stopping.early_stop:
#        print("Early stopping")
#        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print('Smooth is %.5f'%(args.smoo))
# Testing
test()

