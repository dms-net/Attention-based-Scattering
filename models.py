import torch.nn as nn
import torch 
import torch.nn.functional as F
from layers import GC_withres,NGCN
from atten_sct_model import ScattterAttentionLayer,ScattterAttentionLayer_mul_a
class SCT_GAT(nn.Module):
    def __init__(self, nfeat, hid, nclass, dropout, nheads,smoo):
        super(SCT_GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.attentions = [ScattterAttentionLayer_mul_a(nfeat, hid, dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.gc11 = GC_withres(hid*nheads, nclass,smooth=smoo)
    def forward(self,x,adj_p,A_tilde,s1_sct,s2_sct,s3_sct):
        x = F.dropout(x, self.dropout, training=self.training)
        for i,att in enumerate(self.attentions):
            torch.save(att(x,A_tilde,s1_sct,s2_sct,s3_sct)[1],'Attention_dir/cite_attention_%d.pt'%i)
        x = torch.cat([torch.nn.ReLU()(att(x,A_tilde,s1_sct,s2_sct,s3_sct)[0]) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc11(x,adj_p)
        return F.log_softmax(x, dim=1)


class SCT_GAT_wikics(nn.Module):
    def __init__(self, nfeat, hid, nclass, dropout, nheads,smoo):
        super(SCT_GAT_wikics, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.bn0 = nn.BatchNorm1d(nfeat)
        self.bn1 = nn.BatchNorm1d(hid)
        self.attentions = [ScattterAttentionLayer(nfeat, hid, dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.gc11 = GC_withres(hid* nheads, nclass,smooth=smoo)
    def forward(self,x,adj_p,A_tilde,s1_sct,s2_sct,s3_sct):
        x = self.bn0(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for i,att in enumerate(self.attentions):
            torch.save(att(x,A_tilde,s1_sct,s2_sct,s3_sct)[1],'Attention_dir/dblp_attention_%d.pt'%i)
        x = torch.cat([torch.nn.ReLU()(self.bn1(att(x,A_tilde,s1_sct,s2_sct,s3_sct)[0])) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc11(x,adj_p)
        return F.log_softmax(x, dim=1)

