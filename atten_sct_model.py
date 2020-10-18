import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScattterAttentionLayer(nn.Module):
    '''
    Scattering Attention Layer
    '''
    def __init__(self,in_features, out_features, dropout, alpha=0.1):
        super(ScattterAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
#        self.W1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
#        self.W2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
#        self.W3 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W3.data, gain=1.414)
#        self.W4 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W4.data, gain=1.414)
#        self.W5 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W5.data, gain=1.414)
#        self.W6 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W6.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, input,A_nor,P_sct1,P_sct2,P_sct3):
        support0 = torch.mm(input, self.W)
        N = support0.size()[0]
        h_A =  torch.spmm(A_nor, support0)
        h_A2 =  torch.spmm(A_nor, h_A)
        h_A3 =  torch.spmm(A_nor, h_A2)
        h_sct1 = torch.FloatTensor.abs_(torch.spmm(P_sct1, support0))**1
        h_sct2 = torch.FloatTensor.abs_(torch.spmm(P_sct2, support0))**1
        h_sct3 = torch.FloatTensor.abs_(torch.spmm(P_sct3, support0))**1
#        h_A =  torch.spmm(A_nor, torch.mm(input, self.W1))
#        h_A2 = torch.spmm(A_nor,torch.spmm(A_nor, torch.mm(input, self.W2)))
#        h_A3 = torch.spmm(A_nor,torch.spmm(A_nor,torch.spmm(A_nor, torch.mm(input, self.W3))))
#        h_sct1 = torch.FloatTensor.abs_(torch.spmm(P_sct1, torch.mm(input, self.W4)))**1
#        h_sct2 = torch.FloatTensor.abs_(torch.spmm(P_sct2, torch.mm(input, self.W5)))**1
#        h_sct3 = torch.FloatTensor.abs_(torch.spmm(P_sct3, torch.mm(input, self.W6)))**1

        '''
        h_A: N by out_features
        h_A^2: N by out_features
        h_A^3: ...
        h_sct_1: ...
        h_sct_2: ...
        h_sct_3：...
        '''
        h = support0
#        h = torch.spmm(A_nor, torch.mm(input, self.W))
        a_input_A = torch.cat([h,h_A]).view(N, -1, 2 * self.out_features)
        a_input_A2 = torch.cat([h,h_A2]).view(N, -1, 2 * self.out_features)
        a_input_A3 = torch.cat([h,h_A3]).view(N, -1, 2 * self.out_features)
        a_input_sct1 = torch.cat([h,h_sct1]).view(N, -1, 2 * self.out_features)
        a_input_sct2 = torch.cat([h,h_sct2]).view(N, -1, 2 * self.out_features)
        a_input_sct3 = torch.cat([h,h_sct3]).view(N, -1, 2 * self.out_features)
        a_input =  torch.cat((a_input_A,a_input_A2,a_input_A3,a_input_sct1,a_input_sct2,a_input_sct3),1).view(N, 6, -1) 
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        '''
        a_input shape
        e shape: (N,-1)
        attention shape: (N,out_features)
        '''

        attention = F.softmax(e, dim=1).view(N, 6, -1)
        h_all = torch.cat((h_A.unsqueeze(dim=2),h_A2.unsqueeze(dim=2),h_A3.unsqueeze(dim=2),\
                h_sct1.unsqueeze(dim=2),h_sct2.unsqueeze(dim=2),h_sct3.unsqueeze(dim=2)),dim=2).view(N, 6, -1)
        h_prime = torch.mul(attention, h_all) # element eise product
        h_prime = torch.mean(h_prime,1)
#        return h_prime
        return [h_prime,attention]

class ScattterAttentionLayer_mul_a(nn.Module):
    '''
    Scattering Attention Layer
    '''
    def __init__(self,in_features, out_features, dropout, alpha=0.1):
        super(ScattterAttentionLayer_mul_a, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
#        self.W1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
#        self.W2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
#        self.W3 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W3.data, gain=1.414)
#        self.W4 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W4.data, gain=1.414)
#        self.W5 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W5.data, gain=1.414)
#        self.W6 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#        nn.init.xavier_uniform_(self.W6.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a3 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a4 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a5 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a6 = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        nn.init.xavier_uniform_(self.a3.data, gain=1.414)
        nn.init.xavier_uniform_(self.a4.data, gain=1.414)
        nn.init.xavier_uniform_(self.a5.data, gain=1.414)
        nn.init.xavier_uniform_(self.a6.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, input,A_nor,P_sct1,P_sct2,P_sct3):
        support0 = torch.mm(input, self.W)
        N = support0.size()[0]
        h_A =  torch.spmm(A_nor, support0)
        h_A2 =  torch.spmm(A_nor, h_A)
        h_A3 =  torch.spmm(A_nor, h_A2)
        h_sct1 = torch.FloatTensor.abs_(torch.spmm(P_sct1, support0))**1
        h_sct2 = torch.FloatTensor.abs_(torch.spmm(P_sct2, support0))**1
        h_sct3 = torch.FloatTensor.abs_(torch.spmm(P_sct3, support0))**1

#        h_A = self.leakyrelu(h_A)
#        h_A2 = self.leakyrelu(h_A2)
#        h_A2 = self.leakyrelu(h_A3)
#        h_sct1 = self.leakyrelu(h_sct1)
#        h_sct2 = self.leakyrelu(h_sct2)
#        h_sct3 = self.leakyrelu(h_sct3)

#        h_A =  torch.spmm(A_nor, torch.mm(input, self.W1))
#        h_A2 = torch.spmm(A_nor,torch.spmm(A_nor, torch.mm(input, self.W2)))
#        h_A3 = torch.spmm(A_nor,torch.spmm(A_nor,torch.spmm(A_nor, torch.mm(input, self.W3))))
#        h_sct1 = torch.FloatTensor.abs_(torch.spmm(P_sct1, torch.mm(input, self.W4)))**1
#        h_sct2 = torch.FloatTensor.abs_(torch.spmm(P_sct2, torch.mm(input, self.W5)))**1
#        h_sct3 = torch.FloatTensor.abs_(torch.spmm(P_sct3, torch.mm(input, self.W6)))**1
        '''
        h_A: N by out_features
        h_A^2: N by out_features
        h_A^3: ...
        h_sct_1: ...
        h_sct_2: ...
        h_sct_3：...
        '''
        h = support0
        a_input_A = torch.cat([h,h_A]).view(N, -1, 2 * self.out_features)
        a_input_A2 = torch.cat([h,h_A2]).view(N, -1, 2 * self.out_features)
        a_input_A3 = torch.cat([h,h_A3]).view(N, -1, 2 * self.out_features)
        a_input_sct1 = torch.cat([h,h_sct1]).view(N, -1, 2 * self.out_features)
        a_input_sct2 = torch.cat([h,h_sct2]).view(N, -1, 2 * self.out_features)
        a_input_sct3 = torch.cat([h,h_sct3]).view(N, -1, 2 * self.out_features)

        atten_ch1 = torch.matmul(a_input_A,self.a1)
        atten_ch2 = torch.matmul(a_input_A2,self.a2)
        atten_ch3 = torch.matmul(a_input_A3,self.a3)
        atten_ch4 = torch.matmul(a_input_sct1,self.a4)
        atten_ch5 = torch.matmul(a_input_sct2,self.a5)
        atten_ch6 = torch.matmul(a_input_sct3,self.a6)
        a_input = torch.cat((atten_ch1,atten_ch2,atten_ch3,atten_ch4,atten_ch5,atten_ch6),1).view(N, 6, -1) 
        e = a_input.squeeze(2)

#        e = self.leakyrelu(a_input.squeeze(2))


#        a_input =  torch.cat((a_input_A,a_input_A2,a_input_A3,a_input_sct1,a_input_sct2,a_input_sct3),1).view(N, 6, -1) 
#        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        '''
        a_input shape
        e shape: (N,-1)
        attention shape: (N,out_features)
        '''

        attention = F.softmax(e, dim=1).view(N, 6, -1)
        h_all = torch.cat((h_A.unsqueeze(dim=2),h_A2.unsqueeze(dim=2),h_A3.unsqueeze(dim=2),\
                h_sct1.unsqueeze(dim=2),h_sct2.unsqueeze(dim=2),h_sct3.unsqueeze(dim=2)),dim=2).view(N, 6, -1)
        h_prime = torch.mul(attention, h_all) # element eise product
        h_prime = torch.mean(h_prime,1)
#        return h_prime
        return [h_prime,attention]
