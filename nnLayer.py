from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from collections import OrderedDict
from torch import nn as nn, einsum
from einops import rearrange
from math import floor, ceil

class MLP_old(nn.Module):
    def __init__(self, inSize, outSize, hiddenSizeList=[], dropout=0.0, name='MLP', actFunc=nn.ReLU, bn=False, outAct=False):
        super(MLP_old, self).__init__()
        self.name = name
        self.bn = nn.BatchNorm1d(inSize) if bn else None
        layers = nn.Sequential()
        for i,os in enumerate(hiddenSizeList):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), actFunc())
            inSize = os
        self.hiddenLayers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(inSize, outSize)
        self.outAct = actFunc() if outAct else None
    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.hiddenLayers(x)
        x = self.out(self.dropout(x))
        if self.outAct is not None:
            x = self.outAct(x)
        return x

class TextEmbedding_old(nn.Module):
    def __init__(self, embeding, freeze=False, dropout=0.2, name='textEmbedding'):
        super(TextEmbedding_old, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeding, dtype=torch.float32), freeze=freeze)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        return self.dropout(self.embedding(x))

class TextSPP(nn.Module):
    def __init__(self, size=128, name='textSpp'):
        super(TextSPP, self).__init__()
        self.name = name
        self.spp = nn.AdaptiveAvgPool1d(size)
    def forward(self, x):
        return self.spp(x)

class TextSPP2(nn.Module):
    def __init__(self, size=128, name='textSpp2'):
        super(TextSPP2, self).__init__()
        self.name = name
        self.spp1 = nn.AdaptiveMaxPool1d(size)
        self.spp2 = nn.AdaptiveAvgPool1d(size)
    def forward(self, x):
        x1 = self.spp1(x).unsqueeze(dim=3) # => batchSize × feaSize × size × 1
        x2 = self.spp2(x).unsqueeze(dim=3) # => batchSize × feaSize × size × 1
        x3 = -self.spp1(-x).unsqueeze(dim=3) # => batchSize × feaSize × size × 1
        return torch.cat([x1,x2,x3], dim=3) # => batchSize × feaSize × size × 3

class TextEmbedding(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding,dtype=torch.float32), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=dropout/2)
        self.dropout2 = nn.Dropout(p=dropout/2)
        self.p = dropout
    def forward(self, x):
        # x: batchSize × seqLen
        if self.p>0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x

class TextEmbedding_1d(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding_1d, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding,dtype=torch.float32), freeze=freeze)
        # self.dropout1 = nn.Dropout2d(p=dropout/2)
        self.dropout = nn.Dropout(p=dropout/2)
    def forward(self, x):
        return self.dropout(self.embedding(x))


        
class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=64, dropout=0.15, name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
                        nn.ELU(),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                        nn.InstanceNorm1d(filterSize),
                        nn.ELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                        nn.InstanceNorm1d(filterSize),
                    )
        self.name = name
    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return x + self.layers(x)

class ResDilaCNNBlocks(nn.Module):
    def __init__(self, feaSize, filterSize, blockNum=10, dilaSizeList=[1,2,4,8,16], dropout=0.15, name='ResDilaCNNBlocks'):
        super(ResDilaCNNBlocks, self).__init__()
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize,filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(dilaSizeList[i%len(dilaSizeList)],filterSize,dropout=dropout))
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x) # => batchSize × seqLen × filterSize
        x = self.blockLayers(x.transpose(1,2)).transpose(1,2) # => batchSize × seqLen × filterSize
        return F.elu(x) # => batchSize × seqLen × filterSize

class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name
    def forward(self, x):
        return self.bn(x)

# class TextCNN(nn.Module):
#     def __init__(self, featureSize, filterSize, contextSizeList, reduction='pool', actFunc=nn.ReLU, bn=False, name='textCNN'):
#         super(TextCNN, self).__init__()
#         moduleList = []
#         for i in range(len(contextSizeList)):
#             moduleList.append(
#                 nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
#             )
#         self.actFunc = actFunc()
#         self.conv1dList = nn.ModuleList(moduleList)
#         self.reduction = reduction
#         self.batcnNorm = nn.BatchNorm1d(filterSize)
#         self.bn = bn
#         self.name = name
#     def forward(self, x):
#         # x: batchSize × seqLen × feaSize
#         x = x.transpose(1,2) # => batchSize × feaSize × seqLen
#         x = [conv(x).transpose(1,2) for conv in self.conv1dList] # => scaleNum * (batchSize × seqLen × filterSize)

#         if self.bn:
#             x = [self.batcnNorm(i) for i in x]
#         x = [self.actFunc(i) for i in x]

#         if self.reduction=='pool':
#             x = [F.adaptive_max_pool1d(i.transpose(1,2), 1).squeeze(dim=2) for i in x]
#             return torch.cat(x, dim=1) # => batchSize × scaleNum*filterSize
#         elif self.reduction=='cpool':
#             x = torch.cat([i.unsqueeze(dim=3) for i in x], dim=3)
#             return torch.max(x, 3)[0] # => batchSize × seqLen × filterSize
#         elif self.reduction=='none':
#             return x # => scaleNum * (batchSize × seqLen × filterSize)

# 使用Tanh激活
class TextCNN(nn.Module):
    def __init__(self, featureSize, filterSize, contextSizeList, reduction='pool', actFunc=nn.ReLU, bn=False, name='textCNN'):
        super(TextCNN, self).__init__()
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
            )
        # self.actFunc = actFunc()
        self.conv1dList = nn.ModuleList(moduleList)
        self.reduction = reduction
        self.batcnNorm = nn.BatchNorm1d(filterSize)
        self.bn = bn
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x).transpose(1,2) for conv in self.conv1dList] # => scaleNum * (batchSize × seqLen × filterSize)

        if self.bn:
            x = [self.batcnNorm(i) for i in x]
        x = [F.tanh(i) for i in x]

        if self.reduction=='pool':
            x = [F.adaptive_max_pool1d(i.transpose(1,2), 1).squeeze(dim=2) for i in x]
            return torch.cat(x, dim=1) # => batchSize × scaleNum*filterSize
        elif self.reduction=='cpool':
            x = torch.cat([i.unsqueeze(dim=3) for i in x], dim=3)
            return torch.max(x, 3)[0] # => batchSize × seqLen × filterSize
        elif self.reduction=='none':
            return x # => scaleNum * (batchSize × seqLen × filterSize)

class TextLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, bidirectional=True, name='textBiLSTM'):
        super(TextLSTM, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=bidirectional, batch_first=True, num_layers=num_layers, dropout=dropout)

    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biLSTM(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2
    def orthogonalize_gate(self):
        nn.init.orthogonal_(self.biLSTM.weight_ih_l0)
        nn.init.orthogonal_(self.biLSTM.weight_hh_l0)
        nn.init.ones_(self.biLSTM.bias_ih_l0)
        nn.init.ones_(self.biLSTM.bias_hh_l0)

class TextGRU(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, bidirectional=True, name='textBiGRU'):
        super(TextGRU, self).__init__()
        self.name = name
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=bidirectional, batch_first=True, num_layers=num_layers, dropout=dropout)

    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biGRU(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]

        return output # output: batchSize × seqLen × hiddenSize*2

class FastText(nn.Module):
    def __init__(self, feaSize, name='fastText'):
        super(FastText, self).__init__()
        self.name = name
    def forward(self, x, xLen):
        # x: batchSize × seqLen × feaSize; xLen: batchSize
        x = torch.sum(x, dim=1) / xLen.float().view(-1,1)
        return x

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, outBn=False, outAct=False, outDp=False, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        hiddens,bns = [],[]
        for i,os in enumerate(hiddenList):
            hiddens.append( nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
    def forward(self, x):
        for h,bn in zip(self.hiddens,self.bns):
            x = h(x)
            if self.bnEveryLayer:
                if len(x.shape)==3:
                    x = bn(x.transpose(1,2)).transpose(1,2)
                else:
                    x = bn(x)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn: x = self.bns[-1](x)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x

class GCN(nn.Module):
    def __init__(self, inSize, outSize, hiddenSizeList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, outBn=False, outAct=False, outDp=False, resnet=False, name='GCN', actFunc=nn.ReLU):
        super(GCN, self).__init__()
        self.name = name
        hiddens,bns = [],[]
        for i,os in enumerate(hiddenSizeList):
            hiddens.append(nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet
    def forward(self, x, L):
        # x: nodeNum × feaSize; L: batchSize × nodeNum × nodeNum
        for h,bn in zip(self.hiddens,self.bns):
            a = h(torch.matmul(L,x)) # => batchSize × nodeNum × os
            if self.bnEveryLayer:
                if len(L.shape)==3:
                    a = bn(a.transpose(1,2)).transpose(1,2)
                else:
                    a = bn(a)
            a = self.actFunc(a)
            if self.dpEveryLayer:
                a = self.dropout(a)
            if self.resnet and a.shape==x.shape:
                a += x
            x = a
        a = self.out(torch.matmul(L, x)) # => batchSize × nodeNum × outSize
        if self.outBn:
            if len(L.shape)==3:
                a = self.bns[-1](a.transpose(1,2)).transpose(1,2)
            else:
                a = self.bns[-1](a)
        if self.outAct: a = self.actFunc(a)
        if self.outDp: a = self.dropout(a)
        if self.resnet and a.shape==x.shape:
            a += x
        x = a
        return x

class TextAttention(nn.Module):
    def __init__(self, method, name='textAttention'):
        super(TextAttention, self).__init__()
        self.attn = LuongAttention(method)
        self.name = name
    def forward(self, sequence, reference):
        # sequence: batchSize × seqLen × feaSize; reference: batchSize × classNum × feaSize
        alpha = self.attn(reference, sequence) # => batchSize × classNum × seqLen
        return torch.matmul(alpha, sequence) # => batchSize × classNum × feaSize

class ICDAttention(nn.Module):
    def __init__(self, inSize, classNum, transpose=False, name='ICDAttention'):
        super(ICDAttention, self).__init__()
        self.transpose = transpose
        self.U = nn.Linear(inSize, classNum)
        self.name = name
    def forward(self, X):
        # X: batchSize × seqLen × inSize
        alpha = F.softmax(self.U(X), dim=1) # => batchSize × seqLen × classNum
        X = torch.matmul(X.transpose(1,2), alpha) # => batchSize × inSize × classNum
        return X.transpose(1,2)

class LAATAttention(nn.Module):
    def __init__(self, inSize, classNum,d_a, transpose=False, name='LAATAttention'):
        super(LAATAttention, self).__init__()
        self.transpose = transpose
        self.first_linears = nn.Linear(inSize, d_a)
        self.U = nn.Linear(d_a, classNum)
        self.name = name
    def forward(self, X):
        # X: batchSize × seqLen × inSize
        Z = F.tanh(self.first_linears(X))
        alpha = F.softmax(self.U(Z), dim=1) # => batchSize × seqLen × classNum
        X = torch.matmul(X.transpose(1,2), alpha) # => batchSize × inSize × classNum
        return X.transpose(1,2)

# 从Linear的weight里面取 1000个位置的weight
class ICDCandiAttention(nn.Module):
    def __init__(self, inSize, classNum, transpose=False, name='ICDCandiAttention'):
        super(ICDCandiAttention, self).__init__()
        self.transpose = transpose
        self.U = nn.Linear(inSize, classNum)
        self.name = name
    def forward(self, X, candidate):
        # X: batchSize × seqLen × inSize
        batchLabelVec =torch.stack([self.U.weight[candidate[i]] for i in range(len(candidate))], dim=0) # => batchSize × Candidates num(1000) × inSize        
        alpha = F.softmax(torch.matmul(X, batchLabelVec.transpose(1,2)), dim=1) # => batchSize × seqLen × Candidates num(1000)
        X = torch.matmul(alpha.transpose(1,2),X) # => batchSize x candidateNum x inSize
        return X

class DeepICDAttention(nn.Module):
    def __init__(self, inSize, classNum, hdnDropout=0.1, attnList=[], compress=False, name='DeepICDAttn'):
        super(DeepICDAttention, self).__init__()
        hdns,attns,bns = [],[],[]
        if not compress:
            attnList = attnList + [classNum]
        else:
            self.decode = nn.Sequential(
                            nn.Linear(attnList[-1], classNum),
                            nn.BatchNorm1d(inSize),
                            nn.ReLU()
                            )
        for os in attnList:
            hdns.append(nn.Linear(inSize,inSize))
            attns.append(nn.Linear(inSize,os))
            bns.append(nn.BatchNorm1d(inSize))
        self.hdns = nn.ModuleList(hdns)
        self.attns = nn.ModuleList(attns)
        self.bns = nn.ModuleList(bns)
        self.dropout = nn.Dropout(p=hdnDropout)
        self.compress = compress
        self.name = name
    def forward(self, X):
        # X: batchSize × seqLen × inSize
        for h,a,b in zip(self.hdns,self.attns,self.bns):
            alpha = F.softmax(a(X), dim=1) # => batchSize × seqLen × os
            X = torch.matmul(alpha.transpose(1,2), X) # => batchSize × os × inSize
            X = h(X) # => batchSize × os × inSize
            X = b(X.transpose(1,2)).transpose(1,2) # => batchSize × os × inSize
            X = F.relu(X) # => batchSize × os × inSize
            X = self.dropout(X) # => batchSize × os × inSize
        if self.compress:
            X = self.decode(X.transpose(1,2)).transpose(1,2)
        # => batchSize × classNum × inSize
        return X

class DeepICDDescAttention(nn.Module):
    def __init__(self, inSize, classNum, labSize=1024, hdnDropout=0.1, attnList=[], labDescVec=None, name='DeepICDAttn'):
        super(DeepICDDescAttention, self).__init__()
        hdns,attns,bns = [],[],[]
        for i,os in enumerate(attnList):
            attns.append(nn.Linear(inSize,os))
            if i==len(attnList)-1:
                hdns.append(nn.Linear(inSize, labSize))
                inSize = labSize
            else:
                hdns.append(nn.Linear(inSize,inSize))
            bns.append(nn.BatchNorm1d(inSize))
        self.hdns = nn.ModuleList(hdns)
        self.attns = nn.ModuleList(attns)
        self.bns = nn.ModuleList(bns)
        self.dropout = nn.Dropout(p=hdnDropout)
        self.labDescVec = nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32)) if labDescVec is not None else None
        self.name = name
    def forward(self, X, labDescVec=None):
        if labDescVec is None:
            labDescVec = self.labDescVec
        # X: batchSize × seqLen × inSize
        for h,a,b in zip(self.hdns,self.attns,self.bns):
            alpha = F.softmax(a(X), dim=1) # => batchSize × seqLen × os
            X = torch.matmul(alpha.transpose(1,2), X) # => batchSize × os × inSize
            X = h(X) # => batchSize × os × inSize
            X = b(X.transpose(1,2)).transpose(1,2) # => batchSize × os × inSize
            X = F.relu(X) # => batchSize × os × inSize
            X = self.dropout(X) # => batchSize × os × inSize
        # X => batchSize × os × icdSize         labDescVec => icdSize × 1000
        alpha = F.softmax(torch.matmul(X, labDescVec.transpose(0,1)), dim=1) # => batchSize × os × classNum
        X = torch.matmul(alpha.transpose(1,2), X) # => batchSize × classNum × inSize
        return X

class DeepICDDescCandiAttention(nn.Module):
    def __init__(self, inSize, classNum, labSize=1024, hdnDropout=0.1, attnList=[], labDescVec=None, name='DeepICDDescCandiAttention'):
        super(DeepICDDescCandiAttention, self).__init__()
        hdns,attns,bns = [],[],[]
        for i,os in enumerate(attnList):
            attns.append(nn.Linear(inSize,os))
            if i==len(attnList)-1:
                hdns.append(nn.Linear(inSize, labSize))
                inSize = labSize
            else:
                hdns.append(nn.Linear(inSize,inSize))
            bns.append(nn.BatchNorm1d(inSize))
        self.hdns = nn.ModuleList(hdns)
        self.attns = nn.ModuleList(attns)
        self.bns = nn.ModuleList(bns)
        self.dropout = nn.Dropout(p=hdnDropout)
        self.labDescVec = nn.Parameter(torch.tensor(labDescVec, dtype=torch.float32)) if labDescVec is not None else None
        self.name = name
    def forward(self, X, candidate, labDescVec=None):
        if labDescVec is None:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            labDescVec = self.labDescVec
        # X:            batchSize × seqLen × inSize
        # labDescVec:   Class num × 1024      tensor
        # 拿到当前batch 对应的candidates       np.array 索引
        for h,a,b in zip(self.hdns,self.attns,self.bns):
            alpha = F.softmax(a(X), dim=1) # => batchSize × seqLen × os
            X = torch.matmul(alpha.transpose(1,2), X) # => batchSize × os × inSize
            X = h(X) # => batchSize × os × inSize
            X = b(X.transpose(1,2)).transpose(1,2) # => batchSize × os × inSize
            X = F.relu(X) # => batchSize × os × inSize
            X = self.dropout(X) # => batchSize × os × inSize
        # => batchSize × os × icdSize
        # 根据Candidates 取出 每个batch对应的标签矩阵 batchSize × 1000 × 1024
        # 源：    labDescVec:   Class num × 1024      np.array  二维
        # 目标:                 batchSize x 1000 x1024          三维（二维concat）
        batchLabelVec =torch.stack([labDescVec[candidate[i]] for i in range(len(candidate))], dim=0) # => batchSize × Candidates num(1000) × 1024
        alpha = F.softmax(torch.matmul(X, batchLabelVec.transpose(1,2)), dim=1) # => batchSize × os × Candidates num(1000)
        X = torch.matmul(alpha.transpose(1,2), X) # => batchSize × Candidates num(1000) × inSize
        return X
    
class LuongAttention(nn.Module):
    def __init__(self, method):
        super(LuongAttention, self).__init__()
        self.method = method
    def dot_score(self, hidden, encoderOutput):
        # hidden: batchSize × classNum × hiddenSize; encoderOutput: batchSize × seq_len × hiddenSize
        return torch.matmul(encoderOutput, hidden.transpose(-1,-2)) # => batchSize × seq_len × classNum
    def forward(self, hidden, encoderOutput):
        attentionScore = self.dot_score(hidden, encoderOutput).transpose(-1,-2)
        # attentionScore: batchSize × classNum × seq_len
        return F.softmax(attentionScore, dim=-1) # => batchSize × classNum × seq_len

class SimpleAttention(nn.Module):
    def __init__(self, inSize, actFunc=nn.Tanh(), name='SimpleAttention'):
        super(SimpleAttention, self).__init__()
        self.name = name
        self.W = nn.Linear(inSize, int(inSize//2))
        self.U = nn.Linear(int(inSize//2), 1)
        self.actFunc = actFunc
    def forward(self, input):
        # input: batchSize × seqLen × inSize
        x = self.W(input) # => batchSize × seqLen × inSize//2
        H = self.actFunc(x) # => batchSize × seqLen × inSize//2
        alpha = F.softmax(self.U(H), dim=1) # => batchSize × seqLen × 1
        return self.actFunc( torch.matmul(input.transpose(1,2), alpha).squeeze(2) ) # => batchSize × inSize

class KnowledgeAttention(nn.Module):
    def __init__(self, noteFeaSize, titleFeaSize, name='knowledgeAttention'):
        super(KnowledgeAttention, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(noteFeaSize, titleFeaSize),
            nn.Tanh()
            )
        self.labWeight = None
        self.name = name
    def forward(self, noteConved, titleEncoded):
        # noteConved: batchSize × noteFeaSize; titleEncoded: titleNum × titleFeaSize
        x = self.linear(noteConved) # => batchSize × titleFeaSize
        attnWeight = F.softmax(torch.matmul(x, titleEncoded.transpose(0,1)), dim=1) # => batchSize × titleNum
        self.labWeight = attnWeight.detach().cpu().numpy()
        return torch.matmul(attnWeight, titleEncoded) # => batchSize × titleFeaSize

class InterationAttention(nn.Module):
    def __init__(self, feaSize1, feaSize2, dropout=0.0, attnType='poolAttn', name='interAttn'):
        super(InterationAttention, self).__init__()
        self.attnFunc = {'poolAttn':self.pooling_attention,
                         'poolAttn_s':self.pooling_attention_s,
                         'catSimAttn':self.concat_simple_attention,
                         'plaAttn':self.plane_attention,
                         'plaAttn_s':self.plane_attention_s}
        assert attnType in self.attnFunc.keys()
        self.name = name
        self.U = nn.Linear(feaSize1, feaSize2)
        self.W = nn.Linear(feaSize2, 1)
        self.simpleAttn1 = SimpleAttention(feaSize1+feaSize2)
        self.simpleAttn2 = SimpleAttention(feaSize1+feaSize2)
        self.feaSize1,self.feaSize2 = feaSize1,feaSize2
        self.attnType = attnType
        self.dropout = nn.Dropout(dropout)

    def pooling_attention_s(self, x, y):
        u = self.U(x).unsqueeze(dim=2) # => batchSize × seqLen1 × 1 × feaSize2
        v = y.unsqueeze(dim=1) # => batchSize × 1 × seqLen2 × feaSize2
        alpha = torch.sum(u*v,dim=3) # => batchSize × seqLen1 × seqLen2
        xAlpha,_ = torch.max(alpha, dim=2, keepdim=True) # => batchSize × seqLen1 × 1
        x = torch.matmul(x.transpose(1,2), F.softmax(xAlpha,dim=1)).squeeze(dim=2) # => batchSize × feaSize1
        yAlpha,_ = torch.max(alpha, dim=1, keepdim=True) # => batchSize × 1 × seqLen2
        y = torch.matmul(F.softmax(yAlpha,dim=2), y).squeeze(dim=1) # => batchSize × feaSize2
        return torch.cat([x,y], dim=1) # => batchSize × (feaSize1+feaSize2)

    def pooling_attention(self, x, y):
        u = self.U(x).unsqueeze(dim=2) # => batchSize × seqLen1 × 1 × feaSize2
        v = y.unsqueeze(dim=1) # => batchSize × 1 × seqLen2 × feaSize2
        alpha = F.tanh(u*v) # => batchSize × seqLen1 × seqLen2 × feaSize2
        alpha = self.W(alpha).squeeze(dim=3) # => batchSize × seqLen1 × seqLen2
        xAlpha,_ = torch.max(alpha, dim=2, keepdim=True) # => batchSize × seqLen1 × 1
        x = torch.matmul(x.transpose(1,2), F.softmax(xAlpha,dim=1)).squeeze(dim=2) # => batchSize × feaSize1
        yAlpha,_ = torch.max(alpha, dim=1, keepdim=True) # => batchSize × 1 × seqLen2
        y = torch.matmul(F.softmax(yAlpha,dim=2), y).squeeze(dim=1) # => batchSize × feaSize2
        return torch.cat([x,y], dim=1) # => batchSize × (feaSize1+feaSize2)

    def concat_simple_attention(self, x, y):
        x_pooled,_ = torch.max(x, dim=1) # => batchSize × feaSize1
        y_pooled,_ = torch.max(y, dim=1) # => batchSize × feaSize2
        u = torch.cat([x, y_pooled.unsqueeze(dim=1).expand(-1,x.shape[1],-1)], dim=2) # => batchSize × seqLen1 × (feaSize1+feaSize2)
        v = torch.cat([y, x_pooled.unsqueeze(dim=1).expand(-1,y.shape[1],-1)], dim=2) # => batchSize × seqLen2 × (feaSize1+feaSize2)
        x,y = self.simpleAttn1(u)[:,:self.feaSize1],self.simpleAttn2(v)[:,:self.feaSize2]
        return torch.cat([x,y], dim=1) # => batchSize × (feaSize1+feaSize2)

    def plane_attention_s(self, x, y):
        u = self.U(x).unsqueeze(dim=2) # => batchSize × seqLen1 × 1 × feaSize2
        v = y.unsqueeze(dim=1) # => batchSize × 1 × seqLen2 × feaSize2
        alpha = torch.sum(u*v,dim=3) # => batchSize × seqLen1 × seqLen2
        alpha = F.softmax(alpha.flatten(1,2),dim=1).unsqueeze(dim=1) # => batchSize × 1 × seqLen1*seqLen2

        x,y = x.unsqueeze(dim=2).expand(-1,-1,y.shape[1],-1),y.unsqueeze(dim=1).expand(-1,x.shape[1],-1,-1) # => batchSize × seqLen1 × seqLen2 × feaSize
        xy = torch.cat([x,y], dim=3).flatten(1,2) # => batchSize × seqLen1*seqLen2 × (feaSize1+feaSize2)
        return torch.matmul(alpha, xy).squeeze(dim=1) # => batchSize × (feaSize1+feaSize2)

    def plane_attention(self, x, y):
        u = self.U(x).unsqueeze(dim=2) # => batchSize × seqLen1 × 1 × feaSize2
        v = y.unsqueeze(dim=1) # => batchSize × 1 × seqLen2 × feaSize2
        alpha = F.tanh(u*v) # => batchSize × seqLen1 × seqLen2 × feaSize2
        alpha = self.W(alpha).squeeze(dim=3) # => batchSize × seqLen1 × seqLen2
        alpha = F.softmax(alpha.flatten(1,2),dim=1).unsqueeze(dim=1) # => batchSize × 1 × seqLen1*seqLen2

        x,y = x.unsqueeze(dim=2).expand(-1,-1,y.shape[1],-1),y.unsqueeze(dim=1).expand(-1,x.shape[1],-1,-1) # => batchSize × seqLen1 × seqLen2 × feaSize
        xy = torch.cat([x,y], dim=3).flatten(1,2) # => batchSize × seqLen1*seqLen2 × (feaSize1+feaSize2)
        return torch.matmul(alpha, xy).squeeze(dim=1) # => batchSize × (feaSize1+feaSize2)

    def forward(self, x, y):
        # x: batchSize × seqLen1 × feaSize1; y: batchSize × seqLen2 × feaSize2
        return self.dropout(self.attnFunc[self.attnType](x,y)) # => batchSize × (feaSize1+feaSize2)

class SelfAttention(nn.Module):
    def __init__(self, featureSize, dk, multiNum, name='selfAttn'):
        super(SelfAttention, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WK = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WV = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WO = nn.Linear(self.dk*multiNum, featureSize)
        self.name = name
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        queries = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        keys    = [self.WK[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        values  = [self.WV[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        scores  = [torch.bmm(queries[i], keys[i].transpose(1,2))/np.sqrt(self.dk) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × seqLen)
        # mask <EOS> padding
        if xlen is not None:
            for i in range(len(scores)):
                mask = torch.zeros(scores[0].shape, dtype=torch.float32, device=scores[i].device) # => batchSize × seqLen × seqLen
                for j,k in enumerate(xlen):
                    mask[j,:,k-1:] -= 999999
                scores[i] = scores[i] + mask
        z = [torch.bmm(F.softmax(scores[i], dim=2), values[i]) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        z = self.WO(torch.cat(z, dim=2)) # => batchSize × seqLen × feaSize
        return z

class LayerNormAndDropout(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='layerNormAndDropout'):
        super(LayerNormAndDropout, self).__init__()
        self.layerNorm = nn.LayerNorm(feaSize)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        return self.dropout(self.layerNorm(x))

class SimpleSelfAttention(nn.Module):
    def __init__(self, feaSize, name='simpleSelfAttn'):
        super(SimpleSelfAttention, self).__init__()
        self.feaSize = feaSize
        self.WO = nn.Linear(feaSize, feaSize)
        self.name = name
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        querie = x # => batchSize × seqLen × feaSize
        key    = x # => batchSize × seqLen × feaSize
        value  = x # => batchSize × seqLen × feaSize
        score  = torch.bmm(querie, key.transpose(1,2))/np.sqrt(self.feaSize) # => batchSize × seqLen × seqLen
        # mask <EOS> padding
        if xlen is not None:
            mask = torch.zeros(score.shape, dtype=torch.float32, device=score.device) # => batchSize × seqLen × seqLen
            for j,k in enumerate(xlen):
                mask[j,:,k-1:] -= 999999
            score = score + mask
        z = torch.bmm(F.softmax(score, dim=2), value) # => batchSize × seqLen × feaSize
        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class FFN(nn.Module):
    def __init__(self, featureSize, dropout=0.1, name='FFN'):
        super(FFN, self).__init__()
        self.layerNorm1 = nn.LayerNorm(featureSize)
        self.layerNorm2 = nn.LayerNorm(featureSize)
        self.Wffn = nn.Sequential(
                        nn.Linear(featureSize, featureSize*4), 
                        nn.ReLU(),
                        nn.Linear(featureSize*4, featureSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = x + self.dropout(self.layerNorm1(z)) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize
        return z+self.dropout(self.layerNorm2(ffnx)) # => batchSize × seqLen × feaSize

class Transformer(nn.Module):
    def __init__(self, featureSize, dk, multiNum, dropout=0.1):
        super(Transformer, self).__init__()
        self.selfAttn = SelfAttention(featureSize, dk, multiNum)
        self.ffn = FFN(featureSize, dropout)

    def forward(self, input):
        x, xlen = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z = self.selfAttn(x, xlen) # => batchSize × seqLen × feaSize
        return (self.ffn(x, z),xlen) # => batchSize × seqLen × feaSize
        
class TextTransformer(nn.Module):
    def __init__(self, seqMaxLen, layersNum, featureSize, dk, multiNum, dropout=0.1, name='textTransformer'):
        super(TextTransformer, self).__init__()
        posEmb = [[np.sin(pos/10000**(2*i/featureSize)) if i%2==0 else np.cos(pos/10000**(2*i/featureSize)) for i in range(featureSize)] for pos in range(seqMaxLen)]
        self.posEmb = nn.Parameter(torch.tensor(posEmb, dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer(featureSize, dk, multiNum, dropout)) for i in range(layersNum)]
                                     )
                                 )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        x = self.dropout(x+self.posEmb) # => batchSize × seqLen × feaSize
        return self.transformerLayers((x, xlen)) # => batchSize × seqLen × feaSize

class Transformer_Wcnn(nn.Module):
    def __init__(self, featureSize, dk, multiNum, seqMaxLen, dropout=0.1):
        super(Transformer_Wcnn, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WK = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WV = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WO = nn.Linear(self.dk*multiNum, featureSize)
        self.layerNorm1 = nn.LayerNorm([seqMaxLen, featureSize])
        self.layerNorm2 = nn.LayerNorm([seqMaxLen, featureSize])
        self.Wcnn = TextCNN(featureSize, featureSize, [1,3,5], reduction='None', actFunc=nn.ReLU(), name='Wffn_CNN')
        self.Wffn = nn.Sequential(
                        nn.Linear(featureSize*3, featureSize), 
                    )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        queries = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        keys = [self.WK[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        values = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        score = [torch.bmm(queries[i], keys[i].transpose(1,2))/np.sqrt(self.dk) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × seqLen)
        z = [torch.bmm(F.softmax(score[i], dim=2), values[i]) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        z = self.WO(torch.cat(z, dim=2)) # => batchSize × seqLen × feaSize
        z = x + self.dropout(self.layerNorm1(z)) # => batchSize × seqLen × feaSize
        ffnx = torch.cat(self.Wcnn(z), dim=2) # => batchSize × seqLen × feaSize*3
        ffnx = self.Wffn(ffnx) # => batchSize × seqLen × feaSize
        return z+self.dropout(self.layerNorm2(ffnx)) # => batchSize × seqLen × feaSize

class TextTransformer_Wcnn(nn.Module):
    def __init__(self, layersNum, featureSize, dk, multiNum, seqMaxLen, dropout=0.1, name='textTransformer'):
        super(TextTransformer_Wcnn, self).__init__()
        #posEmb = [[np.sin(pos/10000**(2*i/featureSize)) if i%2==0 else np.cos(pos/10000**(2*i/featureSize)) for i in range(featureSize)] for pos in range(seqMaxLen)]
        #self.posEmb = nn.Parameter(torch.tensor(posEmb, dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        self.transformerLayers = nn.Sequential(
                                        OrderedDict(
                                            [('transformer%d'%i, Transformer_Wcnn(featureSize, dk, multiNum, seqMaxLen, dropout)) for i in range(layersNum)]
                                        )
                                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.dropout(x) # => batchSize × seqLen × feaSize
        return self.transformerLayers(x) # => batchSize × seqLen × feaSize

class HierarchicalSoftmax(nn.Module):
    def __init__(self, inSize, hierarchicalStructure, lab2id, hiddenList1=[], hiddenList2=[], dropout=0.1, name='HierarchicalSoftmax'):
        super(HierarchicalSoftmax, self).__init__()
        self.name = name
        self.dropout = nn.Dropout(p=dropout)
        layers = nn.Sequential()
        for i,os in enumerate(hiddenList1):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), nn.ReLU())
            inSize = os
        self.hiddenLayers1 = layers
        moduleList = [nn.Linear(inSize, len(hierarchicalStructure))]

        layers = nn.Sequential()
        for i,os in enumerate(hiddenList2):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), nn.ReLU())
            inSize = os
        self.hiddenLayers2 = layers

        for i in hierarchicalStructure:
            moduleList.append( nn.Linear(inSize, len(i)) )
            for j in range(len(i)):
                i[j] = lab2id[i[j]]
        self.hierarchicalNum = [len(i) for i in hierarchicalStructure]
        self.restoreIndex = np.argsort(sum(hierarchicalStructure,[]))
        self.linearList = nn.ModuleList(moduleList)
    def forward(self, x):
        # x: batchSize × feaSize
        x = self.hiddenLayers1(x)
        x = self.dropout(x)
        y = [F.softmax(linear(x), dim=1) for linear in self.linearList[:1]]
        x = self.hiddenLayers2(x)
        y += [F.softmax(linear(x), dim=1) for linear in self.linearList[1:]]
        y = torch.cat([y[0][:,i-1].unsqueeze(1)*y[i] for i in range(1,len(y))], dim=1) # => batchSize × classNum
        return y[:,self.restoreIndex]

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gama=2, weight=-1, logit=True):
        super(FocalCrossEntropyLoss, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32), requires_grad=False)
        self.gama = gama
        self.logit = logit
    def forward(self, Y_pre, Y):
        if self.logit:
            Y_pre = F.softmax(Y_pre, dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        if self.weight.shape!=torch.Size([]):
            w = self.weight[Y]
        else:
            w = torch.tensor([1.0 for i in range(len(Y))], device=self.weight.device)
        w = (w/w.sum()).reshape(-1)
        return (-w*((1-P)**self.gama * torch.log(P))).sum()

class ContinusCrossEntropyLoss(nn.Module):
    def __init__(self, gama=2):
        super(ContinusCrossEntropyLoss, self).__init__()
        self.gama = gama
    def forward(self, Y_logit, Y):
        Y_pre = F.softmax(Y_logit, dim=1)
        lab_pre = Y_pre.argmax(dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        w = ((1+(lab_pre-Y).abs())**self.gama).float()
        w = (w/w.sum()).reshape(-1)
        return (-w*torch.log(P)).sum()

class PairWiseRankingLoss(nn.Module):
    def __init__(self, gama=1):
        super(PairWiseRankingLoss, self).__init__()
        self.gama = gama
    def forward(self, Y_logit, Y):
        # Y_logit, Y: batchSize1 × batchSize2;
        Y_pre = F.sigmoid(Y_logit)
        loss,cnt = 0,0
        for y_pre,y in zip(Y_pre,Y):
            # batchSize2
            neg = y_pre[y==0].unsqueeze(dim=1) # negNum × 1
            pos = y_pre[y==1].unsqueeze(dim=0) # 1 × posNum
            tmp = self.gama+(neg-pos) # => negNum × posNum
            tmp[tmp<0] = 0
            loss += tmp.sum()
            cnt += tmp.shape[0]*tmp.shape[1]
        return loss

class MultiLabelCircleLoss(nn.Module):
    def __init__(self):
        super(MultiLabelCircleLoss, self).__init__()
    def forward(self, Y_logit, Y):
        loss,cnt = 0,0
        for yp,yt in zip(Y_logit,Y):
            neg = yp[yt==0]
            pos = yp[yt==1]
            loss += torch.log(1+torch.exp(neg).sum()) + torch.log(1+torch.exp(-pos).sum())
            #loss += torch.log(1+(F.sigmoid(neg)**2*torch.exp(neg)).sum()) + torch.log(1+((1-F.sigmoid(pos))**2*torch.exp(-pos)).sum())
            #loss += len(yp) * (torch.log(1+torch.exp(neg).sum()/len(neg)) + torch.log(1+torch.exp(-pos).sum()/len(pos)))
            cnt += 1
        return loss/cnt

'''
import torch
from nnLayer import *
Y = torch.tensor([0,2], dtype=torch.long)
Y_logit = torch.tensor([[0.1,0.9,1],[0.6,2,0.4]], dtype=torch.float32)
CCEL = ContinusCrossEntropyLoss()
CCEL(Y_logit, Y)
'''

class MultiTaskCEL(nn.Module):
    def __init__(self, lossBalanced=True, ageW=1, genderW=1, name='MTCEL'):
        super(MultiTaskCEL, self).__init__()
        self.genderCriterion,self.ageCriterion = nn.CrossEntropyLoss(),nn.CrossEntropyLoss()#ContinusCrossEntropyLoss()#
        self.genderS,self.ageS = nn.Parameter(torch.zeros(1,dtype=torch.float), requires_grad=lossBalanced),nn.Parameter(torch.zeros(1,dtype=torch.float), requires_grad=lossBalanced)
        self.lossBalanced = lossBalanced
        self.name = name
        self.ageW,self.genderW = ageW,genderW
    def forward(self, genderY_logit, genderY, ageY_logit, ageY):
        if self.lossBalanced:
            return self.genderW * torch.exp(-self.genderS) * self.genderCriterion(genderY_logit,genderY) + self.ageW * torch.exp(-self.ageS) * self.ageCriterion(ageY_logit,ageY) + (self.genderS+self.ageS)/2
        else:
            return self.genderW * self.genderCriterion(genderY_logit,genderY) + self.ageW * self.ageCriterion(ageY_logit,ageY)


def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


def rope_flash(x,axis):
    # axis = [1,2]
    shape = list(x.shape)
    if isinstance(axis, int):
        axis = [axis]
    spatial_shape = [shape[i] for i in axis]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    
    position = torch.reshape(torch.tensor(np.arange(total_len),dtype=torch.float32,device=x.device),spatial_shape)
    
    for i in range(axis[-1] + 1, len(shape)-1, 1):
        position = position.unsqueeze(dim=-1)
    
    half_size = shape[-1] // 2
    freq_seq = torch.tensor(np.arange(half_size),dtype=torch.float32,device=x.device)/float(half_size)
    inv_freq = 10000 ** -freq_seq
    sinusoid = einsum('...,d-> ...d', position, inv_freq)
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([ x1 * cos- x2 * sin, x2 * cos + x1 * sin ], dim=-1)


class rel_pos_bias(nn.Module):
    def __init__(self, name='rel_pos_bias'):
        super(rel_pos_bias, self).__init__()
        # rel_pos_bias
        self.a = nn.Parameter(torch.ones(128))
        self.b = nn.Parameter(torch.ones(128))
        truncated_normal_(self.a, std = 0.02)
        truncated_normal_(self.b, std = 0.02)
        self.name = name

    def forward(self, c):
        a = rope_flash(self.a[None,:].repeat(c, 1),axis=0)
        b = rope_flash(self.b[None,:].repeat(c, 1),axis=0)
        t = einsum('mk,nk-> mn', a, b)
        return t

# class flash_linear_trans(nn.Module):
#     def __init__(self, embSize, seqMaxLen, chunk_length,trans_s,index_layer, expansion_factor=2,shift_tokens=True):
#         super(flash_linear_trans, self).__init__()
#         self.expansion_factor = expansion_factor
#         self.seqMaxLen = seqMaxLen
#         self.s = trans_s
#         self.e = embSize * self.expansion_factor  # e e=2d
#         self.chunk_length = chunk_length    # 分块 划分为 num_chunks = n/c 个长度为chunk_length(c)的块
        
#         self.UV = nn.Linear(embSize, expansion_factor * embSize * 2 + trans_s)  # 这是单独建立e*d,e*d,s*d的简便写法
#         truncated_normal_(self.UV.weight, std=0.02)

#         self.gamma = nn.Parameter(torch.ones(4, trans_s))
#         self.beta = nn.Parameter(torch.zeros(4, trans_s))
#         truncated_normal_(self.gamma, std = 0.02)
#         # self.out_2d = nn.Linear(self.e, self.e)        
#         if (index_layer>0):
#             self.out = nn.Linear(self.e, self.e)
#             self.layerNorm = nn.LayerNorm(embSize*2)
#         else:    
#             self.out = nn.Linear(self.e, embSize)
#             self.layerNorm = nn.LayerNorm(embSize)
#         truncated_normal_(self.out.weight, std=0.02)
        
#         #self.layerNorm = RMSNorm()
#         self.actFun = nn.SiLU()
#         self.actFun_ReLU = nn.ReLU()
#         self.shift_tokens = shift_tokens
#         self.dropout = nn.Dropout(p=0.1)
#         self.dropout02 = nn.Dropout(p=0.2)
#         self.rel_pos_bias = rel_pos_bias()

#         self.index_layer = index_layer

#     def forward(self, input, prev=None):
#         # input => batchSize × seqLen × embSize
#         # 分块
#         # 分成 batchSize × numChunks(g) × chunkLength(c) × embSize(d)        
#         if self.shift_tokens:
#             x_shift, x_pass = input.chunk(2, dim = -1)
#             x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
#             x = torch.cat((x_shift, x_pass), dim = -1)   

#         padding = padding_to_multiple_of(self.seqMaxLen, self.chunk_length)

#         if padding>0:
#             x = F.pad(x, ( 0, 0, 0, padding))

#         num_chunks = int(x.shape[1]/self.chunk_length)

#         chunks = torch.chunk(x, num_chunks, dim=1)

#         x = torch.cat([chunks[i].unsqueeze(1) for i in range(len(chunks))], dim=1) # bgcd

#         _, g, c, d = list(x.shape)  # g表示n/c,即 num_chunks, d 表示 embeddingSize
        
#         U, V, base = torch.split(self.actFun(self.UV(x)), [self.e, self.e, self.s],dim=-1)  # => base=Z : bgcs  V: bgce

#         base = einsum('...r, hr-> ...hr', base, self.gamma)+self.beta

#         base = rope_flash(base, axis=[1,2])

#         quad_q, quad_k, lin_q, lin_k = torch.chunk(base, 4, dim=-2)
#         quad_q = torch.squeeze(quad_q, dim=3)
#         quad_k = torch.squeeze(quad_k, dim=3)
#         lin_q = torch.squeeze(lin_q, dim=3)
#         lin_k = torch.squeeze(lin_k, dim=3)

#         bias = self.rel_pos_bias(c)
#         if prev is not None:
#             attn = torch.matmul(quad_q, quad_k.transpose(2, 3))/ self.s**0.5 + bias + prev
#         else:
#             attn = torch.matmul(quad_q, quad_k.transpose(2, 3))/ self.s**0.5 + bias    
#         # # bias = self.rel_pos_bias(c)
#         # if prev is not None:
#         #     attn = torch.matmul(quad_q, quad_k.transpose(2, 3))/ self.s**0.5 + prev
#         # else:
#         #     attn = torch.matmul(quad_q, quad_k.transpose(2, 3))/ self.s**0.5   
#         prev = attn
#         A = F.softmax(prev,dim=-1)
#         A = self.dropout(A)     
#         quadratic = torch.matmul(A, V) # quadratic => bgce

#         lin_kv = torch.matmul(lin_k.transpose(2,3),V)
#         linear = torch.matmul(lin_q,lin_kv) /self.seqMaxLen # linear => bgce

#         U,quadratic, linear = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :self.seqMaxLen], (U,quadratic, linear))
#         x = self.out(U * (quadratic+linear))
        
#         if(self.index_layer>0): #第二层
#             # x = self.out_2d(U * (quadratic+linear))
#             x = self.layerNorm(self.dropout(x))  # x => batchSize × SeqLen × [embSize*2]
#         else:
#             # GAU #第一层
#             # x = self.out(U * (quadratic+linear))
#             # Post Norm
#             x = self.layerNorm(input + self.dropout(x))  # x => batchSize × SeqLen × embSize
#         return x, prev


# class FLASHLayer(nn.Module):
#     def __init__(self,embSize,seqMaxLen,chunk_length,trans_s,index_layer):
#         super(FLASHLayer, self).__init__()
#         self.flash_linear_trans = flash_linear_trans(embSize,seqMaxLen,chunk_length,trans_s,index_layer)

#     def forward(self, input,prev):
#         # x: batchSize × seqLen × embSize
#         x,prev = self.flash_linear_trans(input,prev)
#         return x,prev    # => batchSize × seqLen × embSize


# class FLASH(nn.Module): # 这里的feaSize = embeddingSize
#     def __init__(self, seqMaxLen, embSize, numLayers, chunk_length,trans_s,name='FLASH'):
#         super(FLASH, self).__init__()
#         self.FLASHLayers = nn.Sequential(
#                                      OrderedDict([
#                                           ('FLASHLayer_%d'%i, FLASHLayer(embSize, seqMaxLen,chunk_length,trans_s,i)) for i in range(numLayers)
#                                      ])
#                                  )
#         self.name = name
        
#     def forward(self, x):
#         # x: batchSize × seqLen × feaSize
#         prev = None
#         for lay in self.FLASHLayers:
#             x, prev = lay(x, prev=prev)
#         return x    # => batchSize × seqLen × feaSize


class flash_linear_trans(nn.Module):
    def __init__(self, embSize, seqMaxLen, chunk_length, trans_s, expansion_factor=2, s=300, shift_tokens=True):
        super(flash_linear_trans, self).__init__()
        self.expansion_factor = expansion_factor
        self.seqMaxLen = seqMaxLen
        self.s = trans_s
        self.e = embSize * self.expansion_factor  # e e=2d
        self.chunk_length = chunk_length    # 分块 划分为 num_chunks = n/c 个长度为chunk_length(c)的块
        
        self.UV = nn.Linear(embSize, expansion_factor * embSize * 2 + trans_s)  # 这是单独建立e*d,e*d,s*d的简便写法
        truncated_normal_(self.UV.weight, std=0.02)

        self.gamma = nn.Parameter(torch.ones(4, trans_s))
        self.beta = nn.Parameter(torch.zeros(4, trans_s))
        truncated_normal_(self.gamma, std = 0.02)
        
        self.out = nn.Linear(self.e, embSize)
        truncated_normal_(self.out.weight, std=0.02)
        
        self.layerNorm = nn.LayerNorm(embSize)
        #self.layerNorm = RMSNorm()
        self.actFun = nn.SiLU()
        self.actFun_ReLU = nn.ReLU()
        self.shift_tokens = shift_tokens
        self.dropout = nn.Dropout(p=0.1)
        self.dropout02 = nn.Dropout(p=0.2)
        self.rel_pos_bias = rel_pos_bias()

    def forward(self, input, prev=None):
        # input => batchSize × seqLen × embSize
        # 分块
        # 分成 batchSize × numChunks(g) × chunkLength(c) × embSize(d)        
        if self.shift_tokens:
            x_shift, x_pass = input.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            x = torch.cat((x_shift, x_pass), dim = -1)   

        padding = padding_to_multiple_of(self.seqMaxLen, self.chunk_length)

        if padding>0:
            x = F.pad(x, ( 0, 0, 0, padding))

        num_chunks = int(x.shape[1]/self.chunk_length)

        chunks = torch.chunk(x, num_chunks, dim=1)

        x = torch.cat([chunks[i].unsqueeze(1) for i in range(len(chunks))], dim=1) # bgcd

        _, g, c, d = list(x.shape)  # g表示n/c,即 num_chunks, d 表示 embeddingSize
        
        U, V, base = torch.split(self.actFun(self.UV(x)), [self.e, self.e, self.s],dim=-1)  # => base=Z : bgcs  V: bgce

        base = einsum('...r, hr-> ...hr', base, self.gamma)+self.beta

        base = rope_flash(base, axis=[1,2])

        quad_q, quad_k, lin_q, lin_k = torch.chunk(base, 4, dim=-2)
        quad_q = torch.squeeze(quad_q, dim=3)
        quad_k = torch.squeeze(quad_k, dim=3)
        lin_q = torch.squeeze(lin_q, dim=3)
        lin_k = torch.squeeze(lin_k, dim=3)

        bias = self.rel_pos_bias(c)
        if prev is not None:
            attn = torch.matmul(quad_q, quad_k.transpose(2, 3))/ self.s**0.5 + bias + prev
        else:
            attn = torch.matmul(quad_q, quad_k.transpose(2, 3))/ self.s**0.5 + bias    
        prev = attn
        A = F.softmax(prev,dim=-1)
        A = self.dropout(A)     
        quadratic = torch.matmul(A, V) # quadratic => bgce

        lin_kv = torch.matmul(lin_k.transpose(2,3),V)
        linear = torch.matmul(lin_q,lin_kv) /self.seqMaxLen # linear => bgce

        U,quadratic, linear = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :self.seqMaxLen], (U,quadratic, linear))
        
        # GAU
        x = self.out(U * (quadratic+linear))
        
        # Post Norm
        
        x = self.layerNorm(input + self.dropout(x))  # x => batchSize × SeqLen × embSize
        return x, prev


class FLASHLayer(nn.Module):
    def __init__(self,embSize,seqMaxLen,chunk_length,trans_s):
        super(FLASHLayer, self).__init__()
        self.flash_linear_trans = flash_linear_trans(embSize,seqMaxLen,chunk_length,trans_s)

    def forward(self, input,prev):
        # x: batchSize × seqLen × embSize
        x,prev = self.flash_linear_trans(input,prev)
        return x,prev    # => batchSize × seqLen × embSize


class FLASH(nn.Module): # 这里的feaSize = embeddingSize
    def __init__(self, seqMaxLen, embSize, numLayers, chunk_length, trans_s, name='FLASH'):
        super(FLASH, self).__init__()
        self.FLASHLayers = nn.Sequential(
                                     OrderedDict([
                                          ('FLASHLayer_%d'%i, FLASHLayer(embSize, seqMaxLen, chunk_length, trans_s)) for i in range(numLayers)
                                     ])
                                 )
        self.name = name
        
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        prev = None
        for lay in self.FLASHLayers:
            x, prev = lay(x, prev=prev)
        return x    # => batchSize × seqLen × feaSize
