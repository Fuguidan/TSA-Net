import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from frequency import FFT_b

########################################################################################


class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x

class TwoStreamFeatureExtractor(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(TwoStreamFeatureExtractor, self).__init__()
        drate = 0.5
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,

            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.features6 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=20, stride=4, bias=False, padding=10),
            nn.BatchNorm1d(64),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=9, stride=3, padding=1)
        )
        self.dropout = nn.Dropout(drate)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        f = FFT_b(x)
        x6 = self.features6(f)
        x_concat = torch.cat((x1, x2, x6), dim=2)
        x_concat = self.dropout(x_concat)

        return x_concat

##########################################################################################


def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedSelfAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(30, 30, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class ResidualBlock(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''
    def __init__(self, d_model):
        super(ResidualBlock, self).__init__()
        self.norm = LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, 200)
        self.linear2 = nn.Linear(200, d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        sub = self.norm(x)
        sub = self.linear1(sub)
        sub = F.relu(sub)
        sub = self.dropout1(sub)
        sub = self.linear2(sub)
        sub = self.dropout2(sub)
        return x + sub

class FeatureContextLearning(nn.Module):
    def __init__(self, d_model):
        super(FeatureContextLearning, self).__init__()
        h = 5  # number of attention heads
        self.conv = CausalConv1d(30, 30, kernel_size=7, stride=1, dilation=1)
        self.norm = LayerNorm(d_model)
        self.mhsa = MultiHeadedSelfAttention(h, d_model)
        self.rb = ResidualBlock(d_model=120)

    def forward(self, x):
        query = self.conv(x)
        attn = self.mhsa(self.norm(query), x, x)
        attn = attn + query
        attn = self.rb(attn)
        return attn

class TemporalSpectralAttnNet(nn.Module):
    def __init__(self):
        super(TemporalSpectralAttnNet, self).__init__()

        d_model = 120  # set to be 100 for SHHS dataset
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.tfe = TwoStreamFeatureExtractor(afr_reduced_cnn_size)
        self.RE = nn.Sequential(
            nn.Conv1d(128, 30, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(30),
            nn.ReLU(inplace=True)
        )
        self.fcl = FeatureContextLearning(d_model)
        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x):
        x_feat = self.tfe(x)
        x_feat = self.RE(x_feat)
        x_encode = self.fcl(x_feat)
        x_encode = x_encode.contiguous().view(x_encode.shape[0], -1)
        final_output = self.fc(x_encode)
        return final_output



