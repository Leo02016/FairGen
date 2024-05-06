import torch
from torch import nn
import math
import torch.nn.functional as F
from copy import deepcopy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 0.00000001)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Discriminator(nn.Module):
    def __init__(self, d_model, num, dropout=0.5):
        super(Discriminator, self).__init__()
        self.d_model = d_model
        self.classifier = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(128, num))

    def forward(self, seq, identity=None, labels=None):
        if labels is not None:
            # cost-sensitive loss (J_p in eq 3)
            logits_prob = F.softmax(self.classifier(seq), dim=-1)
            loss_over_node = self.cross_entropy_loss(logits_prob, labels)
            loss = torch.mean(loss_over_node)
            # group fairness loss
            c = labels.shape[1]
            # count = [[0, 0] for _ in range(c)]
            pr = [[0, 0] for _ in range(c)]
            for i in range(c):
                # index of samples from class i
                index = (labels[:, i] == 1).nonzero()
                index = index.view(index.shape[1], -1)[0]
                # index of the unprotected set
                index_2 = (identity[index] == 0).nonzero()
                index_2 = index_2.view(index_2.shape[1], -1)[0]
                pr[i][0] = torch.mean(logits_prob[index_2])
                # index of the protected set
                index_3 = (identity[index] == 1).nonzero()
                index_3 = index_3.view(index_3.shape[1], -1)[0]
                pr[i][1] = torch.mean(logits_prob[index_3])
                # (J_F in eq 3)
                loss += torch.abs(pr[i][0] - pr[i][1])/c
            return loss
        else:
            return F.softmax(self.classifier(seq), dim=-1)


    def cross_entropy_loss(self, pred, labels):
        return torch.sum(torch.mul(labels, torch.log(pred)), dim=1) * (-1)


class BiLevelMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(BiLevelMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.coef = 0.5

    def forward(self, seq, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = seq.size(0)
        query_r, key_r, value_r = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (seq, seq, seq))]
        x, self.attn = attention(query_r, key_r, value_r, mask=mask,
                                 dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Encoder(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.res_0 = Resnet_0(size, dropout)
        self.res_1 = Resnet_1(size, dropout)
        self.size = size

    def forward(self, x_1, mask=None):
        "Transformer Encoder"
        x = self.res_0(x_1, lambda x_1: self.self_attn(x_1,  mask))  # Encoder self-attention
        return self.res_1(x, self.feed_forward)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


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


class Resnet_0(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(Resnet_0, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x_1 + self.dropout(sublayer(self.norm(x_1)))


class Resnet_1(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(Resnet_1, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        h, N, dropout = config.h, config.N, config.dropout
        d_model, d_ff = config.d_model, config.d_ff
        attn = BiLevelMultiHeadedAttention(h, d_model)
        ff = FeedForward(d_model, d_ff, dropout)
        self.norm = LayerNorm(d_model)
        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        # Fully-Connected Layer
        self.fc = nn.Linear(d_model, config.output_size)


    def forward(self, x_1):
        encoded_sents = self.encoder(x_1)
        final_feature_map = encoded_sents[:, -1, :]
        final_out = self.fc(final_feature_map)
        return F.softmax(final_out, dim=-1)
