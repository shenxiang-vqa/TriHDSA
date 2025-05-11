import torch
import numpy as np
import math, random, json
import torch.nn as nn
import torch.nn.functional as F

from openitm.models.mcan.capsatt_sdcam import caps_att
from openitm.models.mcan.capsatt_visualmap import caps_visual

''' 
==================
    Operations
==================
'''

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6, dim=-1):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.dim = dim
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = x.std(self.dim, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=__C.ATTFLAT_MLP_SIZE,
            out_size=__C.ATTFLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.linear_merge = nn.Linear(__C.HSIZE * __C.ATTFLAT_GLIMPSES, __C.ATTFLAT_OUT_SIZE)

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.ATTFLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class AttFlat_caps(nn.Module):
    def __init__(self, __C):
        super(AttFlat_caps, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=__C.ATTFLAT_MLP_SIZE,
            out_size=__C.ATTFLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HSIZE * __C.ATTFLAT_GLIMPSES,
            __C.HSIZE
        )
    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.ATTFLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted





class MHAtt(nn.Module):
    def __init__(self, __C, base=64, hsize_k=None, bias=False):
        super(MHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.HSIZE_INSIDE)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=__C.HSIZE * 4,
            out_size=__C.HSIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt =  MHAtt(__C, base=64, hsize_k=None)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HSIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HSIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()


        self.mhatt1 = MHAtt(__C, base=64, hsize_k=None)
        self.mhatt2 = MHAtt(__C, base=64, hsize_k=None)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HSIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HSIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HSIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

'''
# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()
        self.__C = __C
        self.enc_list = nn.ModuleList([SA(__C) for _ in range(6)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(6)])

    def forward(self, y, x, y_mask, x_mask):

        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)
            
        print(y.shape)
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)
        print(x.shape)

        return y, x
'''
# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(6)])
        self.dec_list_1 = nn.ModuleList([SGA(__C) for _ in range(6)])
        self.dec_list_2 = nn.ModuleList([SGA(__C) for _ in range(6)])

        self.attflat_lang = AttFlat_caps(__C)
        self.attflat_img = AttFlat_caps(__C)
        self.attflat_img_2 = AttFlat_caps(__C)

        self.mid_feat_extract_img = caps_att(num_iterations=3, num_capsules=100, dim=__C.HSIZE,
                                             out_dim=__C.HSIZE)
        self.mid_feat_extract_lang = caps_att(num_iterations=3, num_capsules=100, dim=__C.HSIZE,
                                              out_dim=__C.HSIZE)

        self.caps_visualmap_branch1_img = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HSIZE,
                                                      out_dim=__C.HSIZE)
        self.caps_visualmap_branch1_lang = caps_visual(num_iterations=3, num_capsules=100, dim=__C.HSIZE,
                                                       out_dim=__C.HSIZE)

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)
        lang_query_reserve = self.attflat_lang(x, x_mask)
        lang_query_branch1 = lang_query_branch2 = lang_query_reserve
        y_branch1 = y_branch2 = y

        for dec in self.dec_list_1:
            y_branch1 = dec(y_branch1, x, y_mask, x_mask)
            img_query_branch1, c_visual = self.caps_visualmap_branch1_img(lang_query_branch1, y_branch1,
                                                                          y_mask)  # b,1024 b,100
            lang_query_branch1, c_lang = self.caps_visualmap_branch1_lang(img_query_branch1, x, x_mask)  # b,1024 b,14

        img_feat_query = self.attflat_img(y_branch1, y_mask)  # torch.Size([64, 512])
        img_feat_f = torch.cat([img_feat_query, img_query_branch1], dim=-1)  # b,2048
        lang_feat_f = torch.cat([lang_query_reserve, lang_query_branch1], dim=-1)  # b,2048
        proj_feat = lang_feat_f + img_feat_f

        y_att_mask = att_mask(c_visual, y_mask.squeeze(1).squeeze(1))  # b,100
        mul_y = torch.ones_like(y_branch2)  # b,100,512
        mul_y = mul_y.masked_fill(y_att_mask.unsqueeze(-1), 0.3)
        y_branch2 = y_branch2 * mul_y

        for dec in self.dec_list_2:
            y_branch2 = dec(y_branch2, x, y_mask, x_mask)
            img_query_branch2 = self.mid_feat_extract_img(lang_query_branch2, y_branch2, y_mask)  # b,1024 b,100
            lang_query_branch2 = self.mid_feat_extract_lang(img_query_branch2, x, x_mask)  # b,1024 b,14

        img_feat_query_2 = self.attflat_img_2(y_branch2, y_mask)  # torch.Size([64, 512])
        img_feat_f_2 = torch.cat([img_feat_query_2, img_query_branch2], dim=-1)  # b,2048
        lang_feat_f_2 = torch.cat([lang_query_reserve, lang_query_branch2], dim=-1)  # b,2048
        proj_feat_2 = lang_feat_f_2 + img_feat_f_2
        #print(proi_feat.shape)
        #print(proi_feat_2.shape)

        return proj_feat, proj_feat_2

def att_mask(att, att_mask):
    value, att_argmax = att.topk(att.size(1), dim=1, largest=True)
    b = att.size(1) - att_mask.sum(dim=1)
    b = b * 0.5
    mid_list = []
    for i in range(att.size(0)):
        mid = value[i][int(b[i].item())]
        mid_list.append(mid)
    mid_t = torch.stack(mid_list, dim=0).unsqueeze(-1)  # b,1
    mid_t = mid_t.repeat(1, att.size(1))
    mask = mid_t > att
    return mask
     
