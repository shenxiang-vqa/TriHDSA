# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
from openvqa.models.mcan.capsatt_sdcam import caps_att
from openvqa.models.mcan.capsatt_visualmap import caps_visual
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class AttFlat_caps(nn.Module):
    def __init__(self, __C):
        super(AttFlat_caps, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.HIDDEN_SIZE
        )
    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        
        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

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

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

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

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        return y, x
'''      
    
    
    
# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------
   
    
    
    
    
# language mask       
def att_mask(att, att_mask): #att[64,100]
    value, att_argmax = att.topk(att.size(1), dim=1, largest=True)
    b = att.size(1) - att_mask.sum(dim=1)
    b = b * 0.5  #64
    mid_list = []
    for i in range(att.size(0)):
        mid = value[i][int(b[i].item())]
        mid_list.append(mid)
    mid_t = torch.stack(mid_list, dim=0).unsqueeze(-1)  # b,1
    mid_t = mid_t.repeat(1, att.size(1))
    mask = mid_t > att
    return mask
    
   
    
def fuse_features(image_feat, language_feat, scale=1.0, sigma=0.1, pow=2.0):
    """
    对图像特征和语言特征进行融合
    
    参数：
        image_feat (torch.Tensor)：图像特征张量，形状为 (batch_size, feat_dim)
        language_feat (torch.Tensor)：语言特征张量，形状为 (batch_size, feat_dim)
        scale (float)：尺度参数，默认为1.0
        sigma (float)：sigma 参数，默认为0.1
        pow (float)：幂参数，默认为2.0
    
    返回：
        proj_feat (torch.Tensor)：融合后的特征张量，形状与输入特征相同
    """
    verify_s = (torch.sigmoid(F.normalize(image_feat, p=2, dim=-1)) * torch.sigmoid(F.normalize(language_feat, p=2, dim=-1))).sum(dim=-1, keepdim=True)
    verify_s = scale * torch.exp(-(1 - verify_s).pow(pow) / (2 * sigma ** 2))
    proj_feat = verify_s * image_feat +  language_feat
    
    return proj_feat


def jaccard_projection(image_feat, language_feat):
    """
    Jaccard projection function.
    Args:
        image_feat: Image features tensor, size (batch_size, feature_dim)
        language_feat: Language features tensor, size (batch_size, feature_dim)
    
    Returns:
        proj_feat: Projected features tensor, size (batch_size, feature_dim)
    """
    intersection = torch.min(image_feat, language_feat)
    union = torch.max(image_feat, language_feat)
    epsilon = 1e-8
    jaccard_similarity = (intersection.sum(dim=-1, keepdim=True) + epsilon) / (union.sum(dim=-1, keepdim=True) + epsilon)
    proj_feat = image_feat + jaccard_similarity * language_feat
    return proj_feat



  
   
class MCA_attmask15(nn.Module):
    def __init__(self, __C):
        super(MCA_attmask15, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list_1 = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])
        self.dec_list_2 = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

        self.attflat_lang = AttFlat_caps(__C)
        self.attflat_img = AttFlat_caps(__C)
        self.attflat_img_2 = AttFlat_caps(__C)

        self.mid_feat_extract_img = caps_att(num_iterations=4, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                             out_dim=__C.HIDDEN_SIZE)
        self.mid_feat_extract_lang = caps_att(num_iterations=4, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                              out_dim=__C.HIDDEN_SIZE)

        self.caps_visualmap_branch1_img = caps_visual(num_iterations=4, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                                      out_dim=__C.HIDDEN_SIZE)
        self.caps_visualmap_branch1_lang = caps_visual(num_iterations=4, num_capsules=100, dim=__C.HIDDEN_SIZE,
                                                       out_dim=__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)
        lang_query_reserve = self.attflat_lang(x, x_mask)
        lang_query_branch1 = lang_query_branch2 = lang_query_reserve
        y_branch1 = y_branch2 = y  #64,100,512

        for dec in self.dec_list_1:
            y_branch1 = dec(y_branch1, x, y_mask, x_mask)
            img_query_branch1, c_visual = self.caps_visualmap_branch1_img(lang_query_branch1, y_branch1,y_mask)  # b,512  b,100
            lang_query_branch1, c_lang = self.caps_visualmap_branch1_lang(img_query_branch1, x, x_mask)  # b,512    b,14

        img_feat_query = self.attflat_img(y_branch1, y_mask)  # torch.Size([64, 512])
        img_feat_f = torch.cat([img_feat_query, img_query_branch1], dim=-1)  # b,1024
        lang_feat_f = torch.cat([lang_query_reserve, lang_query_branch1], dim=-1)  # b,1024
        
        #proj_feat = jaccard_projection (img_feat_f, lang_feat_f) 
         
        
        w_att = lang_feat_f + img_feat_f
        w1_att= torch.sigmoid(w_att)
        lang_feat_f = w1_att * lang_feat_f
        img_feat_f = (1-w1_att) * img_feat_f
        proj_feat = lang_feat_f + img_feat_f
        

        y_att_mask = att_mask(c_visual, y_mask.squeeze(1).squeeze(1))  # b,100
        mul_y = torch.ones_like(y_branch2)  # b,100,512
        mul_y = mul_y.masked_fill(y_att_mask.unsqueeze(-1), 0.3)
        y_branch2 = y_branch2 * mul_y

        for dec in self.dec_list_2:
            y_branch2 = dec(y_branch2, x, y_mask, x_mask)
            img_query_branch2 = self.mid_feat_extract_img(lang_query_branch2, y_branch2,y_mask)  # b,1024 b,100
            lang_query_branch2 = self.mid_feat_extract_lang(img_query_branch2, x, x_mask)  # b,1024 b,14

        img_feat_query_2 = self.attflat_img_2(y_branch2, y_mask)  # torch.Size([64, 512])
        img_feat_f_2 = torch.cat([img_feat_query_2, img_query_branch2], dim=-1)  # b,2048
        lang_feat_f_2 = torch.cat([lang_query_reserve, lang_query_branch2], dim=-1)  # b,2048
        
        w_att1 = lang_feat_f_2 + img_feat_f_2
        w2_att= torch.sigmoid(w_att1)
        lang_feat_f = w2_att * lang_feat_f
        img_feat_f = (1-w2_att) * img_feat_f
        proj_feat_2 = lang_feat_f_2 + img_feat_f_2
        
        
        #proj_feat_2 = jaccard_projection (img_feat_f_2, lang_feat_f_2) 
        
        return proj_feat, proj_feat_2
