import torch
import torch.nn as nn
import torch.nn.functional as F
from openvgd.models.mcan.modules import AttFlat, LayerNorm, MCA_ED
from transformers import BertModel
from transformers import AlbertModel
from transformers import RobertaModel


class Net_Full(nn.Module):
    def __init__(self, __C, init_dict):
        super(Net_Full, self).__init__()
        self.__C = __C
        
        self.embedding = nn.Embedding(num_embeddings=init_dict['token_size'], embedding_dim=__C.WORD_EMBED_SIZE)
        self.embedding.weight.data.copy_(torch.from_numpy(init_dict['pretrained_emb']))
        
        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HSIZE,
            num_layers=1,
            batch_first=True
        )

        imgfeat_linear_size = __C.FRCNFEAT_SIZE
        if __C.BBOX_FEATURE:
            self.bboxfeat_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.imgfeat_linear = nn.Linear(imgfeat_linear_size, __C.HSIZE)

        self.backnone = MCA_ED(__C)
        self.attflat_x = AttFlat(__C)
        self.attflat_y = AttFlat(__C)
        self.proj_norm = LayerNorm(__C.ATTFLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.ATTFLAT_OUT_SIZE, 1)


    def forward(self, input):
        frcn_feat, bbox_feat, ques_ix = input

        # with torch.no_grad():
        # Make mask for attention learning
        if self.__C.USE_BERT:
            x_mask = self.make_mask(ques_ix[:, 1:-1].unsqueeze(2))
        else:
            x_mask = self.make_mask(ques_ix.unsqueeze(2))
        y_mask = self.make_mask(frcn_feat)

        # Pre-process Language Feature
        if self.__C.BERT_ENCODER:
            outputs = self.bert_encoder(ques_ix)
            last_hidden_state = outputs[0]
            x_in = last_hidden_state[:, 1:-1, :]  # remove CLS and SEP, making this to max_token=14 # (64,14,768)
        elif not self.__C.BERT_ENCODER and self.__C.USE_BERT:
            outputs = self.bert_layer(ques_ix)
            hidden_states = outputs[2]  # All Outputs of bert model's hidden
            concat_layers = torch.cat([hidden_states[i] for i in [-1, -2, -3, -4, -5]], dim=-1)
            # print(concat_layers.shape)
            concat_layers = concat_layers[:, 1:-1, :]
            # print(concat_layers.shape)
            if self.__C.USE_LSTM:
                x_in, _ = self.lstm(concat_layers)  # into LSTM
            else:
                x_in = self.bert_conn(concat_layers)  # into BERT_CONN_Layer

        elif self.__C.GLOVE_FEATURE:
            lang_feat = self.embedding(ques_ix)
            x_in, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        if self.__C.BBOX_FEATURE:
            bbox_feat = self.bboxfeat_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        y_in = self.imgfeat_linear(frcn_feat)

        x_out, y_out = self.backnone(x_in, y_in, x_mask, y_mask)
        #print(x_out.shape) torch.Size([3200, 50, 512])
        #print(y_out.shape) torch.Size([3200, 36, 512])
    
        #fusion_1,fusion_2 = self.backnone(x_in, y_in, x_mask, y_mask)
        #x_out = self.attflat_x(x_out, x_mask)
        #y_out = self.attflat_y(y_out, y_mask)
        #xy_out = x_out + y_out
        x_out = self.proj_norm(x_out)
        y_out = self.proj_norm(y_out)
        
        
        #xy_out_1 = self.proj_norm(fusion_1)
        #xy_out_2 = self.proj_norm(fusion_2)
        scores_1 = self.proj(x_out).squeeze(-1)
        scores_1 = torch.sigmoid(scores_1)
        
        scores_2 = self.proj(y_out).squeeze(-1)
        scores_2 = torch.sigmoid(scores_2)
        
        
        
        
        #scores = self.proj(xy_out).squeeze(-1)
        #scores = torch.sigmoid(scores)

        return scores_1, scores_2

    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)

