import numpy as np
import random
import scipy.stats as ss
import torch, math
import torch.nn as nn
from ..subNets import BertTextEncoder
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from sklearn.preprocessing import MinMaxScaler
from minepy import MINE
from .rdc import rdc, rdc_torch
from timm.models.layers import Mlp, DropPath
###-----MyModel-------##
# mae:  0.77090746
# corr:  0.7578500411013062
# mult_acc:  0.4577259475218659
# Accuracy (pos/neg)  0.8201219512195121
# f_score 0.8204003576465237
# best mae on test : 0.77090746  already save at 2023-06-21_10_15_18
#########加了HSIC损失系数为0.3 head为4



# mae:  0.7816912 MOSI
# corr:  0.7621103273999171
# mult_acc:  0.45481049562682213
# Accuracy (pos/neg)  0.8170731707317073
# f_score 0.8164197371514444

# mae:  0.5423009 MOSEI
# corr:  0.7595694738563693
# mult_acc:  0.530081650193382
# Accuracy (pos/neg)  0.8534031413612565
# f_score 0.8536544137130745
# best mae on test : 0.5423009  already save at 2023-06-21_21_21_43

#########加了HSIC损失系数为0.3 head为4



def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)

def rdc_cal(x, y):
    x = x.cpu().data.numpy()
    y = y.cpu().data.numpy()
    b = x.shape[0]
    pc = []
    for i in range(b):
        a = x[i]
        c = y[i]
        # a = np.array(x[i])#.detach().cpu())
        # c = np.array(y[i])#.detach().cpu())
        # r = rdc(a, c)
        r = rdc(x[i], y[i])
        pc.append(np.exp(r))
    p = np.array(pc)
    p = torch.from_numpy(np.abs(p))
    return p.to('cuda')
def weight_dot(x, y):
    b = y.shape[0]
    pc = []
    for i in range(b):
        pc.append((y[i] * x[i]))
    p = torch.stack(pc, dim=0)
    # p = torch.from_numpy(p)
    return p
# let's define a simple model that can deal with multimodal variable length sequence

class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, q):
        k = x
        v = x
        B, N  = x.shape

        attn = (q @ k.transpose(-1, 0)) * self.scale
        attn = attn.softmax(dim=-1)

        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print(self.alpha)
        x = (attn @ v).transpose(0, 1).reshape(B, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, q):
        data = self.attn(self.norm1(x), q)
        x = x + self.drop_path(data)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.feature_dims[0]
        self.visual_size = config.feature_dims[2]
        self.acoustic_size = config.feature_dims[1]


        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes if config.train_mode == "classification" else 1
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        self.use_trans = False
        # print('aaa')
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        if self.config.use_bert:
            # model_name = r'D:\Speech\codes\MISA-master_2\MISA-master\bert-base-uncased'# 'bert-base-uncased'#'./roberta'
            # model_name = 'bert-base-uncased'
            # Initializing a BERT bert-base-uncased style configuration
            # bertconfig = BertConfig.from_pretrained(model_name, output_hidden_states=True)
            self.bertmodel = BertTextEncoder(use_finetune=config.use_finetune, transformers=config.transformers, pretrained=config.pretrained)
        else:
            self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)



        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        if self.use_trans:
            self.project_v = nn.Sequential()
            self.project_v.add_module('project_v', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.project_v.add_module('project_v_activation', self.activation)
            self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

            self.project_a = nn.Sequential()
            self.project_a.add_module('project_a', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.project_a.add_module('project_a_activation', self.activation)
            self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_v = nn.Sequential()
            self.project_v.add_module('project_v',
                                      nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
            self.project_v.add_module('project_v_activation', self.activation)
            self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

            self.project_a = nn.Sequential()
            self.project_a.add_module('project_a',
                                      nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
            self.project_a.add_module('project_a_activation', self.activation)
            self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size*1))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        # self.private_t.add_module('project_t_layer_norm_2', nn.LayerNorm(config.hidden_size))
        # self.private_t.add_module('private_t_2',
        #                           nn.Linear(in_features=config.hidden_size*3, out_features=config.hidden_size))
        # self.private_t.add_module('private_t_activation_2', nn.Sigmoid())

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        # self.private_v.add_module('private_v_layer_norm_2', nn.LayerNorm(config.hidden_size))
        # self.private_v.add_module('private_v_2',
        #                           nn.Linear(in_features=config.hidden_size*3, out_features=config.hidden_size))
        # self.private_v.add_module('private_v_activation_2', nn.Sigmoid())

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_1', nn.Sigmoid())
        # self.private_a.add_module('private_a_2',
        #                           nn.Linear(in_features=config.hidden_size*3, out_features=config.hidden_size))
        # self.private_a.add_module('private_a_activation_2', nn.Sigmoid())
        # self.private_a.add_module('project_a_layer_norm_2', nn.LayerNorm(config.hidden_size))
        # ##############################
        # self.private_a.add_module('project_a_2',
        #                           nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        # self.private_a.add_module('project_a_activation_2', self.activation)
        #
        # self.private_v.add_module('project_v_2',
        #                           nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        # self.project_v.add_module('project_v_activation_2', self.activation)
        #
        #
        # self.private_t.add_module('project_t_2',
        #                           nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        # self.private_t.add_module('project_t_activation_2', self.activation)
        #

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1',
                               nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 1))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        self.shared1 = nn.Sequential()

        self.shared1.add_module('shared1_1',
                               nn.Linear(in_features=config.hidden_size * 2, out_features=config.hidden_size* 2))
        self.shared1.add_module('shared1_activation_1', nn.Sigmoid())
        # self.shared1.add_module('shared1_layer_norm_1', nn.LayerNorm(config.hidden_size * 1))

        # self.shared.add_module('shared__layer_norm_1', nn.LayerNorm(config.hidden_size*3))
        # self.shared.add_module('shared_2', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        # self.shared.add_module('shared_activation_2', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))

        # self.commlayer_norm = nn.LayerNorm(self.config.hidden_size)
        # self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))
        # self.commlin = nn.Linear(in_features=self.config.hidden_size , out_features=self.config.hidden_size * 1)

        # self.transformer_encoder1 = Block(dim=self.config.hidden_size, num_heads=2, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
        #          drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)#nn.TransformerEncoder(encoder_layer, num_layers=1)
        # self.transformer_encoder2 = Block(dim=self.config.hidden_size, num_heads=2, mlp_ratio=4., qkv_bias=False,
        #                                   drop=0., attn_drop=0.,
        #                                   drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        # self.transformer_encoder3 = Block(dim=self.config.hidden_size, num_heads=2, mlp_ratio=4., qkv_bias=False,
        #                                   drop=0., attn_drop=0.,
        #                                   drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        #
        self.outT = nn.Linear(in_features=self.config.hidden_size, out_features=output_size)
        self.outV = nn.Linear(in_features=self.config.hidden_size, out_features=output_size)
        self.outA = nn.Linear(in_features=self.config.hidden_size, out_features=output_size)
        self.outC = nn.Linear(in_features=self.config.hidden_size, out_features=output_size)

        # self.w = nn.Parameter(torch.ones(3))
        # encoder_layerL = nn.TransformerEncoderLayer(d_model=768, nhead=2)
        # encoder_layerV = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        # encoder_layerA = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        encoder_layerC = nn.TransformerEncoderLayer(d_model=self.config.hidden_size*2, nhead=2)
        # self.transformer_encoderL = nn.TransformerEncoder(encoder_layerL, num_layers=3)
        # self.transformer_encoderV = nn.TransformerEncoder(encoder_layerV, num_layers=2)
        # self.transformer_encoderA = nn.TransformerEncoder(encoder_layerA, num_layers=2)
        self.transformer_encoderC = nn.TransformerEncoder(encoder_layerC, num_layers=1)
        self.prenetV = nn.Linear(20, self.config.hidden_size)  # for mosi
        # self.prenetV = nn.Linear(35, self.config.hidden_size) # for mosei
        self.prenetA = nn.Linear(5, self.config.hidden_size)
        self.prenetT = nn.Linear(300, self.config.hidden_size)
        # self.fcV = nn.Linear(35 * 256, self.config.hidden_size)
        # self.fcA = nn.Linear(35 * 256, self.config.hidden_size)
        # self.fcT = nn.Linear(35 * 256, self.config.hidden_size)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = padded_h1.permute(1, 0, 2)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2
    def extract_featuresbytransformer(self, sequence, lengths, prenet, transformerM):
        pre = prenet(sequence)
        # packed_sequence = pack_padded_sequence(sequence, lengths.cpu())
        final_h1 =transformerM(pre)
        return final_h1.transpose(0, 1)
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.p_t)
        self.utt_v = (self.utt_private_v + self.p_v)
        self.utt_a = (self.utt_private_a + self.p_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)
    def alignment(self, sentences, visual, acoustic):
        bert_sent, bert_sent_mask, bert_sent_type = sentences[:, 0, :], sentences[:, 1, :], sentences[:, 2, :]

        batch_size = sentences.size(0)


        if self.config.use_bert:
            bert_output = self.bertmodel(sentences)

            # bert_output = bert_output[0]
            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)

            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)

            if self.use_trans:
                bert_output = self.transformer_encoderL(masked_output).mean(dim=1)
            else:
                bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
            utterance_text = bert_output
        # else:
        #     # extract features from text modality
        #     sentences = self.embed(sentences)
        #
        #     final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        #     utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        lengths = mask_len.squeeze().int().detach().cpu().view(-1)
        if not self.use_trans:
            # # # extract features from visual modality
            final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
            utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
            #
            # # extract features from acoustic modality
            final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
            utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        else:
            # extract features from visual modality by transformer
            utterance_video = self.extract_featuresbytransformer(visual, lengths, self.prenetV, self.transformer_encoderV).mean(dim=0)

            # extract features from acoustic modality by transformer
            utterance_audio = self.extract_featuresbytransformer(acoustic, lengths, self.prenetA, self.transformer_encoderA).mean(dim=0)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)


        # 1-LAYER TRANSFORMER FUSION
        self.utt_shared_t = self.shared(self.utt_t_orig)
        self.utt_shared_v = self.shared(self.utt_v_orig)
        self.utt_shared_a = self.shared(self.utt_a_orig)
        t = self.utt_shared_t.detach().cpu().numpy()
        a = self.utt_shared_a.detach().cpu().numpy()
        v = self.utt_shared_v.detach().cpu().numpy()

        nanct = np.isnan(t)
        nanca = np.isnan(a)
        nancv = np.isnan(v)
        if (nanct == True).any():
            if (nanca == True).any():
                if (nancv == True).any():
                    print('nan')
        self.p_1 = rdc_cal(self.utt_shared_a, self.utt_shared_t)
        self.p_2 = rdc_cal(self.utt_shared_v, self.utt_shared_t)
        self.p_3 = rdc_cal(self.utt_shared_a, self.utt_shared_v)
        self.p_t = weight_dot(self.p_1 + self.p_2, self.utt_shared_t)
        self.p_a = weight_dot(self.p_1 + self.p_3, self.utt_shared_a)
        self.p_v = weight_dot(self.p_3 +self. p_2, self.utt_shared_v)
        self.utt_shared1_t = self.shared1(torch.cat((self.p_t, self.utt_private_t), dim=1))
        self.utt_shared1_v = self.shared1(torch.cat((self.p_v, self.utt_private_v), dim=1))
        self.utt_shared1_a = self.shared1(torch.cat((self.p_a, self.utt_private_a), dim=1))


        self.commen = self.utt_shared_a + self.utt_shared_v + self.utt_shared_t
        # self.commen = torch.cat((self.utt_shared_a, self.utt_shared_v, self.utt_shared_t), dim=1)
        # self.commen = torch.cat((self.p_t, self.p_a, self.p_v), dim=1)
        # self.commen = torch.cat((self.utt_private_t, self.utt_private_a, self.utt_private_v), dim=1)
        # self.commen = self.commlin(self.commen)
        # self.commen = self.commlayer_norm(self.commen)
        self.conT = self.outT(self.utt_private_t)
        self.conA = self.outA(self.utt_private_a)
        self.conV = self.outV(self.utt_private_v)
        self.conC = self.outC(self.commen)

        # self.mergeav = self.transformer_encoder1(self.utt_private_a, self.utt_shared1_a)
        # self.mergeva = self.transformer_encoder2(self.utt_private_v, self.utt_shared1_v)
        # self.mergeat = self.transformer_encoder3(self.utt_private_t, self.utt_shared1_t)
        # For reconstruction
        self.reconstruct()
        # h = torch.cat((self.mergeav, self.mergeva, self.mergeat), dim=1)
        h = torch.stack((self.utt_shared1_a, self.utt_shared1_v, self.utt_shared1_t), dim=0)
        h = self.transformer_encoderC(h)#.squeeze()
        h = torch.cat((h[0], h[1], h[2]), dim=1)
        # print('aaa')
        o = self.fusion(h)
        # mm = MinMaxScaler((-3, 3))
        # data = mm.fit_transform(o.cpu().detach())

        return o
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)


        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)



    def forward(self, text, audio, video):

        o = self.alignment(text, video, audio)
        tmp = {
            "M": o
        }
        return tmp