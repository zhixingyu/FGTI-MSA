import numpy as np
import torch, math
import torch.nn as nn
from ..subNets import BertTextEncoder
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence
from .rdc import rdc
from timm.models.layers import Mlp, DropPath
from scipy.stats import gamma
class HSIC(nn.Module):
    def __init__(self):
        super(HSIC, self).__init__()

    def forward(self, X, Y, alph = 0.8):
        """
        X, Y are numpy vectors with row - sample, col - dim
        alph is the significance level
        auto choose median to be the kernel width
        """
        n = X.shape[0]

        # ----- width of X -----
        Xmed = X

        G = np.sum(Xmed*Xmed, 1)
        G = G.reshape(n, 1)
        Q = np.tile(G, (1, n) )
        R = np.tile(G.T, (n, 1) )

        dists = Q + R - 2* np.dot(Xmed, Xmed.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n**2, 1)

        width_x = np.sqrt( 0.5 * np.median(dists[dists>0]) )
        # ----- -----

        # ----- width of X -----
        Ymed = Y

        G = np.sum(Ymed*Ymed, 1).reshape(n,1)
        Q = np.tile(G, (1, n) )
        R = np.tile(G.T, (n, 1) )

        dists = Q + R - 2* np.dot(Ymed, Ymed.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n**2, 1)

        width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
        # ----- -----

        bone = np.ones((n, 1), dtype = float)
        H = np.identity(n) - np.ones((n,n), dtype = float) / n

        K = self.rbf_dot(X, X, width_x)
        L = self.rbf_dot(Y, Y, width_y)

        Kc = np.dot(np.dot(H, K), H)
        Lc = np.dot(np.dot(H, L), H)

        testStat = np.sum(Kc.T * Lc) / n

        varHSIC = (Kc * Lc / 6)**2

        varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

        varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

        K = K - np.diag(np.diag(K))
        L = L - np.diag(np.diag(L))

        muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
        muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

        mHSIC = (1 + muX * muY - muX - muY) / n

        al = mHSIC**2 / varHSIC
        bet = varHSIC*n / mHSIC

        thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

        return (testStat, thresh)

    def rbf_dot(self, pattern1, pattern2, deg):
        size1 = pattern1.shape
        size2 = pattern2.shape

        G = np.sum(pattern1 * pattern1, 1).reshape(size1[0], 1)
        H = np.sum(pattern2 * pattern2, 1).reshape(size2[0], 1)

        Q = np.tile(G, (1, size2[0]))
        R = np.tile(H.T, (size1[0], 1))

        H = Q + R - 2 * np.dot(pattern1, pattern2.T)

        H = np.exp(-H / 2 / (deg ** 2))

        return H
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
        r = rdc(x[i], y[i])
        pc.append(np.exp(r))
    p = np.array(pc)
    p = torch.from_numpy(np.abs(p))
    return p.to('cuda')
def weight_dot(x, y):
    xx= torch.log(x)/torch.log(2)
    b = y.shape[0]
    pc = []
    for i in range(b):
        pc.append((y[i] * xx[i]))
    p = torch.stack(pc, dim=0)
    return p

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
        attn = self.attn_drop(attn)
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

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_1', nn.Sigmoid())

        self.shared = nn.Sequential()
        self.shared.add_module('shared_1',
                               nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 1))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        self.shared1 = nn.Sequential()

        self.shared1.add_module('shared1_1',
                               nn.Linear(in_features=config.hidden_size * 2, out_features=config.hidden_size* 2))
        self.shared1.add_module('shared1_activation_1', nn.Sigmoid())

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))

        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))

        self.outT = nn.Linear(in_features=self.config.hidden_size, out_features=output_size)
        self.outV = nn.Linear(in_features=self.config.hidden_size, out_features=output_size)
        self.outA = nn.Linear(in_features=self.config.hidden_size, out_features=output_size)
        self.outC = nn.Linear(in_features=self.config.hidden_size, out_features=output_size)
        encoder_layerC = nn.TransformerEncoderLayer(d_model=self.config.hidden_size*2, nhead=2)
        self.attnn = Attention(self.config.hidden_size)

        self.transformer_encoderC = nn.TransformerEncoder(encoder_layerC, num_layers=1)
        self.prenetV = nn.Linear(20, self.config.hidden_size)  # for mosi
        # self.prenetV = nn.Linear(35, self.config.hidden_size) # for mosei
        self.prenetA = nn.Linear(5, self.config.hidden_size)
        self.prenetT = nn.Linear(300, self.config.hidden_size)
        self.hsic=HSIC()
        self.mha = Block(dim=self.config.hidden_size, num_heads=8)
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

        bert_output = self.bertmodel(sentences)
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        if self.use_trans:
            bert_output = self.transformer_encoderL(masked_output).mean(dim=1)
        else:
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        utterance_text = bert_output
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
        self.p_1 = self.hsic(self.utt_shared_a, self.utt_shared_t)
        self.p_2 = self.hsic(self.utt_shared_v, self.utt_shared_t)
        self.p_3 = self.hsic(self.utt_shared_a, self.utt_shared_v)
        self.p_t = weight_dot(self.p_1 + self.p_2, self.utt_shared_t)
        self.p_a = weight_dot(self.p_1 + self.p_3, self.utt_shared_a)
        self.p_v = weight_dot(self.p_3 +self. p_2, self.utt_shared_v)

        self.commen = self.utt_shared_a + self.utt_shared_v + self.utt_shared_t
        self.conT = self.outT(self.utt_private_t)
        self.conA = self.outA(self.utt_private_a)
        self.conV = self.outV(self.utt_private_v)
        self.conC = self.outC(self.commen)
        self.reconstruct()
        p1 = torch.cat((self.utt_shared_a[0], self.utt_shared_v[1], self.utt_shared_t[2]))
        h = torch.stack((self.utt_shared_a, self.utt_shared_v, self.utt_shared_t), dim=0)
        h = self.attnn(h)
        p2 = torch.cat((h[0], h[1], h[2]), dim=1)
        p = p1+p2
        h2 = torch.stack((self.private_a, self.private_v, self.private_t), dim=0)
        h = self.mha(h2,p)
        h = torch.cat((h[0], h[1], h[2]), dim=1)
        o = self.fusion(h)
        return o

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
