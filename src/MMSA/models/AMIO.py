"""
AMIO -- All Model in One
"""
import torch.nn as nn

from .multiTask import *
from .singleTask import *
from .missingTask import *
from .subNets import AlignSubNet


class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            # single-task
            'tfn': TFN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'graph_mfn': Graph_MFN,
            'mctn': MCTN,
            'bert_mag': BERT_MAG,
            'mult': MULT,
            'fgti': FGTI,
            'misawordc': MISAwoRDC,
            'misaat': MISAAT,
            'misaav': MISAAV,
            'misavt': MISAVT,
            'misaaddition': MISAAddition,
            'misaconcatation': MISAConcatation,
            'misawoprivate': MISAwoPrivate,
            'misawocommen': MISAwoCommen,
            'mfm': MFM,
            'mmim': MMIM,
            # multi-task
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'self_mm': SELF_MM,
            # missing-task
            'tfr_net': TFR_NET
        }
        self.need_model_aligned = args.get('need_model_aligned', None)
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_model_aligned):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()
        lastModel = self.MODEL_MAP[args['model_name']]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)
