from MMSA import MMSA_run
import pickle
import numpy as np
import pandas as pd
import os

#CMU-MOSI
if __name__ == '__main__':
    dropout = 0.1
    MMSA_run(model_name='FGTI', discription='FGTI',
                 dropout=dropout
                 , dataset_name='mosi',
                 seeds=[1900], config_file='config/config_regression.json', gpu_ids=[0])  # 0.1
