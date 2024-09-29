from MMSA import MMSA_run
import pickle
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    dropout = 0.6
    MMSA_run(model_name='FGTI',
             discription='FGTI-MOSEI'
             ,dropout=dropout, dataset_name='mosei',
             seeds=[1111], config_file='config/config_regression.json', gpu_ids=[0])
    