#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:16:16 2023

@author: yanshi
"""
import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import subprocess
import config
# %%
data_dir=config.data_dir
config_t1w_brain_path=config.t1w_brain_path
config_t2w_brain_path=config.t2w_brain_path
data_loss_sub=config.data_loss_sub
data_sub=config.data_sub
sublist = pd.read_csv(config.raw_sub_list_path, header=None)
config_t1w_brain_2_path=config.t1w_brain_2_path
config_t2w_brain_2_path=config.t2w_brain_2_path
template_mni_2mm_path=config.template_mni_2mm_path

# %% ====================================================
with tqdm(total=len(sublist)) as pbar:
    for i,row in sublist.iterrows():
        sub=str(row[0])
        t1w_brain_path=config_t1w_brain_path.format(sub)
        t2w_brain_path=config_t2w_brain_path.format(sub)
        if os.path.exists(t1w_brain_path) is False or os.path.exists(t2w_brain_path) is False:
            with open(data_loss_sub,mode='a',newline='') as fl:
                writer=csv.writer(fl)
                writer.writerow([sub])
        else:
            with open(data_sub,mode='a',newline='') as f:
                writer=csv.writer(f)
                writer.writerow([sub])
                t1w_brain_2_path=config_t1w_brain_2_path.format(sub)
                t2w_brain_2_path=config_t2w_brain_2_path.format(sub)
                if os.path.exists(t1w_brain_2_path) is False or os.path.exists(t2w_brain_2_path) is False:              
                    flirt_t1w = f'flirt -in {t1w_brain_path} -ref {template_mni_2mm_path} -applyisoxfm 2 -out {t1w_brain_2_path}'
                    flirt_t2w = f'flirt -in {t2w_brain_path} -ref {template_mni_2mm_path} -applyisoxfm 2 -out {t2w_brain_2_path}'           
                    subprocess.call(flirt_t1w, shell=True)  
                    subprocess.call(flirt_t2w, shell=True)
        pbar.update(1)
    

