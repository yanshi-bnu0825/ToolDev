#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:21:20 2023

@author: yanshi
"""


# %%
import nibabel as nib
import os
import subprocess
import numpy as np
import pandas as pd
#import cb_tools 
from tqdm import tqdm
import config

# %% =================================================== 
# compute t1w/t2w
# without mask - original space
'''
with tqdm(total=len(sublist)) as pbar:
    # test version:
    for sub in ['996782','994273']:
    #for sub in sublist['Sub']:
        # orig space
        data_sub_path = os.path.join(save_dir_indi, f'{index}_{sub}.nii.gz')
        if os.path.exists(data_sub_path) is False:

            t1w_brain_path = os.path.join(data_dir, sub, 'MNINonLinear', 'T1w_restore_brain.nii.gz')
            t2w_brain_path = os.path.join(data_dir, sub, 'MNINonLinear', 'T2w_restore_brain.nii.gz')
            t1w = nib.load(t1w_brain_path).get_fdata() 
            t2w = nib.load(t2w_brain_path).get_fdata()
            
            ## -------------------------------------
            # # thr 1.5 IQR
            # t1w = cb_tools.thr_IQR(t1w, times=1.5)
            # t2w = cb_tools.thr_IQR(t2w, times=1.5)
            ratio = t1w / t2w  
            
            # # 2nd thr
            # ratio = cb_tools.thr_IQR(ratio, times=1.5)
            ## -------------------------------------

            ratio = np.nan_to_num(np.asarray(ratio))
            img_temp = nib.Nifti1Image(ratio, None, nib.load(t1w_brain_path).header)
            nib.save(img_temp, data_sub_path)

        pbar.update(1)

print('compute t1w/t2w - without mask - original space done')
'''
data_dir=config.data_dir
config_t1w_brain_path=config.t1w_brain_path
data_sub=config.data_sub
sublist = pd.read_csv(data_sub, header=0)
config_t1w_brain_2_path=config.t1w_brain_2_path
config_t2w_brain_2_path=config.t2w_brain_2_path
config_myeline_file_path=config.myeline_file_path
# %% =================================================== 
# compute t1w/t2w
# without mask - 2mm
with tqdm(total=len(sublist)) as pbar:
    # test version:
    #for sub in ['996782','994273']:
    for i,row in sublist.iterrows():
        sub=str(row[0])
        # 2mm
        myeline_file_path=config_myeline_file_path.format(sub)
        t1w_brain_2_path=config_t1w_brain_2_path.format(sub)
        t2w_brain_2_path=config_t2w_brain_2_path.format(sub)
        if os.path.exists(myeline_file_path) is False:
            t1w = nib.load(t1w_brain_2_path).get_fdata() 
            t2w = nib.load(t2w_brain_2_path).get_fdata()
            
            ## -------------------------------------
            # # thr 1.5 IQR
            # t1w = cb_tools.thr_IQR(t1w, times=1.5)
            # t2w = cb_tools.thr_IQR(t2w, times=1.5)
            ratio = t1w / t2w  
            
            # # 2nd thr
            # ratio = cb_tools.thr_IQR(ratio, times=1.5)
            ## -------------------------------------
            ratio = np.nan_to_num(np.asarray(ratio))
            t1w_brain_path=config_t1w_brain_path.format(sub)
            img_temp = nib.Nifti1Image(ratio, None, nib.load(t1w_brain_path).header)
            nib.save(img_temp,myeline_file_path)
        pbar.update(1)
print('compute t1w/t2w - without mask - 2mm done')
