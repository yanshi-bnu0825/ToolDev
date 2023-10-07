import os
import numpy as np
import pandas as pd
import scipy.io as sio
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join as pjoin
from had_utils import save2cifti, roi_mask
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable

# define path
# make sure the dataset_path are modified based on your personal dataset downloading directory
dataset_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/data_upload/HAD'
ciftify_path = pjoin(dataset_path, 'derivatives', 'ciftify')
stim_path = pjoin(dataset_path, 'stimuli')
support_path = './support_files'
# change to path of current file
os.chdir(os.path.abspath(''))

# prepare params
sub_names = ['sub-%02d'%(i+1) for i in range(30)]
# sub_names = ['sub-01']
n_cycle = 4
n_sub = len(sub_names)
# prepare beta
beta_sum = np.zeros((n_sub, 180, 59412))
for sub_idx, sub_name in enumerate(sub_names):
    # extract beta in each cycle results
    beta_sub = np.zeros((4, 180, 59412))
    for cycle_idx in range(n_cycle):
        beta_sub[cycle_idx] = nib.load(pjoin(ciftify_path, sub_name, 'results', f'ses-action01_task-action_cycle-{cycle_idx+1}_beta.dscalar.nii')).get_fdata()
    beta_sub = beta_sub.mean(axis=0)
    # scale data
    scaler = StandardScaler()
    beta_sum[sub_idx] = scaler.fit_transform(beta_sub)
    print(f'Finish loading beta: {sub_name}')
beta_sum = beta_sum.mean(axis=0)

# sort the class order to make categories in the same superordinate class presented adjacently
class_order = np.load(pjoin(support_path, 'class_order.npy'))
# save class order into csv
ordering_table = pd.DataFrame({'class_order':class_order})
ordering_table.to_csv(pjoin(support_path, 'class_order.csv'), index=False)
class_selected = os.listdir(stim_path)
class_selected.sort()

# resort the beta class index 
beta_sorted = np.zeros(beta_sum.shape)
for class_idx, class_name in enumerate(class_order):
    beta_sorted[class_idx] = beta_sum[class_selected.index(class_name)]

# define plot details
fig, ax = plt.subplots(1, len(region_sum), figsize=(70, 15))
# fig, ax = plt.subplots(1, len(region_sum), figsize=(7*1.5, 1.5*1.5))
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams.update({'font.size': 14, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
cmap = plt.cm.get_cmap('RdBu')
cmap = cmap.reversed()

rdm_sum = np.zeros((len(region_sum), 180, 180))
rdm_triu_sum = np.zeros((len(region_sum), int(180*179/2)))
for idx, check_roi in enumerate(region_sum):
    mask = mask_sum[idx]
    # specialized voxel loc
    class_pattern = beta_sorted[:, mask]
    # generate rdm
    rdm = np.corrcoef(class_pattern)
    np.fill_diagonal(rdm, 0)
    rdm_sum[idx] = rdm
    rdm_triu = rdm[np.triu_indices(rdm.shape[0], k=1)]
    rdm_triu_sum[idx] = rdm_triu
    # plot rdm 
    axes = ax[idx]
    im = axes.imshow(rdm, cmap=cmap, vmin=-0.8, vmax=0.8) 
    axes.set_axis_off()
# # shown colorbar if you want
# divider = make_axes_locatable(axes)
# cax = divider.append_axes('right', size='5%', pad=0.25)
# cbar = fig.colorbar(im, cax=cax, orientation='vertical')
# cbar.ax.tick_params(labelsize=30)
plt.show()