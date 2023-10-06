import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import Ridge
from os.path import join as pjoin
from nilearn.glm.first_level import make_first_level_design_matrix
from had_utils import save2cifti

# define path
# make sure the dataset_path are modified based on your personal dataset downloading directory
dataset_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/data_upload/HAD'
ciftify_path = f'{dataset_path}/derivatives/ciftify'
nifti_path = f'{dataset_path}'
support_path = './support_files'
result_path = './results'
save_path = pjoin(result_path, 'beta')
if not os.path.exists(save_path):
    os.makedirs(save_path)
# change to path of current file
os.chdir(os.path.dirname(__file__))

# prepare params
sub_names = ['sub-%02d'%(i+1) for i in range(30)]
# sub_names = ['sub-24']
tr, begin_dur, n_tr, n_event, n_run, n_cycle, n_class = 2, 12, 156, 60, 12, 4, 180
frame_times = np.arange(n_tr*3) * tr 
sess_name = 'ses-action01'
alpha = 0.1

# start modeling
for sub_idx, sub_name in enumerate(sub_names):
    # prepare basic path
    sub_nifti_path = pjoin(nifti_path, sub_name)
    result_path = pjoin(ciftify_path, sub_name, 'results')
    sub_func_path = pjoin(sub_nifti_path, sess_name, 'func')
    events_file = sorted([i for i in os.listdir(sub_func_path) if i.endswith('.csv')])
    # loop in cycles to perform linear regression model
    for cycle_idx in range(n_cycle):
        beta = np.zeros((n_class, 59412), dtype=np.float32)
        # intial containers
        trial_type_cycle = []
        stim_names_cycle = []
        dtseries_cycle = np.zeros((3*n_tr, 91282))
        onset_cycle = np.zeros((3*n_event))
        duration_cycle = np.zeros((3*n_event))
        for run_idx in range(3):
            run_file_idx = run_idx + cycle_idx * 3 + 1
            events_file_name = '%s_ses-action01_task-action_run-%02d_events.tsv'%(sub_name, run_file_idx)
            run_name = 'ses-action01_task-action_run-%d'% run_file_idx
            # fit design matrix based on trial onset time
            events_raw = pd.read_csv(pjoin(sub_func_path, events_file_name), sep='\t')
            duration = events_raw['duration']
            onset = events_raw['onset'].to_numpy() + begin_dur
            label_tmp = events_raw['trial_type'].to_numpy()
            trial_type = ['category-%03d'%idx for idx in label_tmp]
            stim_names = [_.split('/')[-1] for _ in events_raw['stim_file']]
            # load time series
            dtseries_path = pjoin(result_path, f'{run_name}_Atlas.dtseries.nii')
            dtseries = nib.load(dtseries_path).get_fdata()
            # concantenate all info into cycle params
            dtseries_cycle[n_tr*run_idx:n_tr*(run_idx+1)] = dtseries
            onset_cycle[n_event*run_idx:n_event*(run_idx+1)] = onset + run_idx * n_tr * tr
            duration_cycle[n_event*run_idx:n_event*(run_idx+1)] = duration
            trial_type_cycle.extend(trial_type)
            stim_names_cycle.extend(stim_names)
        # prepare design matrix
        events = pd.DataFrame({'trial_type':trial_type_cycle, 'onset':onset_cycle, 'duration':duration_cycle})
        design_matrix = make_first_level_design_matrix(frame_times, events, drift_model=None, hrf_model='spm')
        design_matrix.drop(design_matrix.columns[-1], axis=1, inplace=True)
        # add drift columns
        drift_order = 2
        frame_times_single = np.arange(n_tr) * tr 
        drift_effect = np.zeros((n_tr * 3, 3*(drift_order+1)))
        tmax = float(frame_times_single.max())
        for run_idx in range(3):
            for k in range(drift_order+1):
                drift_effect[n_tr*run_idx:n_tr*(run_idx+1), (drift_order+1)*run_idx+k] = (frame_times_single / tmax) ** k
        drift_effect = pd.DataFrame(drift_effect)
        # concantenate 
        design_matrix = pd.concat([design_matrix.reset_index(drop=True), drift_effect], ignore_index=True, axis=1)  
        # perform GLM
        reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries_cycle[:, :59412])
        beta = reg.coef_[:, :n_class].transpose(1,0).astype(np.float32)
        print('Finish performing GLM in %s %s model %02d in alpha %.3f'%(sub_name, sess_name, cycle_idx+1, alpha))
        # save beta and label information
        beta_path = pjoin(result_path, f'ses-action01_task-action_cycle-{cycle_idx+1}_beta.dscalar.nii')
        label_path = pjoin(result_path, f'ses-action01_task-action_cycle-{cycle_idx+1}_label.txt')
        # sort stim names info and write into txt 
        stim_names_cycle.sort(key=lambda x:x.split('_')[1])
        with open(label_path, 'w') as f:
            f.writelines([line+'\n' for line in stim_names_cycle])
        # save beta
        beta_path = pjoin(save_path, f'{sub_name}_cycle-{cycle_idx+1}_beta.dscalar.nii')
        temp = nib.load(pjoin(support_path, 'template.dtseries.nii'))
        bm = list(temp.header.get_index_map(1).brain_models)[0:2]
        save2cifti(file_path=beta_path, data=beta, brain_models=bm, map_names=stim_names_cycle)



import numpy as np
import pandas as pd
import scipy.io as sio
from os.path import join as pjoin
import statsmodels.api as sm
from statsmodels.formula.api import ols

main_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/coding_principle/visual'
stim_path = pjoin(main_path, 'stim')
bold_path = pjoin(main_path, 'data', 'feature')
map_path = pjoin(main_path, 'data', 'cortex_map')

# load animacy and size definition
size_path = pjoin(stim_path, 'size_annotation.xlsx')
animacy_path = pjoin(stim_path, 'imagenet_animate_or_not.mat')
size = pd.read_excel(size_path, usecols=[3]).values.squeeze()
animacy = sio.loadmat(animacy_path)['animate_label'].squeeze()

# load brain signal and make contrast
feature = np.load(pjoin(bold_path, 'sub-01-10_imagenet-feature-visual_area_with_ass.npy'))
big_loc, small_loc = np.where(size==2), np.where(size==1)
animate_loc, inanimate_loc = np.where(animacy==1), np.where(animacy==-1)
size_map = np.mean(feature[small_loc], axis=0) - np.mean(feature[big_loc], axis=0) # small - big
animacy_map = np.mean(feature[animate_loc], axis=0) - np.mean(feature[inanimate_loc], axis=0) # animate - inanimate
np.save(pjoin(map_path, 'contrast-cortex_map-animacy.npy'), animacy_map)
np.save(pjoin(map_path, 'contrast-cortex_map-size.npy'), size_map)
        
# make anova model
p_values = pd.DataFrame(columns=['animacy', 'size', 'interaction'])
for voxel_idx in range(feature.shape[1]): #
    signal = feature[:, voxel_idx]
    # generate dataframe
    # condition order: animate-big, animate-small, inanimate-big, inanimate-small
    df = pd.DataFrame.from_dict({'animacy':animacy, 'size':size, 'signal': signal})
    # perform two-way ANOVA
    model = ols('signal ~ C(animacy) + C(size) + C(animacy):C(size)', data=df).fit()
    tabel = sm.stats.anova_lm(model, typ=2)
    p_values.loc[voxel_idx] = tabel.iloc[:3, 3].tolist()
    print(f'finish voxel {voxel_idx}')
p_values.to_csv(pjoin(map_path, 'contrast-p_values.csv'), index=False)


#%% generate three maps: main fact: animacy and size; interaction maps
import numpy as np
import pandas as pd
from os.path import join as pjoin

main_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/coding_principle/visual'
stim_path = pjoin(main_path, 'stim')
bold_path = pjoin(main_path, 'data', 'feature')
map_path = pjoin(main_path, 'data', 'cortex_map')

animacy_map = np.load(pjoin(map_path, 'contrast-cortex_map-animacy.npy'))
size_map = np.load(pjoin(map_path, 'contrast-cortex_map-size.npy'))
p_values = pd.read_csv(pjoin(map_path, 'contrast-p_values.csv'))

sig = 0.001
# define interaction map
interaction = np.zeros(animacy_map.shape)
for voxel_idx in np.where(p_values.iloc[:, 1].values < sig)[0]:
    animacy_value = animacy_map[voxel_idx]
    size_value = size_map[voxel_idx]
    if (animacy_value > 0) & (size_value < 0): # animate - big
        interaction[voxel_idx] = 1
    elif (animacy_value < 0) & (size_value < 0): # inanimate - big
        interaction[voxel_idx] = 2
    elif (animacy_value < 0) & (size_value > 0): # inanimate - small
        interaction[voxel_idx] = 3
    elif (animacy_value > 0) & (size_value > 0): # animate - small
        interaction[voxel_idx] = 4
    print(f'finish voxel {voxel_idx}')

# define main fact map
animacy_map[p_values.iloc[:, 0].values >= sig] = 0
size_map[p_values.iloc[:, 1].values >= sig] = 0
np.save(pjoin(map_path, f'contrast-cortex_map-animacy-sig_{sig}.npy'), animacy_map)
np.save(pjoin(map_path, f'contrast-cortex_map-size-sig_{sig}.npy'), size_map)
np.save(pjoin(map_path, f'contrast-cortex_map-interaction-sig_{sig}.npy'), interaction)


