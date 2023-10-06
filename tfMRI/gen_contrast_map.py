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


