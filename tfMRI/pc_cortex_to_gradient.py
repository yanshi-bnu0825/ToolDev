import numpy as np
import nibabel as nib
from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.gradient import GradientMaps

def save_ciftifile(data, filename):
    template = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/Analysis_derivatives/ciftify/sub-core02/MNINonLinear/Results/ses-ImageNet01_task-object_run-1/ses-ImageNet01_task-object_run-1_Atlas.dtseries.nii'
    ex_cii = nib.load(template)
    if len(data.shape) > 1:
        ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    else:
        ex_cii.header.get_index_map(0).number_of_series_points = 1
        data = data[np.newaxis, :]
    nib.save(nib.Cifti2Image(data.astype(np.float32), ex_cii.header), filename)
    
# define path
main_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/coding_principle/visual'
decoding_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/Analysis_results/imagenet_decoding/voxel'
data_path = pjoin(main_path, 'data')
hierarchy_path = pjoin(main_path, 'hierarchy')
axes_path = pjoin(data_path, 'axes')
map_path = pjoin(data_path, 'cortex_map')
feature_path = pjoin(data_path, 'feature')
surface_path = pjoin(main_path, 'voxel')
result_path = pjoin(main_path, 'results')
target = 'whole_brain_select'
method = 'pca'
n_pc = 4

# load data
cortex_map = np.load(pjoin(map_path, f'pc-cortex_map-{target}-{method}.npy'))
# feature = np.load(pjoin(feature_path, 'sub-01-10_imagenet-feature-VTC_more.npy')).transpose(1,0)
# vertex_point = np.load(pjoin(surface_path, f'{target}_32k_map.npy'))
voxel_loc = np.load(pjoin(decoding_path, f'sub-01-10_imagenet-{target}_idx.npy'))

for pc_idx in range(n_pc):
    # voxel_loc = nib.load(pjoin(surface_path, f'{target}_more.dtseries.nii')).get_fdata()
    tmp_map = np.zeros(91282)
    pc_map = cortex_map[:, pc_idx]
    tmp_map[voxel_loc] = pc_map
    save_ciftifile(tmp_map, pjoin(surface_path, f'cortex_map-{target}-{method}-pc{pc_idx+1}.dtseries.nii'))

# save 1000 categories map
voxel_loc = nib.load(pjoin(surface_path, f'{target}_more.dtseries.nii')).get_fdata().squeeze()
map_save = np.zeros((voxel_loc.shape[0], feature.shape[1]))
for idx in range(feature.shape[1]):
    map_save[:, idx] = voxel_loc
    map_save[map_save[:, idx] == 1, idx] = feature[:, idx]
save_ciftifile(map_save.transpose(1,0), pjoin(surface_path, f'cortex_map-1000-categories.dtseries.nii'))

generate contrast map in pc1 and pc2
voxel_loc = nib.load(pjoin(surface_path, f'{target}_more.dtseries.nii')).get_fdata()
animacy_map = cortex_map[:, 0]
size_map = cortex_map[:, 1]
interaction_map = np.zeros(animacy_map.shape)
print(interaction_map.shape)
for voxel_idx in range(animacy_map.shape[0]):
    animacy_value = animacy_map[voxel_idx]
    size_value = size_map[voxel_idx]
    if (animacy_value > 0) & (size_value < 0): # animate - big
        interaction_map[voxel_idx] = 1
    elif (animacy_value > 0) & (size_value > 0): # animate - small
        interaction_map[voxel_idx] = 2
    elif (animacy_value < 0) & (size_value > 0): # inanimate - small
        interaction_map[voxel_idx] = 3
    elif (animacy_value < 0) & (size_value < 0): # inanimate - big 
        interaction_map[voxel_idx] = 4
    print(f'finish voxel {voxel_idx}')
    
voxel_loc[voxel_loc == 1] = interaction_map
save_ciftifile(voxel_loc, pjoin(surface_path, f'cortex_map-{target}-{method}-interaction.dtseries.nii'))

# load surface map
surf_lh, surf_rh = load_conte69()

# Merge data and transform surface space
cortex_32k_map = np.zeros((vertex_point.shape[0], n_pc))
for pc_idx in range(n_pc):
    pc_map = cortex_map[:, pc_idx]
    vertex_point[np.where(np.isnan(vertex_point)==False)[0]] = pc_map 
    cortex_32k_map[:, pc_idx] = vertex_point
cortex_32k_map = [cortex_32k_map[:, x] for x in range(n_pc)]

# plot_hemispheres(surf_lh, surf_rh, array_name=cortex_32k_map, size=(1200, 800), 
#                  color_bar=True, label_text=[f'Grad{x+1}' for x in range(n_pc)], zoom=1.5, 
#                  nan_color=(0,0,0,0.5), cmap='seismic', background=(1,1,1), transparent_bg=False, 
#                  screenshot=True, filename=pjoin(result_path, f'gradient_map-{target}-{method}.jpg'))
    
surfs = {'lh': surf_lh, 'rh': surf_rh}
layout = ['lh', 'rh', 'rh', 'lh']
view = ['lateral', 'medial', 'ventral', 'ventral']
share = 'r'

array_name=cortex_32k_map
layout = [layout] * len(array_name)
array_name2 = []
n_pts_lh = surf_lh.n_points
for an in array_name:
    if isinstance(an, np.ndarray):
        name = surf_lh.append_array(an[:n_pts_lh], at='p')
        surf_rh.append_array(an[n_pts_lh:], name=name, at='p')
        array_name2.append(name)
    else:
        array_name2.append(an)
array_name = np.asarray(array_name2)[:, None]

plot_surf(surfs, layout=layout, array_name=array_name, size=(1000, int(n_pc*200)), zoom=1.2, view=view, 
          color_bar=True, label_text=[f'Grad{x+1}' for x in range(n_pc)], share=share, color_range =(-2.72, 2.72), 
          nan_color=(0,0,0,0.5), cmap='seismic', background=(1,1,1), transparent_bg=False, 
          screenshot=True, filename=pjoin(result_path, f'gradient_map-{target}-{method}.jpg'))
