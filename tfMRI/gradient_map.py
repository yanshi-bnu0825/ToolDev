import numpy as np
from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure

# load data
data_path = '/home/ubuntu/Project/data'
hierarchy_path = pjoin(data_path, 'hierarchy')
feature_path = pjoin(data_path, 'cortex_map')
axes_path = pjoin(data_path, 'hypothesized_axes')
surface_path = pjoin(data_path, 'visual_mask')
result_path = '/home/ubuntu/Project/result/gradient_map'
target = 'visual_area_with_ass'
# compute rdm
feature_matrix = np.load(pjoin(feature_path, f'sub-01-10_imagenet-feature-{target}.npy'))
feature_matrix = feature_matrix.transpose((1,0))
# voxel_rdm = np.corrcoef(feature_matrix)

#%% get surface corresponding
import nibabel as nib

with open(pjoin(surface_path, 'vertex_left.txt')) as f:
    vertex_left_list = f.read().splitlines()
with open(pjoin(surface_path, 'vertex_right.txt')) as f:
    vertex_right_list = f.read().splitlines()
# convert to ndarray 
vertex_left = np.array([[int(x.split(' ')[0]), int(x.split(' ')[1])] for x in vertex_left_list])
vertex_right = np.array([[int(x.split(' ')[0]), int(x.split(' ')[1])] for x in vertex_right_list])
del vertex_left_list, vertex_right_list
# get visual_mask
visual_mask = nib.load(pjoin(surface_path, 'HCP-MMP1_visual-cortex1.dlabel.nii')).get_fdata().squeeze()

#%% Merge feature matrix into ROI level
roi_label = visual_mask[np.isnan(visual_mask)==False]
n_roi = np.unique(roi_label).shape[0]
n_class = 1000
roi_matrix = np.zeros((n_roi, n_class))
# loop to merge matrix
for idx,roi_id in enumerate(np.unique(roi_label)):
    roi_loc = roi_label == roi_id
    roi_matrix[idx, :] = np.mean(feature_matrix[roi_loc, :], axis=0)
np.save(pjoin(feature_path, 'roi_matrix.npy'), roi_matrix)

#%% plot surface
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels

# load surface map
surf_lh, surf_rh = load_conte69()

# Map gradients to original parcels
# get gradient values
# map 59412 surface space to 32k space
grad_tmp = np.full(surf_lh.n_points+surf_rh.n_points, np.nan)
grad_tmp[vertex_left[:, 1]] = visual_mask[:vertex_left.shape[0]]
grad_tmp[vertex_right[:, 1]+surf_lh.n_points] = visual_mask[-vertex_right.shape[0]:]
np.save(pjoin(surface_path, f'{target}_32k_map.npy'), grad_tmp)
# tranfer in ROI label: make 126 ROIs index from 0-126
for idx, x in enumerate(np.unique(roi_label)):
    grad_tmp[grad_tmp == x] = idx+1 

# compute gradient map
n_components = 4
method = 'dm'
# define rdm
visualize_level = 'voxel'
if visualize_level == 'ROI':
    rdm = np.corrcoef(roi_matrix)
else:
    rdm = np.corrcoef(feature_matrix)
# fit maps
gm = GradientMaps(n_components=n_components, random_state=0, approach=method)
gm.fit(rdm)
# get all gradients
grad = [None] * n_components
for i, g in enumerate(gm.gradients_.T):
    if visualize_level == 'ROI':
        grad[i] = map_to_labels(g, grad_tmp, mask=np.isnan(grad_tmp)==False, fill=np.nan)
    else:
        grad_tmp[np.where(np.isnan(grad_tmp)==False)[0]] = g
        grad[i] = grad_tmp
np.save(pjoin(feature_path, f'gradient_map-{method}.npy'), gm.gradients_)
        
# plot surface
surfs = {'lh': surf_lh, 'rh': surf_rh}
layout = ['lh', 'rh', 'rh', 'lh']
view = ['lateral', 'medial', 'ventral', 'ventral']
share = 'r'

array_name = grad
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

plot_surf(surfs, layout=layout, array_name=array_name, size=(1000, int(n_components*200)), zoom=1.2, view=view, 
          color_bar=True, label_text=[f'Grad{x+1}' for x in range(n_components)], share=share, 
          nan_color=(0,0,0,0.5), cmap='seismic', background=(1,1,1), transparent_bg=False, 
          screenshot=True, filename=pjoin(result_path, f'gradient_map-{target}-{method}.jpg'))

# plot_hemispheres(surf_lh, surf_rh, array_name=gradients_embedding, size=(1200, 600), 
#                  color_bar=True, label_text=embeddings, zoom=1.5, 
#                  nan_color=(0,0,0,0.5), cmap='jet', background=(1,1,1),)
#                  #screenshot=True, filename=pjoin(result_path, f'gradient_map-{target}.jpg'))
