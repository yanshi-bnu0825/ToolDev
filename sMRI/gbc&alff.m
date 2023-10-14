% The workflow of resample alff & falff
% Created on Mon Jul 5 15:48:49 2021
% @author: Sai Ma

%%
% reset workspeace and variables
clc;
clear;
work_dir=fullfile('/nfs/e1/HCPD_CB/mri/');
mni2suit_dir=fullfile(work_dir,'MNI_to_SUIT')

%%
% read all subject id from subject_list.csv
subid_file = fopen(fullfile(work_dir,'subject_list.csv'));
subject_list=textscan(subid_file,'%s','Delimiter',',');
fclose(subid_file);
subject_id=subject_list{1,1};

%%
% start SPM fmri for SUIT
spm fmri;

%%
for id=1:length(subject_id)
    subject_id{id}
    rfmri_dir=fullfile(work_dir,subject_id{id},'rfMRI');
    % unzip nifti files
    gunzip(fullfile(rfmri_dir,'cerebellum_alff_mni.nii.gz'));
    gunzip(fullfile(rfmri_dir,'cerebellum_cortex_gbc_mni.nii.gz'));
    % resample ALFF data to SUIT speace
    job_alff.subj.affineTr={fullfile(mni2suit_dir,'cerebellum_normalize_affine_linear.mat')};
    job_alff.subj.flowfield={fullfile(mni2suit_dir,'cerebellum_normalize_flowfield_nonlinear.nii')};
    job_alff.subj.resample={fullfile(rfmri_dir,'cerebellum_alff_mni.nii')};
    job_alff.subj.mask={fullfile(mni2suit_dir,'cerebellum_mask_mni.nii')};
    suit_reslice_dartel(job_alff)
    % resample GBC data to SUIT speace
    job_gbc.subj.affineTr={fullfile(mni2suit_dir,'cerebellum_normalize_affine_linear.mat')};
    job_gbc.subj.flowfield={fullfile(mni2suit_dir,'cerebellum_normalize_flowfield_nonlinear.nii')};
    job_gbc.subj.resample={fullfile(rfmri_dir,'cerebellum_cortex_gbc_mni.nii')};
    job_gbc.subj.mask={fullfile(mni2suit_dir,'cerebellum_mask_mni.nii')};
    suit_reslice_dartel(job_gbc)
    % delete native myelin image
    delete(fullfile(rfmri_dir,'cerebellum_alff_mni.nii'));
    delete(fullfile(rfmri_dir,'cerebellum_cortex_gbc_mni.nii'));
    % rename 
    movefile(fullfile(rfmri_dir,'wdcerebellum_alff_mni.nii'),fullfile(rfmri_dir,'cerebellum_alff_suit.nii'))
    movefile(fullfile(rfmri_dir,'wdcerebellum_cortex_gbc_mni.nii'),fullfile(rfmri_dir,'cerebellum_cortex_gbc_suit.nii'))
end
