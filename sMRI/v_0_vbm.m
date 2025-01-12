% The workflow of caculating VBM
% Created on Mon Jul 5 15:48:49 2021
% @author: Sai Ma

%%
% reset workspeace and variables
clc;
clear;
hcpd_path=fullfile('/nfs/m1/hcp/');
work_dir=fullfile('/nfs/z1/userhome/MaSai/workingdir/code/cb_ya/HCPYA_CB/');

%%
% read all subject id from subject_list.csv
subid_file = fopen(fullfile('/nfs/z1/userhome/MaSai/workingdir/code/cb_ya/HCPYA_subject_list.csv'));
subject_list = textscan(subid_file,'%s','Delimiter',',');
fclose(subid_file);
subject_id = subject_list{1,1};

%%
% start SPM fmri for SUIT
spm fmri;

%%
% caculate VBM using SUIT
% for id=1:length(subject_id)
for id=1:2
    % make subject folders
    mkdir(fullfile(work_dir,subject_id{id}));
    mkdir(fullfile(work_dir,subject_id{id},'anat'));
    mkdir(fullfile(work_dir,subject_id{id},'func'));
    % copy native images
    copyfile(fullfile(hcpd_path,subject_id{id},'T1w','T1w_acpc_dc_restore.nii.gz'),fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    copyfile(fullfile(hcpd_path,subject_id{id},'T1w','T2w_acpc_dc_restore.nii.gz'),fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
    % unzip native images
    gunzip(fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    gunzip(fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
    % isolate cerebellum from native brain
    suit_isolate_seg({fullfile(work_dir,subject_id{id},'anat','T1w.nii'),fullfile(work_dir,subject_id{id},'anat','T2w.nii')})
    % normalize native cerebellum
    job_nor.subjND.gray={fullfile(work_dir,subject_id{id},'anat','T1w_seg1.nii')};
    job_nor.subjND.white={fullfile(work_dir,subject_id{id},'anat','T1w_seg2.nii')};
    job_nor.subjND.isolation={fullfile(work_dir,subject_id{id},'anat','c_T1w_pcereb.nii')};
    suit_normalize_dartel(job_nor)
    % caculate VBM
    job_vbm.subj.affineTr={fullfile(work_dir,subject_id{id},'anat','Affine_T1w_seg1.mat')};
    job_vbm.subj.flowfield={fullfile(work_dir,subject_id{id},'anat','u_a_T1w_seg1.nii')};
    job_vbm.subj.resample={fullfile(work_dir,subject_id{id},'anat','T1w_seg1.nii')};
    job_vbm.subj.mask={fullfile(work_dir,subject_id{id},'anat','c_T1w_pcereb.nii')};
    job_vbm.jactransf=1;
    suit_reslice_dartel(job_vbm)
    % delete native images
    delete(fullfile(work_dir,subject_id{id},'anat','T1w.nii'))
    delete(fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    delete(fullfile(work_dir,subject_id{id},'anat','T2w.nii'))
    delete(fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
end

job_nor.subjND.gray={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/cerebellum_graymatter_prob_native.nii'};
job_nor.subjND.white={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/cerebellum_whitematter_prob_native.nii'};
job_nor.subjND.isolation={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/new_cerebellum_mask.nii'};
job_nor.subjND.vox={2 2 2};
suit_normalize_dartel(job_nor)

job_gm.subj.affineTr={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/Affine_cerebellum_graymatter_prob_native.mat'};
job_gm.subj.flowfield={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/u_a_cerebellum_graymatter_prob_native.nii'};
job_gm.subj.resample={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/cerebellum_graymatter_prob_native.nii'};
job_gm.subj.mask={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/new_cerebellum_mask.nii'};
job_gm.jactransf=1;
suit_reslice_dartel(job_gm)

job_my.subj.affineTr={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/Affine_cerebellum_graymatter_prob_native.mat'};
job_my.subj.flowfield={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/u_a_cerebellum_graymatter_prob_native.nii'};
job_my.subj.resample={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/T1wDividedByT2w.nii'};
job_my.subj.mask={'/nfs/e2/workingshop/masai/code/cb/test_res/HCD0001305_test/new_cerebellum_mask.nii'};
suit_reslice_dartel(job_my)

job_my.subj.affineTr={'/nfs/e1/HCPD_CB/mri/HCD0001305_V1_MR/sMRI/cerebellum_normalize_affine_linear.mat'};
job_my.subj.flowfield={'/nfs/e1/HCPD_CB/mri/HCD0001305_V1_MR/sMRI/cerebellum_normalize_flowfield_nonlinear.nii'};
job_my.subj.resample={'/nfs/e1/HCPD_CB/mri/HCD0001305_V1_MR/sMRI/T1w_acpc_dc_restore.nii'};
job_my.subj.mask={'/nfs/e1/HCPD_CB/mri/HCD0001305_V1_MR/sMRI/cerebellum_mask_native.nii'};
suit_reslice_dartel(job_my)

