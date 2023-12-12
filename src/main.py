# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

import os, sys

is_jupyter = hasattr(sys, 'ps1') # https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode

from utils import montage, calculate_adc_map
from recon_utils import sos, largest_connected_component_by_slice
from ng_dwi_recon_fncs import reconstruct_phasenavs, multishot_dwi_recon, ShotRejectionModel, PhaseNavWindowType, calculate_rejection_weights
from shot_rejection import PhaseNavNormalizationType, PhaseNavPerVoxelNormalizationType
from pmr import pseudo_multiple_replica
from cfl import writecfl, readcfl
from cfl_loader import load_epi_cfls
from coil_compression import get_cc_matrix, apply_cc_matrix

# +
import numpy as np
import cupy as cp
import sigpy as sp
import sigpy.plot as pl
import sigpy.mri as mr
import time
import matplotlib.pyplot as plt
import warnings
import argparse

# %matplotlib widget

# +
if is_jupyter:
    volunteer_index = 5
    device_id = 0 # can use -1 if no gpu
    
    args = None
else:
    parser = argparse.ArgumentParser(description='Shot Rejection Reconstruction') 
    parser.add_argument('--volunteer-index', required=True, type=int)
    parser.add_argument('--phasenav-res-fwhm', required=True, type=int)
    parser.add_argument('--phasenav-reject-window', required=False, type=int, default=4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--reject-normalization', help='5: Max 6: Percentile', required=False, type=int, default=6)
    parser.add_argument('--accept-window', type=float, default=None)
    
    args = parser.parse_args()

    if args.reject_normalization == 5 and args.accept_window is not None:
        warnings.warn("Accept window argument is ignored")

    device_id = args.device
    volunteer_index = args.volunteer_index

root_dir = '/home/sdkuser/workspace/data/volunteer' + str(volunteer_index) + '/'
if volunteer_index in [1, 2, 3]:
    sub_dirs =  ['scan1/', 'scan2/']
elif volunteer_index in [4, 5]:
    sub_dirs = ['scan3/']
else:
    raise ValueError('Volunteer index must be between 1 and 5')

recon_device = sp.Device(device_id)
recon_device.use()
to_recon_device = lambda x: sp.to_device(x, recon_device)
xp = recon_device.xp

# set the FFT cache to have maximum 2 plans since it uses a lot of memory
fft_plan_cache = cp.fft.config.get_plan_cache()
fft_plan_cache.set_size(2)

scan_dirs = [root_dir + scan_dir for scan_dir in sub_dirs]
print('Scan directories ' + str(scan_dirs))

ksp_vzsxyc, mask_vzs_y, ksp_cal_vzxycn, sens_asset_zxyc, psi_v = load_epi_cfls(scan_dirs, sp.cpu_device) # load data on CPU then copy to GPU after coil compress   
sens_asset_mask_zxyc = sos(sens_asset_zxyc, 3, keepdims=True) > 0.1

nv, nz, ns, nx, ny, nc = ksp_vzsxyc.shape
nn = 1
ksp_cal_vzxycn = xp.reshape(ksp_cal_vzxycn, (nv, nz, nx, ny, nc, nn))

# +
do_cc = True

if do_cc:
    ncc = nc // 2 # can compress more if low on memory
    ksp_cc_vzsxyc = np.zeros((nv, nz, ns, nx, ny, ncc), dtype=np.complex64)
    ksp_cc_cal_vzxycn = np.zeros((nv, nz, nx, ny, ncc, nn), dtype=np.complex64)
    sens_cc_asset_zxyc = np.zeros((nz, nx, ny, ncc), dtype=np.complex64)

    for zz in range(0, nz):
        cc_mtx = get_cc_matrix(sens_asset_zxyc[zz, ...] , ncc)
        ksp_cc_vzsxyc[:, zz, ...] = apply_cc_matrix(ksp_vzsxyc[:, zz, ...], cc_mtx)
        ksp_cc_cal_vzxycn[:, zz, ..., 0] = apply_cc_matrix(ksp_cal_vzxycn[:, zz, ..., 0], cc_mtx)
        sens_cc_asset_zxyc[zz, ...] = apply_cc_matrix(sens_asset_zxyc[zz, ...], cc_mtx)

    nc = ncc
    ksp_vzsxyc = to_recon_device(ksp_cc_vzsxyc)
    ksp_cal_vzxycn = to_recon_device(ksp_cc_cal_vzxycn)
    sens_asset_zxyc = to_recon_device(sens_cc_asset_zxyc)
else:
    ksp_vzsxyc = to_recon_device(ksp_vzsxyc)
    ksp_cal_vzxycn = to_recon_device(ksp_cal_vzxycn)
    sens_asset_zxyc = to_recon_device(sens_asset_zxyc)

mask_vzs_y = to_recon_device(mask_vzs_y)

# +
im_naive_vzxyc = xp.sum(sp.ifft(ksp_vzsxyc, axes=(-3, -2)), axis=2)

noise_patch_x = slice(0, 20)
noise_patch_y = slice(0, 20)
vol_to_cov = 0
noise_patch = im_naive_vzxyc[vol_to_cov, 0, noise_patch_x, noise_patch_y, :]
noise_covariance = xp.cov(xp.reshape(noise_patch, (-1, nc)).T)

if True:
    plt.figure()
    plt.imshow(np.abs(noise_covariance.get()), cmap='gray')
    plt.title('Empirical Noise Covariance in Image Domain Should be Close to Identity')

if True:
    _ = montage(xp.abs(im_naive_vzxyc[vol_to_cov, 0, ...] / xp.max(xp.abs(im_naive_vzxyc[vol_to_cov, 0, ...].ravel()))), grid_cols=nc//2)
    plt.clim([0, .5])

im_naive_cal_vzxycn = sp.ifft(ksp_cal_vzxycn, axes=(2, 3))
if is_jupyter:
    im_to_show_vzxyc = im_naive_cal_vzxycn[..., 0]
    vol_to_show = 0
    sl_to_show = 7

    im_to_show_xyc = sp.to_device(im_to_show_vzxyc[vol_to_show, sl_to_show, :, :, :] / xp.max(xp.abs(xp.ravel(im_to_show_vzxyc))))
    
    _ = montage(np.abs(im_to_show_xyc), grid_cols=nc//4)
    plt.clim([0, .12])
    _ = montage(np.angle(im_to_show_xyc), grid_cols=nc//4)

# +
ksp_for_sens_estimation_zxyc = ksp_cal_vzxycn[0, ..., 0]

do_espirit = True

im_conjphase_asset_zxy = xp.abs(xp.sum(im_naive_cal_vzxycn[0, ..., 0] * xp.conj(sens_asset_zxyc), axis=-1))
im_conjphase_asset_zxy = im_conjphase_asset_zxy / xp.max(xp.abs(im_conjphase_asset_zxy), axis=(1, 2), keepdims=True)


if do_espirit:
    sens_zxyc = xp.zeros((nz, nx, ny, nc), dtype=np.complex64)

    for zz in range(0, nz):
        ksp_for_sens_estimation_xyc = ksp_for_sens_estimation_zxyc[zz, ...]
        
        c_to_first = sp.linop.Transpose(ksp_for_sens_estimation_xyc.shape, (2, 0, 1))
        ksp_for_sens_estimation_cxy = c_to_first * ksp_for_sens_estimation_xyc
        # use higher thresh on espirit to reduce motion ghosts from appearing in coil sens
        sens_cxy = mr.app.EspiritCalib(ksp_for_sens_estimation_cxy, calib_width=24, device=recon_device, kernel_width=6, thresh=0.04, crop=0.95, show_pbar=False).run()
        
        sens_zxyc[zz, :, :, :] = c_to_first.H * sens_cxy
else:

    sens_zxyc = sens_asset_zxyc 
    
im_support_zxy = sp.to_device(largest_connected_component_by_slice((sos(sens_zxyc, -1) > 0.01).get()), recon_device)
sens_zxyc = sens_zxyc * im_support_zxy[..., np.newaxis]
# -

if is_jupyter:
    #_ = montage(xp.transpose(im_support_zxy, (1, 2, 0)))
    _ = montage(xp.abs(sens_zxyc[sl_to_show, ...]), grid_cols=nc//4)
    plt.title('Coil Sensitivities')
    _ = montage(xp.transpose(im_conjphase_asset_zxy, (1, 2, 0)))
    plt.title('Conjugate Phase Coil Combination Using ASSET Coils')
    plt.clim([0, .5])

# +
im_phasenav_fullres_vzsxy = xp.zeros((nv, nz, ns, nx, ny), dtype=np.complex64)
im_phasenav_lowres_vzsxy = xp.zeros((nv, nz, ns, nx, ny), dtype=np.complex64)

phasenav_res_fwhm = 96 if args is None else args.phasenav_res_fwhm
phasenav_window_size = (phasenav_res_fwhm * 2, phasenav_res_fwhm * 2)
phasenav_window_type = PhaseNavWindowType.HOUSE if phasenav_res_fwhm > ny // 4 else PhaseNavWindowType.TRIANGLE # logic to maintain symmetry if partial fourier sampling

reconstruct_phasenavs_helper = lambda _ksp_zsxyc, _mask_zs_y: reconstruct_phasenavs(_ksp_zsxyc, _mask_zs_y, 
                                                                                    sens_zxyc, phasenav_window_size, phasenav_window_type, recon_device)

print('Phase Nav Res FWHM ' + str(phasenav_res_fwhm) + ' Phase Nav Window Type ' + str(phasenav_window_type))

for vv in range(nv):
    ksp_zsxyc = ksp_vzsxyc[vv, ...]
    mask_zs_y = mask_vzs_y[vv, ...]

    im_phasenav_fullres_zsxy, im_phasenav_lowres_zsxy = reconstruct_phasenavs_helper(ksp_zsxyc, mask_zs_y)

    im_phasenav_fullres_vzsxy[vv, ...] = im_phasenav_fullres_zsxy
    im_phasenav_lowres_vzsxy[vv, ...] = im_phasenav_lowres_zsxy

# +
#reject_window_width = ny // phasenav_res_fwhm * 2
reject_window_width = 4 if args is None else args.phasenav_reject_window
reject_window_shape = (reject_window_width, reject_window_width, 1)
reject_window_stride = (1, 1, 1)
phasenav_weighting_normalization = PhaseNavNormalizationType.NOOP
phasenav_weighting_per_voxel_normalization = PhaseNavPerVoxelNormalizationType.PERCENTILE_PER_RESPIRATORY_PHASE if args is None else PhaseNavPerVoxelNormalizationType(args.reject_normalization)
do_walsh_method_on_magnitude_images = True
percentile_shots_to_keep = xp.array([.15]) if args is None else xp.array([args.accept_window / 100])

print('Reject Window ' + str(reject_window_shape) + ' Walsh method on Magnitude Images ' + str(do_walsh_method_on_magnitude_images) + ' Percentile Shots to Keep ' + str(percentile_shots_to_keep))

calculate_rejection_weights_helper = lambda _im_phasenav_lowres_zsxy: calculate_rejection_weights(_im_phasenav_lowres_zsxy, 
                                            reject_window_shape, reject_window_stride, 
                                            phasenav_weighting_normalization, phasenav_weighting_per_voxel_normalization, 
                                            percentile_shots_to_keep, do_walsh_method_on_magnitude_images)

im_phasenav_weights_vzsxy = xp.zeros((nv, nz, ns, nx, ny), dtype=np.complex64)
im_phasenav_weights_normalized_vzsxy  = xp.zeros((nv, nz, ns, nx, ny), dtype=np.complex64)

for vv in range(nv):
    im_phasenav_lowres_zsxy = im_phasenav_lowres_vzsxy[vv, ...]

    im_phasenav_weights_zsxy, im_phasenav_weights_normalized_zsxy = calculate_rejection_weights_helper(im_phasenav_lowres_zsxy)
    
    im_phasenav_weights_vzsxy[vv, ...] = im_phasenav_weights_zsxy * im_support_zxy[:, np.newaxis, ...]
    im_phasenav_weights_normalized_vzsxy[vv, ...] = im_phasenav_weights_normalized_zsxy * im_support_zxy[:, np.newaxis, ...]       
    

# +
sl_to_show = 11
vol_to_show = -1

zxy_to_yxz = sp.linop.Transpose((ns, nx, ny), (2, 1, 0))
flip = sp.linop.Flip(zxy_to_yxz.oshape, axes=(1,))
phasenavs_to_show = flip * zxy_to_yxz * (im_phasenav_lowres_vzsxy[vol_to_show, sl_to_show, ...] * im_support_zxy[sl_to_show, ...])
phasenavs_to_show = phasenavs_to_show / xp.max(xp.ravel(xp.abs(phasenavs_to_show)))

if is_jupyter:
    n_grid_cols = max(ns // 4, 1)
    _ = montage(xp.abs(phasenavs_to_show), grid_cols=n_grid_cols)
    plt.clim([0, .5])
    plt.title('Phase Navigator Magnitudes Volume ' + str(vol_to_show))
    _ = montage(xp.abs(flip * zxy_to_yxz * im_phasenav_weights_vzsxy[vol_to_show, sl_to_show, ...]), grid_cols=n_grid_cols)
    plt.title('Phase Navigator Weights Volume ' + str(vol_to_show))
    _ = montage(xp.abs(flip * zxy_to_yxz * im_phasenav_weights_normalized_vzsxy[vol_to_show, sl_to_show, ...]), grid_cols=n_grid_cols)
    plt.title('Phase Navigator Weights Normalized Volume ' + str(vol_to_show))
    _ = montage(xp.angle(phasenavs_to_show), grid_cols=n_grid_cols)
    plt.title('Phase Navigator Phases Volume ' + str(vol_to_show))



# +
shot_rejection_models = [ShotRejectionModel.SINGLE_SHOT, ShotRejectionModel.MULTI_SHOT, 
                        ShotRejectionModel.MS_WLSQ_DC, ShotRejectionModel.MS_WLSQ, ShotRejectionModel.MS_SENSE_LIKE]

nw = len(shot_rejection_models) # num weighting types

im_recon_wvzxy = xp.zeros((nw, nv, nz, nx, ny), dtype=np.complex64)
gfactors_wvzxy = xp.zeros((nw, nv, nz, nx, ny), dtype=np.complex64)
snr_maps_wvzxy = xp.zeros((nw, nv, nz, nx, ny), dtype=np.complex64)

if len(scan_dirs) == 2: # has m0 and m1 concatenated along volume dimension
    vol_b0 = [0, 2]
    assert (nv == 4)
else:
    vol_b0 = [0]


for ww, shot_rejection_model in enumerate(shot_rejection_models):
    for vv in range(nv):
        ksp_zsxyc = ksp_vzsxyc[vv, ...]

        if vv in vol_b0:
            im_phasenav_weights_normalized_zsxy = xp.ones_like(im_phasenav_weights_normalized_vzsxy[vv, ...]) # disable phase nav weighting for b=0
        else:
            im_phasenav_weights_normalized_zsxy = im_phasenav_weights_normalized_vzsxy[vv, ...]

        im_phasenav_lowres_zsxy = im_phasenav_lowres_vzsxy[vv, ...]
        mask_zs_y = mask_vzs_y[vv, ...]

        multishot_dwi_recon_helper = lambda _ksp_zsxyc, _mask_zsxy_: multishot_dwi_recon(_ksp_zsxyc, _mask_zsxy_, sens_zxyc,
                                                                            im_phasenav_weights_normalized_zsxy, 
                                                                            xp.exp(1j * xp.angle(im_phasenav_lowres_zsxy)), 
                                                                            shot_rejection_model, 
                                                                            recon_device, niter=10)
        
        gfactor_map, snr_map, im_recon_zxy = pseudo_multiple_replica(multishot_dwi_recon_helper, ksp_zsxyc, im_mask=im_support_zxy, n_replicas=1)
        
        gfactors_wvzxy[ww, vv, ...] = gfactor_map
        snr_maps_wvzxy[ww, vv, ...] = snr_map
        im_recon_wvzxy[ww, vv, ...] = im_recon_zxy

# +
sl_to_show = nz - 2

def tile_vol_dim(im_wvzxy, normalize=False):
    xp = sp.get_array_module(im_wvzxy)
    nv_tile = im_wvzxy.shape[1]
    im_to_show_catted = [0] * nv_tile

    for vv in range(nv_tile):
        im_to_show_wxy = im_wvzxy[:, vv, sl_to_show, ...]
        im_to_show_xyw = xp.transpose(im_to_show_wxy, (1, 2, 0))
        im_to_show_xyw = xp.flip(im_to_show_xyw.swapaxes(0, 1), axis=1)

        if normalize:
            im_to_show_xyw = im_to_show_xyw / xp.max(xp.ravel(im_to_show_xyw))
            #im_to_show_xyv = im_to_show_xyv / xp.max(xp.abs(im_to_show_xyv), axis=(0, 1))
            #im_to_show_xyv = im_to_show_xyv / sos(im_to_show_xyv, (0, 1))

        im_to_show_catted[vv] = im_to_show_xyw

    if nv_tile == 1:
        return im_to_show_xyw
    else:
        return xp.concatenate(im_to_show_catted, axis=0)

im_recon_catted = tile_vol_dim(im_recon_wvzxy, normalize=True)

if is_jupyter:
    _ = montage(xp.abs(im_recon_catted), grid_cols=nw)
    plt.clim([0.0, .5])

    title_str = ''
    for shot_rejection_model in shot_rejection_models:
        title_str += str(shot_rejection_model)[len(ShotRejectionModel.__name__) + 1:] + '    '
    plt.title(title_str)
    plt.ylabel('Different Volumes')

    # snr_map_catted = tile_vol_dim(snr_maps_wvzxy)
    # _ = montage(xp.abs(snr_map_catted), grid_cols=nw)
    # plt.clim([0, 10])
    # plt.colorbar()
    # plt.title('SNR map')

    #gfactor_catted = tile_vol_dim(gfactors_wvzxy)
    #_ = montage(xp.abs(gfactor_catted), grid_cols=nw)
    #plt.colorbar()
    #plt.title('g factor')
    

# +
vol_pairs_to_adc = [(nv - 2, nv - 1)]
w_indices_to_adc = [1, -1]

im_adc_vwzxy = np.zeros((len(vol_pairs_to_adc), len(w_indices_to_adc), nz, nx, ny))

bvalue = 500 if volunteer_index in [1, 2, 3] else 750
for tar_v, vol_pair in enumerate(vol_pairs_to_adc):
    for tar_w, w in enumerate(w_indices_to_adc):
        im_adc_zxy = calculate_adc_map(im_recon_wvzxy[w, vol_pair[0], ...].get(), im_recon_wvzxy[w, vol_pair[1], ...].get(), bvalue)

        im_adc_vwzxy[tar_v, tar_w, ...] = im_adc_zxy
    

# +
im_adc_catted = tile_vol_dim(np.transpose(im_adc_vwzxy, (0, 1, 2, 3, 4)), normalize=False)

if is_jupyter:
    _ = montage(im_adc_catted * 1000, grid_cols=1)
    plt.clim([0, 3])
    plt.colorbar()
# -

 
