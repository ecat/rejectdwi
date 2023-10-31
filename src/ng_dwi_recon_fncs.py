import enum as enum
import numpy as np
import sigpy as sp
import cupy as cp
from recon_utils import sos, triangle_window, gauss_window, house_window, uncenter_crop
from shot_rejection import walsh_method, normalize_phasenav_weights, PhaseNavPerVoxelNormalizationType, PhaseNavNormalizationType

class PhaseNavType(enum.Enum):
    NONE = 0
    PHASE_ONLY = 1
    MAGNITUDE_AND_PHASE = 2

class ShotRejectionModel(enum.Enum):
    SINGLE_SHOT = 0 # reconstructs all shots independently and magnitude combines, no low resolution phase assumption
    SS_SENSE_LIKE = 1
    MULTI_SHOT = 2 # everything below here is multishot
    MS_WLSQ = 3 
    MS_WLSQ_DC = 4
    MS_SENSE_LIKE = 5

class PhaseNavWindowType(enum.Enum):
    BOX = 0
    TRIANGLE = 1
    GAUSSIAN = 2
    HOUSE = 3

def reconstruct_phasenavs(ksp_zsxyc, mask_zsxy_, sens_zxyc, window_size, window_type, device):
    xp = device.xp
    nz, ns, nx, ny, nc = ksp_zsxyc.shape
    im_phasenav_fullres_zsxy = xp.zeros((nz, ns, nx, ny), dtype=np.complex64)

    for z in range(nz):
        im_phasenav_fullres_zsxy[z, ...] = reconstruct_phasenav(ksp_zsxyc[z, ...], mask_zsxy_[z, ...], sens_zxyc[z, ...], device)


    if window_type == PhaseNavWindowType.BOX:
        window = xp.reshape(uncenter_crop(xp.ones(window_size), (nx, ny)), (1, 1, nx, ny))
    elif window_type == PhaseNavWindowType.TRIANGLE:
        window = sp.to_device(xp.reshape(triangle_window(window_size, (nx, ny)), (1, 1, nx, ny)), device)
    elif window_type == PhaseNavWindowType.GAUSSIAN:
        window = sp.to_device(xp.reshape(gauss_window(window_size, (1, 1), full_matrix_size=(nx, ny)), (1, 1, nx, ny)), device)    
    elif window_type == PhaseNavWindowType.HOUSE:
        window = sp.to_device(xp.reshape(house_window(window_size, (nx, ny)), (1, 1, nx, ny)), device)
    else:
        raise ValueError("Invalid window_type")
        
    F = sp.linop.FFT(im_phasenav_fullres_zsxy.shape, axes=(-2, -1))
    im_phasenav_lowres_zsxy = F.H * ((F * im_phasenav_fullres_zsxy) * window)

    return im_phasenav_fullres_zsxy, im_phasenav_lowres_zsxy


def reconstruct_phasenav(ksp_sxyc, mask_sxy, sens_xyc, device, niter=10):    

    xp = device.xp
    ns, nx, ny, nc = ksp_sxyc.shape

    nx_mask = mask_sxy.shape[1]
    mask_sxy_ = xp.reshape(mask_sxy, (ns, nx_mask, ny, 1))
    sens_xyc = xp.reshape(sens_xyc, (nx, ny, nc))

    im_phasenav_sxy = xp.zeros((ns, nx, ny), dtype=np.complex64)
    for shot in range(ns):
        Rs = sp.linop.Reshape((nx, ny, 1), (nx, ny))
        S = sp.linop.Multiply(Rs.oshape, sens_xyc)
        F = sp.linop.FFT(S.oshape, axes=(0, 1))
        D = sp.linop.Multiply(F.oshape, mask_sxy_[shot, ...])

        E = D * F * S * Rs

        EHb = E.H * ksp_sxyc[shot, ...]
        #im_phasenav_sxy[shot, ...] = im_data_xy
        alg = sp.alg.ConjugateGradient(E.H * E, EHb, im_phasenav_sxy[shot, ...], max_iter=niter)
        #alg = sp.alg.GradientMethod(lambda x: E.H * E * x - EHb, im_phasenav_sxy[shot, ...], alpha=1)
        while not alg.done():
            alg.update()

    return im_phasenav_sxy

def reconstruct_dwi_image(ksp_sxyc, sens_xyc, mask_sxy_, phasenav_sxy, phasenav_mag_sxy, shot_rejection_model, niter, device):

    ns, nx, ny, nc = ksp_sxyc.shape
    xp = sp.get_array_module(ksp_sxyc)

    if shot_rejection_model == ShotRejectionModel.SINGLE_SHOT:
        return xp.mean(xp.abs(reconstruct_phasenav(ksp_sxyc, mask_sxy_, sens_xyc, device, niter)), axis=(0,))
    elif shot_rejection_model == ShotRejectionModel.SS_SENSE_LIKE:
        im_pi_by_shot_sxy = reconstruct_phasenav(ksp_sxyc, mask_sxy_, sens_xyc, device, niter)
        
        weights_denominator_sxy = xp.power(sos(phasenav_mag_sxy, axis_in=(0,)), 2)
        weights_denominator_sxy[weights_denominator_sxy == 0] = 1
        weights_sxy = xp.squeeze(phasenav_mag_sxy / weights_denominator_sxy)

        return xp.sum(xp.abs(im_pi_by_shot_sxy) * weights_sxy, axis=(0,))

    Rs = sp.linop.Reshape((1, nx, ny, 1), (nx, ny))
    P = sp.linop.Multiply(Rs.oshape, phasenav_sxy)        
    S = sp.linop.Multiply(P.oshape, sens_xyc)
    F = sp.linop.FFT(S.oshape, axes=(1, 2))
    D = sp.linop.Multiply(F.oshape, mask_sxy_)
        
    if shot_rejection_model == ShotRejectionModel.MS_WLSQ or shot_rejection_model == ShotRejectionModel.MS_WLSQ_DC:

        if shot_rejection_model == ShotRejectionModel.MS_WLSQ_DC:
            phasenav_mag_sxy = xp.mean(phasenav_mag_sxy, axis=(1, 2), keepdims=True)
            phasenav_mag_sxy = phasenav_mag_sxy / xp.max(xp.ravel(phasenav_mag_sxy))
            # remask the WLSQ_DC so that we get some support region in image domain, not completely constant in image domain
            phasenav_mag_sxy = phasenav_mag_sxy * (sos(sens_xyc, axis_in=-1, keepdims=True) > 0)

        W_half = sp.linop.Multiply(D.oshape, xp.sqrt(phasenav_mag_sxy))
        W = sp.linop.Multiply(D.oshape, phasenav_mag_sxy)
        F_dummy = sp.linop.FFT(D.oshape, axes=(1, 2))
        E_tmp = D * F * S * P * Rs
        E = F_dummy * W_half * F_dummy.H * E_tmp                
        EHE = E_tmp.H * F_dummy * W * F_dummy.H * E_tmp
    elif shot_rejection_model == ShotRejectionModel.MS_SENSE_LIKE:
        WP = sp.linop.Multiply(Rs.oshape, phasenav_mag_sxy * phasenav_sxy)
        E = D * F * S * WP * Rs
        EHE = E.H * E
    else:
        E = D * F * S * P * Rs
        EHE = E.H * E            

    EHb = E.H * ksp_sxyc

    im_xy_output = xp.zeros((nx, ny), dtype=np.complex64)    
    alg = sp.alg.ConjugateGradient(EHE, EHb, im_xy_output, max_iter=niter)
    while not alg.done():
        alg.update()

    return im_xy_output


def multishot_dwi_recon(ksp_zsxyc, mask_zsxy_, sens_zxyc, phasenav_mag_zsxy, phasenav_angle_zsxy, shot_rejection_model, device=sp.cpu_device, niter=20):

    if shot_rejection_model in [ShotRejectionModel.MS_WLSQ, ShotRejectionModel.MS_WLSQ_DC, ShotRejectionModel.MS_SENSE_LIKE]:
        assert phasenav_mag_zsxy is not None

    xp = device.xp

    nz, ns, nx, ny, nc = ksp_zsxyc.shape

    sens_zxyc = sp.to_device(sens_zxyc, device) # doesn't take too much memory so can put on gpu all at once

    ksp_zsxyc = xp.reshape(sp.to_device(ksp_zsxyc, device), (nz, ns, nx, ny, nc))
    mask_zsxy_ = xp.reshape(mask_zsxy_[..., 0], mask_zsxy_.shape[0:4] + (1,))
    #mask_zsxy_ = xp.abs(ksp_zsxyc[:, :, :, :, 0, np.newaxis]) > 0

    im_zxy = xp.zeros((nz, nx, ny), dtype=np.complex64)

    #print(shot_rejection_type)
    for z in range(nz):
        #print('slice ', z)

        ksp_sxyc = sp.to_device(ksp_zsxyc[z, ...], device)    
        sens_xyc = sens_zxyc[z, np.newaxis, ...]
        mask_sxy_ = mask_zsxy_[z, ...]
        phasenav_sxy = xp.reshape(phasenav_angle_zsxy[z, ...], (ns, nx, ny, 1))
        phasenav_mag_sxy = xp.reshape(phasenav_mag_zsxy[z, ...], (ns, nx, ny, 1))
        
        im_zxy[z, ...] = reconstruct_dwi_image(ksp_sxyc, sens_xyc, mask_sxy_, phasenav_sxy, phasenav_mag_sxy, shot_rejection_model, niter, device)

    return im_zxy

    
def calculate_rejection_weights(im_phasenav_lowres_zsxy, reject_window_shape, reject_window_stride, 
                                phasenav_weighting_normalization, phasenav_weighting_per_voxel_normalization, percentile_shots_to_keep=None,
                                do_walsh_method_on_magnitude_images=True):
    
    xp = sp.get_array_module(im_phasenav_lowres_zsxy)
    nz, ns, nx, ny = im_phasenav_lowres_zsxy.shape

    im_phasenav_weights_normalized_zsxy = xp.zeros((nz, ns, nx, ny), dtype=np.complex64)
    im_phasenav_weights_zsxy = xp.zeros((nz, ns, nx, ny), dtype=np.complex64)

    for zz in range(nz):
        c_to_last = sp.linop.Transpose((ns, nx, ny), (1, 2, 0))
        r1s = sp.linop.Reshape((nx, ny, 1, ns), c_to_last.oshape)
        R1 = r1s * c_to_last

        if do_walsh_method_on_magnitude_images:
            im_phasenav_to_weight_sxy = xp.abs(im_phasenav_lowres_zsxy[zz, ...])
        else:
            im_phasenav_to_weight_sxy = im_phasenav_lowres_zsxy[zz, ...]

        im_phasenav_weight_sxy = R1.H * walsh_method(R1 * im_phasenav_to_weight_sxy, reject_window_shape, reject_window_stride, alg_str='eig')

        r2s = sp.linop.Reshape((1, nx, ny, 1, ns), c_to_last.oshape)
        R2 = r2s * c_to_last
        
        im_phasenav_weight_normalized_sxy = R2.H * normalize_phasenav_weights(R2 * im_phasenav_weight_sxy, 
                                                                    phasenav_weighting_normalization, 
                                                                    phasenav_weighting_per_voxel_normalization,
                                                                    None,  
                                                                    percentile_shots_to_keep, silent=zz > 0)

        im_phasenav_weights_zsxy[zz, ...] = im_phasenav_weight_sxy 
        im_phasenav_weights_normalized_zsxy[zz, ...] = im_phasenav_weight_normalized_sxy

    return im_phasenav_weights_zsxy, im_phasenav_weights_normalized_zsxy
        
