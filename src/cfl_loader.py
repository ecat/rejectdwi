from cfl import readcfl
import numpy as np
import sigpy as sp
from recon_utils import sos

def load_epi_cfls(root_dirs, device=sp.cpu_device):
    xp = device.xp
    ksp_v_ = []
    mask_v_ = []    
    psi_v_ = []

    for root_dir in root_dirs:
        get_in_path = lambda x: root_dir + '/' + x
        readcfl_to_device = lambda x: sp.to_device(readcfl(x), device)

        ksp_vzsxyc = readcfl_to_device(get_in_path('ksp_vzsxyc'))
        mask_vzs_y = readcfl_to_device(get_in_path('mask_vzs_y'))
        sens_asset_zxyc = readcfl_to_device(get_in_path('sens_asset_zxyc'))
        sens_asset_zxyc = sens_asset_zxyc / xp.max(sos(sens_asset_zxyc, 3))
        psi = readcfl_to_device(get_in_path('psi'))

        nsubvols = ksp_vzsxyc.shape[0]
        for vol_out in range(0, nsubvols):
            ksp_v_.append(ksp_vzsxyc[vol_out, ...])
            mask_v_.append(mask_vzs_y[vol_out, ...])
            psi_v_.append(psi)

    stacked_ksp_v_ = xp.stack(ksp_v_)
    ksp_cal_vzxycn = xp.sum(stacked_ksp_v_, axis=2)
    output = (stacked_ksp_v_, xp.stack(mask_v_), ksp_cal_vzxycn, sens_asset_zxyc, xp.stack(psi_v_))

    return output