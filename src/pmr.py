import sigpy as sp

def pseudo_multiple_replica(recon_function, ksp_measured, n_replicas=10, im_mask=1, cov=None):
    # recon_function reconstructs some kspace, make sure that recon_function accepts a mask so that
    # the mask can be adjusted for the accelerated and unaccelerated acquisitions

    xp = sp.get_array_module(ksp_measured)

    if cov is not None:
        raise NotImplementedError('SNR Maps not properly implemented yet')

    ksp_mask = xp.abs(ksp_measured) > 0
    get_complex_noise = lambda: (xp.random.normal(size=ksp_measured.shape) + 1j * xp.random.normal(size=ksp_measured.shape)) / xp.sqrt(2)

    input_scaling = [0.0, 1] # does the recon with noise only and normal measurements + noise 

    noisy_recon_std_maps = [None] * 2

    for idx, scaling in enumerate(input_scaling):
        
        noise_ksp_mask = xp.ones_like(ksp_measured) if scaling == 0 else ksp_mask

        noisy_input_replicas = [noise_ksp_mask * get_complex_noise() + scaling * ksp_measured for _ in range(n_replicas)]
        noisy_recon_replicas = xp.stack([recon_function(noisy_input, noise_ksp_mask) for noisy_input in noisy_input_replicas], axis=0)
        noisy_recon_std_map = xp.std(noisy_recon_replicas, axis=0)    

        noisy_recon_std_maps[idx] = noisy_recon_std_map

    acceleration_factor = xp.size(ksp_mask) / xp.sum(ksp_mask)
    noiseless_recon = recon_function(ksp_measured, ksp_mask)    

    gfactor = im_mask * xp.nan_to_num(noisy_recon_std_maps[1] / noisy_recon_std_maps[0]) / xp.sqrt(acceleration_factor)
    snr_map = im_mask * xp.nan_to_num(noiseless_recon / noisy_recon_std_maps[1])

    return gfactor, snr_map, noiseless_recon