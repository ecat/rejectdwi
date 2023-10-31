import numpy as np
import sigpy as sp
import cv2

def largest_connected_component_by_slice(mask_zxy):
    mask_zxy = mask_zxy.astype(np.int8)
    nz, nx, ny = mask_zxy.shape
    mask_zxy_out = np.zeros_like(mask_zxy)

    for zz in range(nz):
        mask_xy = mask_zxy[zz, ...]
        connectivity = 4
        analysis = cv2.connectedComponents(mask_xy, connectivity=connectivity, ltype=cv2.CV_16U)
        num_labels, labels= analysis
        
        num_pixels_in_mask = np.zeros((num_labels,), dtype=np.int32)
        for label in range(num_labels):
            # find the largest label, need to overlay the mask because the '0' gets grouped into a connected component
            num_pixels_in_mask[label] = np.sum((label == labels) * mask_xy)

        best_label_index = np.argmax(num_pixels_in_mask)

        mask_xy = (labels == best_label_index)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_eroded_xy = cv2.morphologyEx(mask_xy.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask_zxy_out[zz, ...] = mask_eroded_xy

    return mask_zxy_out.astype(np.float32)


def upsample(x, full_matrix_size):
    ksp = sp.fft(x)
    ksp = uncenter_crop(ksp, full_matrix_size)
    return sp.ifft(ksp)

def uncenter_crop(x, full_matrix_size):
    assert(len(full_matrix_size) == x.ndim)    
    assert np.all(np.logical_or(np.array(full_matrix_size) % 2 == 0, np.array(full_matrix_size) == -1)), "Does not support odd matrix sizes, use -1 for full"
    inner_matrix_size = x.shape
    output_matrix_size = [x.shape[dim] if full_matrix_size[dim] == -1 else full_matrix_size[dim] for dim in range(x.ndim)]
    xp = sp.get_array_module(x)
    with sp.get_device(x):        
        y_full = xp.zeros(output_matrix_size, dtype=x.dtype)
        slicer = (slice(full_matrix_size[dim]//2 - inner_matrix_size[dim]//2, full_matrix_size[dim]//2 + inner_matrix_size[dim]//2) if full_matrix_size[dim] > 0 else slice(None) for dim in range(0, x.ndim))
        y_full[tuple(slicer)] = x
    return y_full

def get_center_mask(full_matrix_size, inner_matrix_size, dtype=np.complex64):
    # returns a mask with center is ones
    assert(len(full_matrix_size) == len(inner_matrix_size))
    return uncenter_crop(np.ones(inner_matrix_size, dtype=dtype), full_matrix_size)

def center_crop(x, crop_size):
    # center crops the array x to a dimension specified by crop size, if crop_size has a -1 it will take the whole axis    
    assert(x.ndim == len(crop_size))
    assert(all(tuple(crop_size[dim] <= x.shape[dim] for dim in range(0, x.ndim))))    
    slicer = tuple(slice((x.shape[dim] - crop_size[dim])//2 if crop_size[dim] > 0 else None, (x.shape[dim] + crop_size[dim])//2 if crop_size[dim] > 0 else None) for dim in range(0, x.ndim))
    return x[slicer]

def house_window(triangle_window_dims, full_matrix_size):
    # returns a ramp within inner_matrix_size that is 0.5 at point inner_matrix_size, zeros outside
    # triangle_window_dims is the size of the full triangle
    inner_matrix_size_fwhm = tuple( x // 2 for x in triangle_window_dims)

    return uncenter_crop(center_crop(triangle_window(triangle_window_dims, full_matrix_size), inner_matrix_size_fwhm), full_matrix_size)

def triangle_window(inner_matrix_size, full_matrix_size=None):    
    # returns a ramp triangle window within inner_matrix_size, zeros everywhere else    
    if full_matrix_size is None:
        full_matrix_size = inner_matrix_size
    assert(all(tuple((inner_matrix_size[dim] <= full_matrix_size[dim],) for dim in range(len(inner_matrix_size)))))
    coords = np.meshgrid(*tuple(((np.linspace(-1, 1, inner_matrix_size[sz]),) for sz in range(len(inner_matrix_size)))), indexing='ij')

    triangle_window = np.ones(inner_matrix_size, dtype=np.float32)
    for dim in range(len(coords)):    
        triangle_window = triangle_window * np.abs(1 - np.abs(coords[dim]))
    
    return uncenter_crop(triangle_window, full_matrix_size)

def gauss_window(inner_matrix_size, inner_matrix_sigma, full_matrix_size=None):
    # inner_matrix_sigma is standard deviation across the inner matrix shape 
    if full_matrix_size is None:
        full_matrix_size = inner_matrix_size    
    coords = np.meshgrid(*tuple(((np.linspace(-1, 1, inner_matrix_size[sz]),) for sz in range(len(inner_matrix_size)))), indexing='ij')

    gauss_window = np.ones(inner_matrix_size, dtype=np.float32)    

    for dim, sigma in enumerate(inner_matrix_sigma):
        gauss_window = gauss_window * np.exp(-np.square(coords[dim]) / (2 * (sigma ** 2)))

    return uncenter_crop(gauss_window, full_matrix_size)

def fermi_filter(matrix_size, r=None, w=10):
    """
    Returns a fermi filter operator to apply to k-space data of a specified size.
    
    Assumes the k-space data is centered.
    
    Parameters
    ----------
    matrix_size : tuple of ints
        Specifies the size of the k-space on which to apply the Fermi Filter
    r : int or tuple of ints (Default is matrix_size/2)
        Specifies the radius of the filter
    w : int or tuple of ints
        Specifies the width of the filter
    
    Returns
    -------
    H : numpy array
        An array that specifies the element-wise coefficients for the specified
        Fermi filter. Apply by doing y * H.
    """
    if r is None:
        r = tuple(int(i/2) for i in matrix_size)
    elif isinstance(r, int):
        r = tuple(r for i in matrix_size)
        
    h = []
    for n in matrix_size:
        hn = np.linspace(-int(n/2), int(n/2)-1, num=n)
        hn = (1 + np.exp((hn - r)/w)) ** -1
        h.append(hn)

def whiten(x, cov):
    """
    Applies a Cholesky whitening transform to the data.
    
    Parameters
    ----------
    x : numpy/cupy array
        Input array whose first dimension corresponds to the number of coils
    cov : numpy/cupy array
        Covariance matrix of the noise present in x
        
    Returns
    -------
    y : numpy/cupy array
        A whitened version of x with the same dimensions
    """
    
    # Check that the whitening transform can be applied
    if x.shape[-2] != cov.shape[0]:
        raise ValueError("The first dimension of x and the provided covariance matrix do not match.")
    
    device = sp.get_device(x)
    xp = device.xp
    cov = sp.to_device(cov, device)
        
    # Get the whitening transform operator
    L = xp.linalg.cholesky(xp.linalg.inv(cov))
    LH = L.conj().T
    
    # Apply the whitening transform and return the result
    y = LH @ x
    return y

def sos(matrix_in, axis_in, keepdims=False):
    device = sp.get_device(matrix_in)
    xp = device.xp
    with device:
        matrix_out = xp.sqrt(xp.sum(xp.abs(matrix_in) ** 2, axis=axis_in, keepdims=keepdims))
    return matrix_out

def match_device(a, b):
    # copies a to the same device as b
    return sp.to_device(a, sp.backend.get_device(b))