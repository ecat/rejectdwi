import os
import sys

import numpy as np
import sigpy as sp
import cupy as cp
import warnings
import matplotlib.pyplot as plt

def filter_files(dirpath, filter):
    all_files = [os.path.join(basedir, filename) for basedir, dirs, files in os.walk(dirpath) for filename in files if (filter in filename) ]
    return all_files

def montage(im_xyz, grid_cols=4, normalize=False, title=None, do_show=False):
    assert(im_xyz.ndim == 3), "input must have three dimensions"
    im_xyz = sp.to_device(im_xyz)
    nx, ny, nz = im_xyz.shape  
    if (nz % grid_cols != 0):
        warnings.warn("Number of requested grid columns does not evenly divide nz, some slices will be absent")

    slicer = tuple((slice(None), slice(None), slice(row * grid_cols, min((row+1) * grid_cols, nz))) for row in range(0, nz//grid_cols))    
    im_to_show = np.vstack(tuple(np.reshape(np.transpose(im_xyz[s], (0, 2, 1)), (nx, -1)) for s in slicer))
    scale = np.max(np.abs(np.ravel(im_to_show))) if normalize else 1.
    
    plt.figure()
    plt.imshow(im_to_show / scale, cmap='gray', aspect=1)

    if title is not None:
        plt.title(title)

    if do_show:
        plt.show()

    return im_to_show

def print_and_clear_cupy_memory(do_clear=True):

    mempool = cp.get_default_memory_pool()
    print("Before clear " + str(mempool.used_bytes()))

    if do_clear:
        mempool.free_all_blocks()
        print("After clear " + str(mempool.used_bytes()))

def print_and_clear_cupy_fft_cache(do_print=True, do_clear=False):

    cache = cp.fft.config.get_plan_cache()
    if do_print:
        cache.show_info()

    if do_clear:    
        cache.clear()
        print("Cleared FFT Cache")

def largest_h5_from_dir(dirpath):

    all_files = [os.path.join(basedir, filename) for basedir, dirs, files in os.walk(dirpath) for filename in files if ("h5" in filename and "ScanArchive" in filename) ]
    sorted_files = sorted(all_files, key=os.path.getsize, reverse=True) # sort descending

    if len(sorted_files) > 1:
        warnings.warn("Multiple h5 in directory")

    return sorted_files[0]

def calculate_adc_map(im_xyz_b0, im_xyz_dw, diff_bval):
    
    with np.errstate(divide='ignore', invalid='ignore'):
        im_divide_xyz = np.true_divide(np.abs(im_xyz_dw), np.abs(im_xyz_b0))
        im_log_xyz = np.log(im_divide_xyz)
        im_log_xyz[im_log_xyz == np.inf] = 0
        im_log_xyz = np.nan_to_num(im_log_xyz)
        
    adc_map = np.abs(-1/diff_bval * im_log_xyz)
    adc_map[adc_map > 5e-3] = 0
    
        
    return np.abs(adc_map)    

def get_us_linear_transform(x):
    """
    Gets the slope and intercept of the linear transform to convert the data contained in the numpy array
    to unsigned short (uint16) values.
    """
    ii16 = np.iinfo(np.int16)
    y = np.array([x.min(), x.max()])
    A = np.array([[ii16.min + 1, 1], [ii16.max - 1, 1]]) # don't go exactly to min/max to avoid rounding errors
    params = np.linalg.solve(A, y)
    slope = params[0]
    intercept = params[1]
    
    return slope, intercept

def convert_to_int16_full_range(unscaled_im):
    assert(unscaled_im.dtype == np.float32)
    slope, intercept = get_us_linear_transform(unscaled_im)
    return ((unscaled_im - intercept) / slope), slope, intercept

def get_git_hash():
    # need this to get rid of the repository ownersehip error when this is run from the docker container
    # https://confluence.atlassian.com/bbkb/git-command-returns-fatal-error-about-the-repository-being-owned-by-someone-else-1167744132.html
    os.system("git config --global --add safe.directory /home/sdkuser/workspace")
    git_exit_code = os.system("git rev-parse --verify HEAD >> /dev/null")
    git_rev = None
    if git_exit_code == 0:
        git_rev = os.popen("git rev-parse --verify HEAD").read().strip()

    return git_rev, git_exit_code

def add_git_hash_to_dicom_header(ds):
    # ds is object generated from dcmread
    tag = get_git_hash()
    #tag = '108391ahbou110hadlh2o8'
    if type(tag) is str:
        ds.add_new([0x0018, 0x9315], "CS", tag[0:16].upper()) # only append 16 since that is the number that is supported

    return ds

