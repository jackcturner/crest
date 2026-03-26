from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import scipy.ndimage as nd

def _pc2cd(hdr, key=' '):
    """
    Convert a PC matrix to a CD matrix.

    Arguments
    ---------
    hdr (astropy.io.fits.Header)
        Astropy header containing PC matrix to be converted.
    key (str)
        Additional key attached to the PC keywords.

    Returns
    -------
    hdr (astropy.io.fits.Header)
        Astropy table including generated CD matrix.
    """

    key = key.strip()
    cdelt1 = hdr.pop(f'CDELT1{key:.1s}', 1)
    cdelt2 = hdr.pop(f'CDELT2{key:.1s}', 1)
    hdr[f'CD1_1{key:.1s}'] = (cdelt1 * hdr.pop(f'PC1_1{key:.1s}', 1),
                              'partial of first axis coordinate w.r.t. x')
    hdr[f'CD1_2{key:.1s}'] = (cdelt1 * hdr.pop(f'PC1_2{key:.1s}', 0),
                              'partial of first axis coordinate w.r.t. y')
    hdr[f'CD2_1{key:.1s}'] = (cdelt2 * hdr.pop(f'PC2_1{key:.1s}', 0),
                              'partial of second axis coordinate w.r.t. x')
    hdr[f'CD2_2{key:.1s}'] = (cdelt2 * hdr.pop(f'PC2_2{key:.1s}', 1),
                              'partial of second axis coordinate w.r.t. y')
    return hdr


def _block_reduce(img, block_size, mode='sum'):
    """
    Reduce a 2D image by combining non-overlapping blocks.

    Arguments
    -----------
    img (numpy.ndarray)
        The 2D image to be reduced.
    block_size (tuple)
        The size of the blocks to combine, given as (by, bx).
    mode (str)
        The method to use when combining pixels. Options are:
        - 'sum': Sum the pixel values in each block.
        - 'quad': Sum of squares in each block.
        - 'mean': Mean within each block.

    Returns
    -------
    reshaped (numpy.ndarray)
        The reduced 2D image.
    """

    if img.ndim != 2:
        raise ValueError('Only 2D images are supported')
    by, bx = block_size
    if by <= 0 or bx <= 0:
        raise ValueError('Block dimensions must be positive integers')

    ny = img.shape[0] // by
    nx = img.shape[1] // bx
    if ny == 0 or nx == 0:
        raise ValueError('Block size larger than image dimensions')

    trimmed = img[: ny * by, : nx * bx]
    reshaped = trimmed.reshape(ny, by, nx, bx)

    if mode == 'sum':
        return reshaped.sum(axis=(1, 3))
    elif mode == 'quad':
        return np.sqrt((reshaped ** 2).sum(axis=(1, 3)))
    elif mode == 'mean':
        return reshaped.sum(axis=(1, 3)) / (by * bx)
    else:
        raise KeyError(f'{mode} is not an available method')

def rebin_image(image_path, source_scale, target_scale, method='sum'):
    """
    Rebin an image to a lower resolution pixel scale by combining pixels.
    
    Arguments
    ---------
    image_path (str)
        Path to the fits image file to be rebinned.
    source_scale (float)
        The pixel scale of the input image in arcseconds.
    target_scale (float)
        The desired output pixel scale in arcseconds.
    method (str)
        The method to use when combining pixels.
        Either 'sum', 'mean' or'quad'.

    Returns
    -------
    hdu (astropy.io.fits.PrimaryHDU)
        The rebinned image as an astropy HDU object.
    """

    # Read the image.
    with fits.open(image_path) as hdul:
        img = hdul[0].data
        wcs = WCS(hdul[0].header)

    # Calculate the scale factor for resizing.
    scale_factor = int(target_scale/source_scale)
    block_size = (scale_factor, scale_factor)

    # Create a new WCS for the rebinned image
    wcs_rebinned = wcs.slice((np.s_[:None:int(scale_factor)], np.s_[:None:int(scale_factor)]))
    wcs_header = wcs_rebinned.to_header()
    wcs_header = _pc2cd(wcs_header)
    
    # Rebin the image using requested method.
    if method == 'sum':
        rebinned_image = _block_reduce(img, block_size, mode='sum')
    elif method == 'quad':
        rebinned_image = _block_reduce(img, block_size, mode='quad')
    elif method == 'mean':
        rebinned_image = _block_reduce(img, block_size, mode='mean')
    else:
        raise KeyError(f'{method} is not an available method')

    # Return an HDU object.
    hdu = fits.PrimaryHDU(rebinned_image.astype(np.float32), header=wcs_header)
    hdu.header["REBIN"] = (source_scale, 'Image has been rebinned from this scale')

    return hdu

def generate_error_map(science_path, weight_path, exposure_path, grow=4):
    """
    Generate an error map including Poisson noise from science, weight
    and exposure images. See:
    https://dawn-cph.github.io/dja/blog/2023/07/18/image-data-products/

    Arguments
    ---------
    science_path (str)
        Path to the science fits image.
    weight_path (str)
        Path to the weight fits image.
    exposure_path (str):
        Path to the exposure fits image.
    grow (int)
        The factor by which to grow the exposure map.
        
    Returns
    -------
    hdu (astropy.io.fits.PrimaryHDU)
        The generated error map as an astropy HDU object.
    """

    # Load in each image.
    sci = fits.getdata(science_path)
    exp, exp_header = fits.getdata(exposure_path, header = True)
    wht, wht_header = fits.getdata(weight_path, header = True)

    # Grow the exposure map to the original frame if required.
    if grow > 1:
        factor = int(grow)
        offset = factor // 2

        full_exp = np.zeros(sci.shape, dtype=int)
        full_exp[offset::factor, offset::factor] = (exp * 1).astype(int)
        full_exp = nd.maximum_filter(full_exp, factor)
    else:
        full_exp = exp.astype(int)

    # Determine multiplicative factors that have been applied since the
    # original count-rate images.
    phot_scale = 1.
    for k in ['PHOTMJSR','PHOTSCAL']:
        try:
            print(f'{k} {exp_header[k]:.3f}')
            phot_scale /= exp_header[k]
        except:
            print(f'{k} not found.')

    # Unit and pixel area scale factors.
    if 'OPHOTFNU' in exp_header:
        phot_scale *= exp_header['PHOTFNU'] / exp_header['OPHOTFNU']

    # Poisson variance in mosaic DN.
    effective_gain = (phot_scale * full_exp)
    var_poisson_dn = np.maximum(sci, 0) / effective_gain

    # Original variance from the weight image.
    var_wht = 1/wht

    # New total variance.
    var_total = var_wht + var_poisson_dn
    full_wht = 1 / var_total
    full_wht[var_total <= 0] = 0

    # Convert to an error map.
    err = np.where(full_wht==0, 0, 1/np.sqrt(full_wht))

    primary_hdu = fits.PrimaryHDU(err.astype(np.float32), header=wht_header)
    hdu = fits.HDUList([primary_hdu])

    return hdu

def create_stack(science_paths, weight_paths, weight_is_rms=False, hdr_index=0,):
    """
    Create a variance weighted stacked image.

    Arguments
    ---------
    science_paths (List[str])
        Filenames of science images to stack.
    weight_paths (List[str])
        Filenames of the corresponding weight images.
    weight_is_rms (bool)
        If True, the weight images are treated as RMS maps.
    hdr_index (int)
        Use the header from this element of science_paths in the for the
        stacked image.

    Returns
    -------
    sci_hdu (astropy.io.fits.PrimaryHDU)
        The stacked science image as an astropy HDU object.
    wht_hdu (astropy.io.fits.PrimaryHDU)
        The stacked weight image as an astropy HDU object.
    """

    if len(science_paths) != len(weight_paths):
        raise ValueError('The number of science and weight images must be equal.')
    if len(science_paths) <= 1:
        raise ValueError('At least two images are required to create a stack.')

    # Get the image size and header.
    img, sci_hdr = fits.getdata(science_paths[hdr_index], header=True)
    wht_hdr = fits.getheader(weight_paths[hdr_index])

    shape = img.data.shape
    stack_sci = np.zeros(shape)
    stack_wht = np.zeros(shape)

    # Stack the images. 
    for sci, wht in zip(science_paths, weight_paths):
        wht_ = fits.getdata(wht)
        if weight_is_rms:
            wht_ = 1/(wht_**2)

        stack_sci += fits.getdata(sci) * wht_
        stack_wht += wht_

    stack_sci /= stack_wht

    sci_hdu = fits.PrimaryHDU(stack_sci.astype(np.float32), header=sci_hdr)
    wht_hdu = fits.PrimaryHDU(stack_wht.astype(np.float32), header=wht_hdr)

    return sci_hdu, wht_hdu