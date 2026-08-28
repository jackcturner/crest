import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import scipy.ndimage as nd
import scipy.optimize as opt

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
    rms_hdu (astropy.io.fits.PrimaryHDU)
        The stacked rms image as an astropy HDU object.
    """

    if len(science_paths) != len(weight_paths):
        raise ValueError('The number of science and weight images must be equal.')
    if len(science_paths) <= 1:
        raise ValueError('At least two images are required to create a stack.')

    # Get the image size and header.
    img, sci_hdr = fits.getdata(science_paths[hdr_index], header=True)
    wht_hdr = fits.getheader(weight_paths[hdr_index])

    shape = img.shape
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
    stack_wht = 1 / np.sqrt(stack_wht)

    sci_hdu = fits.PrimaryHDU(stack_sci.astype(np.float32), header=sci_hdr)
    rms_hdu = fits.PrimaryHDU(stack_wht.astype(np.float32), header=wht_hdr)

    return sci_hdu, rms_hdu

def max_sn_image(science_paths, rms_paths):
    """
    Create an image where each pixel value is the maximum signal-to-noise
    ratio across a set of filter images.
    
    Arguments
    ---------
    science_paths (List[str])
        Filenames of science images to stack.
    rms_paths (List[str])
        Filenames of the corresponding RMS images.
        
    Returns
    -------
    det_hdu (astropy.io.fits.PrimaryHDU)
        The detection image as an astropy HDU object.
    """

    # Construct an empty new image.
    with fits.open(science_paths[0]) as hdul:
        shape = hdul[0].data.shape
        wcs = WCS(hdul[0].header)

    max_sn = np.full(shape, 0, dtype=np.float32)

    # Comput the maximum S/N in each pixel.
    for sfile, rfile in zip(science_paths, rms_paths):
        sci = fits.getdata(sfile)
        rms = 1 / np.sqrt(fits.getdata(rfile))
        valid = np.isfinite(sci) & np.isfinite(rms) & (rms > 0)
        rms[~valid] = 0

        sn = np.zeros_like(sci, dtype=np.float32)
        np.divide(sci, rms, out=sn, where=valid)

        max_sn = np.maximum(max_sn, sn)

    # Replace invalid pixels.
    max_sn[~np.isfinite(max_sn)] = 0.0

    # Record the names of the images used.
    header = wcs.to_header()
    header["HISTORY"] = "Science images used:"
    for path in science_paths:
        header["HISTORY"] = f"{os.path.basename(path)}"

    header["HISTORY"] = "RMS images used:"
    for path in rms_paths:
        header["HISTORY"] = f"{os.path.basename(path)}"

    det_hdu = fits.PrimaryHDU(max_sn.astype(np.float32), header=header)

    return det_hdu

def chi2_image(science_paths, weight_paths):
    """
    Construct a chi-squared image from a set of science and 
    weight images.
    
    Arguments
    ---------
    science_paths (List[str])
        Filenames of science images to stack.
    weight_paths (List[str])
        Filenames of the corresponding weight images.
        
    Returns
    -------
    det_hdu (astropy.io.fits.PrimaryHDU)
        The detection image as an astropy HDU object.
    wht_hdu (astropy.io.fits.PrimaryHDU)
        The weight image as an astropy HDU object.
    """

    # Construct empty new images.
    with fits.open(science_paths[0]) as hdul:
        shape = hdul[0].data.shape
        wcs = WCS(hdul[0].header)

    sum_w_s2 = np.zeros(shape, dtype=np.float32)
    n_pix = np.zeros(shape, dtype=np.int16)

    # Sum the weighted squares of the science images.
    for sfile, wfile in zip(science_paths, weight_paths):
        sci = fits.getdata(sfile)
        wht = fits.getdata(wfile)

        valid = wht > 0

        sum_w_s2[valid] += wht[valid] * sci[valid]**2
        n_pix[valid] += 1

    # Compute the chi-squared image.
    chi2 = np.zeros_like(sum_w_s2)
    valid = n_pix > 0

    chi2[valid] = np.sqrt(sum_w_s2[valid]) / np.sqrt(n_pix[valid])
    det_wht = (n_pix > 0)

    # Record the names of the images used.
    header = wcs.to_header()
    header["HISTORY"] = "Science images used:"
    for path in science_paths:
        header["HISTORY"] = f"{os.path.basename(path)}"
    header["HISTORY"] = "Weight images used:"
    for path in weight_paths:
        header["HISTORY"] = f"{os.path.basename(path)}"

    det_hdu = fits.PrimaryHDU(chi2.astype(np.float32), header=header)
    wht_hdu = fits.PrimaryHDU(det_wht.astype(np.float32), header=header)

    return det_hdu, wht_hdu

def Gaussian_2D(coord, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """
    2D Gaussian fitting function.
    
    Arguments
    ---------
    coord (List[float])
        The x,y coordinate at which to evaluate the Gaussian.
    xo (float)
        The x-coordinate of the centre.
    yo (float)
        The y-coordinate of the centre.
    sigma_x (float)
        Standard deviation in the x direction in pixels.
    sigma_y (float)
        Standard deviation in the y direction in pixels.
    amplitude (float)
        Amplitude of the gaussian.
    offset (float)
        Offset to apply to the Gaussian values.

    Returns
    -------
    flat_gaussian (numpy.ndarray)
        1D flattened Gaussian distribution.
    """

    gaussian = offset + amplitude*np.exp( - (((coord[0]-float(xo))**2)/(2*sigma_x**2)
                                             + ((coord[1]-float(yo))**2)/(2*sigma_y**2)))

    flat_gaussian = gaussian.ravel()

    return flat_gaussian

def get_PSF_FWHM(psf_path):
    """
    Get FWHM of a PSF in x and y directions using Gaussian fitting.

    Arguments
    ---------
    psf_path (str)
        Filename of fits image PSF.

    Returns
    -------
    fwhm (List[float])
        Measured FWHM in x and y directions.
    """

    # Read the PSF array from the fits file.
    img = fits.getdata(psf_path)

    # Create an x and y grid.
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    
    # Some parameter inital guesses
    initial_guess = [img.shape[1]/2,img.shape[0]/2,10,10,1,0]

    # Fit with a Gaussian model.
    popt, pcov = opt.curve_fit(Gaussian_2D, (x, y), 
                               img.ravel(), p0 = initial_guess)
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]

    # Convert the standard deviations to FWHM.
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))

    fwhm = [FWHM_x, FWHM_y]

    return fwhm