import os
import re
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import scipy.ndimage as nd
from regions import Regions

def regions_to_mask(image_path, region_path):
    """
    Convert a DS9 region file to an image mask.
    
    Arguments
    ---------
    image_path (str)
        Path to fits image to be masked.
    region_path (str)
        Path to DS9 ".reg" file containing the masking regions.

    Returns
    -------
    mask_hdu (astropy.io.fits.PrimaryHDU)
        The generated mask as a fits HDU.
    """

    # Extract the WCS information from the image being masked
    img, hdr = fits.getdata(image_path, header=True)
    wcs = WCS(hdr)

    # Use WCS to convert regions to pixel coordinates.
    regions = Regions.read(region_path, format='ds9')
    pixcoords = [scoord.to_pixel(wcs) for scoord in regions]

    # Build a combined mask by processing each region separately.
    combined_mask = np.zeros_like(img, dtype=bool)
    for region in pixcoords:
        try:
            reg_mask = region.to_mask()
            reg_img = reg_mask.to_image(shape=img.shape).astype(bool)
            combined_mask = np.logical_or(combined_mask, reg_img)

        # Skip regions that cannot be processed.
        except Exception:
            warnings.warn('Skipping a region that could not be converted to a mask.',
                          stacklevel=2)            
            continue

    mask_hdu = fits.PrimaryHDU(combined_mask.astype(np.uint8), header=wcs.to_header())

    return mask_hdu

def create_edge_mask(image_paths, off_image=0, buffer_size=5, threshold=0.1, n_pixels=50):
    """
    Use binary hole filling and sobel filters to identify and mask
    image edges and merge multiple masks into a single combined mask.
    
    Arguments
    ---------
    image_paths (str, List[str])
        Paths to the images from which to create the edge mask.
    off_image (float)
        The value indicating an off detector region.
    buffer_size (int)
        The width of buffer around the image array edge within which to 
        ignore edges. Helps if there is no off_image buffer.
    threshold (float)
        Threshold for edge identification.
    n_pixels (int)
        Number of  pixels to use when dilating the edge mask.

    Returns
    -------
    mask_hdu (astropy.io.fits.PrimaryHDU)
        The generated edge mask as a fits HDU.
    """

    # Convert string to list if required.
    if type(image_paths) == str:
        image_paths = [image_paths]

    masks = []
    for path in image_paths:

        print(f'Finding edges in {path}...')

        # Fill any missing pixels that would be considered an edge.
        sci, hdr = fits.getdata(path, header=True)
        sci = nd.binary_fill_holes(sci)

        # Identify off-image regions
        off_image_mask = (sci == off_image) | (np.isnan(sci))

        # Do not create a mask if the edge identified is with this many
        # pixels of the image edge.
        buffer_mask = np.zeros_like(sci, dtype=bool)
        buffer_mask[:buffer_size, :] = True 
        buffer_mask[-buffer_size:, :] = True  
        buffer_mask[:, :buffer_size] = True  
        buffer_mask[:, -buffer_size:] = True 

        # Detect the edges.
        edges_x = nd.sobel(sci, axis=0)
        edges_y = nd.sobel(sci, axis=1)
        edges = np.sqrt(edges_x**2 + edges_y**2)

        # Reequire a minimum threshold.
        edge_mask = edges > threshold

        # Combine off-image mask and dilated edge mask.
        comb_mask = np.logical_and(off_image_mask, edge_mask)
        # Exclude buffer zone from masking
        comb_mask[buffer_mask] = False  

        # Dilate to get the edge mask
        final_mask = nd.binary_dilation(comb_mask, iterations=n_pixels) 

        masks.append(final_mask)
    
    # Merge masks if required.
    if len(masks) > 1:
        print('Merging...')
        combined_mask = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined_mask |= mask
    else:
        combined_mask = masks[0]

    hdr = fits.getheader(image_paths[0])
    wcs = WCS(hdr)
    mask_hdu = fits.PrimaryHDU(combined_mask.astype(np.uint8), header=wcs.to_header())

    return mask_hdu

def clean_regions(region_file):
    """
    Remove regions with zero area from a region file. These can 
    otherwise cause problems when converting to a mask.

    Arguments
    ---------
    region_file (str)
        Path to region file to clean.
    """

    # Remove any non-numeric characters.
    def clean_value(value):
        return float(re.sub(r'[^\d.]+', '', value))
    
    with open(region_file, 'r') as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        # Always keep lines that don't start with standard region types.
        if not any(line.startswith(region) for region in ['ellipse', 'box', 'circle']):
            filtered_lines.append(line)
            continue

        # Get dimensions of each region type and only kepp is > 0.
        if 'box' in line:
            parts = line.split('(')[1].split(')')[0].split(',')
            width = clean_value(parts[2])
            height = clean_value(parts[3])
            if width > 0 and height > 0:
                filtered_lines.append(line)
        elif 'circle' in line:
            parts = line.split('(')[1].split(')')[0].split(',')
            radius = clean_value(parts[2])
            if radius > 0:
                filtered_lines.append(line)
        elif 'ellipse' in line:
            parts = line.split('(')[1].split(')')[0].split(',')
            width = clean_value(parts[2])
            height = clean_value(parts[3])
            if width > 0 and height > 0:
                filtered_lines.append(line)

    # Overwrite the old region file.
    os.remove(region_file)
    with open(region_file, 'w') as f:
        f.writelines(filtered_lines)

    return