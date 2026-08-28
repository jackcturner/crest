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
        off_image_mask = (sci == off_image) | (~np.isfinite(sci))

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

def gaia_table_to_regions(catalogue, threshold, outname, major_axis=1.0, minor_axis=1.0,
                          colour="green", width=1, include_text=False):
    """
    Create a DS9 region file with ellipses at GAIA source positions, 
    based on a preconstructed catalogue.

    Arguments
    ---------
    catalogue (astropy.table.Table)
        Table containing GAIA source information.
    threshold (float)
        Star probability threshold for region placement.
    outname (str)
        Output DS9 region path.
    major_axis (float)
        Ellipse semi-major axis in arcsec. Adjust based on resolution.
    minor_axis (float)
        Ellipse semi-minor axis in arcsec. Adjust based on resolution.
    colour (str)
        DS9 region colour. Adjust to improve visibility.
    width (int)
        DS9 region line width. Adjust to improve visibility.
    include_text (bool)
        If True, include source_id labels in the region text field.
    """

    required_cols = ["ra", "dec", "classprob_dsc_combmod_star"]
    for col in required_cols:
        if col not in catalogue.colnames:
            raise KeyError(f"Missing required column '{col}' in catalogue.")

    has_source_id = "source_id" in catalogue.colnames
    if not has_source_id and ("SOURCE_ID" in catalogue.colnames):
        source_id_col = "SOURCE_ID"
        has_source_id = True
    else:
        source_id_col = "source_id"

    # Select source above the star classification threshold.
    mask = np.asarray(catalogue["classprob_dsc_combmod_star"]) > threshold
    selected = catalogue[mask]

    # Construct the region file header.
    header = [
        "# Region file format: DS9 version 4.1",
        f"global color={colour} width={width} dashlist=8 3 font='helvetica 10 normal' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "icrs",
    ]

    # Place each region.
    with open(outname, "w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")

        for row in selected:
            ra = float(row["ra"])
            dec = float(row["dec"])
            region = f"ellipse({ra:.8f},{dec:.8f},{major_axis}\",{minor_axis}\",{0.0})"

            if include_text and has_source_id:
                source_id = row[source_id_col]
                region += f" # text={{SOURCE_ID={source_id}}}"

            f.write(region + "\n")

    print(f"Wrote {len(selected)} regions to {outname}")

    return outname

def gaia_query_to_regions(imgs, threshold, outname, gaia_table="gaiadr3.gaia_source", 
                          major_axis=1.0, minor_axis=1.0, colour="green", width=1, 
                          include_text=False, tap_url="https://gaia.ari.uni-heidelberg.de/tap"):
    """
    Query the GAIA database to create a DS9 region file with ellipses at 
    source positions.

    Arguments
    ---------
    imgs (List[str])
        Find GAIA sources within the coverage of these images.
    threshold (float)
        Star probability threshold for region placement.
    outname (str)
        Output DS9 region path.
    gaia_table (str)
        GAIA table to query.
    major_axis (float)
        Ellipse semi-major axis in arcsec. Adjust based on resolution.
    minor_axis (float)
        Ellipse semi-minor axis in arcsec. Adjust based on resolution.
    colour (str)
        DS9 region colour. Adjust to improve visibility.
    width (int)
        DS9 region line width. Adjust to improve visibility.
    include_text (bool)
        If True, include source_id labels in the region text field.
    tap_url (str)
        URL of the GAIA TAP service.

    Returns
    -------
    tables (List[astropy.table.Table])
        GAIA result tables for each image.
    region_paths (List[str])
        DS9 region file paths for each image.
    """

    from astroquery.utils.tap.core import TapPlus
    tap = TapPlus(url=tap_url)

    tables = []
    region_paths = []

    columns = [
        "source_id", "ra", "dec", "phot_g_mean_mag",
        "classprob_dsc_combmod_star",
        "classprob_dsc_combmod_quasar",
        "classprob_dsc_combmod_galaxy",
        "parallax", "parallax_error", "parallax_over_error",
    ]
    col_str = ", ".join(columns)

    for img_path in imgs:
        _, hdr = fits.getdata(img_path, header=True)
        ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
        wcs = WCS(hdr)

        pixel_corners = np.array([[0, 0], [0, ny], [nx, ny], [nx, 0]])
        ra_vals, dec_vals = wcs.pixel_to_world_values(
            pixel_corners[:, 0], pixel_corners[:, 1]
        )

        min_ra, max_ra = np.min(ra_vals), np.max(ra_vals)
        min_dec, max_dec = np.min(dec_vals), np.max(dec_vals)

        width_deg = max_ra - min_ra
        height_deg = max_dec - min_dec
        center_ra = (min_ra + max_ra) / 2
        center_dec = (min_dec + max_dec) / 2

        query = f"""
        SELECT {col_str}
        FROM {gaia_table}
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            BOX('ICRS', {center_ra}, {center_dec}, {width_deg}, {height_deg})
        )
        """

        job = tap.launch_job(query)
        result_table = job.get_results()
        tables.append(result_table)

        gaia_table_to_regions(
            result_table, threshold, outname, major_axis, minor_axis, colour, width, include_text)
        region_paths.append(outname)

    return tables, region_paths