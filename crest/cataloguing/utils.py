import os
import h5py
import urllib.request
from urllib.error import URLError
from xml.etree import ElementTree

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

def merge_catalogues(catalogue_paths, labels, merged_path='merged_catalogue.hdf5'):
    """
    Combine multiple hdf5 catalogues produced by CREST into a single 
    catalogue where each has its own group within the photometry group.
    
    Arguments
    ---------
    catalogue_paths (List[str])
        A list of paths to the catalogues to be combined.
    labels (List[str])
        List of group names for each catalogue added to the new
        catalogue.
    merged_path (str)
        Path to the merged catalogue to output.
    """

    if len(catalogue_paths) != len(labels):
        raise ValueError('Number of catalogue paths and labels must be the same.')

    # Create the new catalogue and its base photometry group.
    with h5py.File(merged_path, 'w') as newcat:
        newcat.create_group('photometry')

        # Store the data from each catalogue in its own group.
        for label, catalogue_path in zip(labels, catalogue_paths):

            with h5py.File(catalogue_path, 'r') as cat:

                if 'photometry' not in cat.keys():
                    raise KeyError(('Catalogue should have a base "photometry" group.'
                                    'Was this catalogue produced by CREST?'))
                
                group = f'photometry/{label}'
                newcat.create_group(group)

                # Add each dataset and attribute to the new catalogue.
                for key in cat['photometry'].keys():

                    # Have consistent naming convention for 
                    # Source Extractor apertures.
                    if (key == 'FLUX_APER') or (key == 'FLUXERR_APER'):
                        newcat[f'{group}/{key}_0'] = cat[f'photometry/{key}'][()]
                    else:
                        newcat[f'{group}/{key}'] = cat[f'photometry/{key}'][()]

                for key in cat['photometry'].attrs.keys():
                    newcat[f'{group}'].attrs[key] = cat['photometry'].attrs[key]
    
    return merged_path

# Effective wavelengths in microns of NED correction bands.
NED_FILTER_WAVELENGTH_UM = {
    'Landolt U': 0.35, 'Landolt B': 0.43, 'Landolt V': 0.54, 'Landolt R': 0.64, 'Landolt I': 0.80,
    'CTIO U': 0.37, 'CTIO B': 0.43, 'CTIO V': 0.55, 'CTIO R': 0.65, 'CTIO I': 0.80, 'UKIRT J': 1.25,
    'UKIRT H': 1.66, 'UKIRT K': 2.19, 'UKIRT L': 3.78, 'Gunn g': 0.52, 'Gunn r': 0.66, 
    'Gunn i': 0.79, 'Gunn z': 0.91, 'Spinrad R_S': 0.69, 'Stromgren u': 0.35, 'Stromgren b': 0.47,
    'Stromgren v': 0.41, 'Stromgren beta': 0.49, 'Stromgren y': 0.55, 'SDSS u': 0.36, 
    'SDSS g': 0.47, 'SDSS r': 0.62, 'SDSS i': 0.75, 'SDSS z': 0.89, 'DSS-II g': 0.46, 
    'DSS-II r': 0.65, 'DSS-II i': 0.81, 'PS1 g': 0.49, 'PS1 r': 0.62, 'PS1 i': 0.75, 'PS1 z': 0.87,
    'PS1 y': 0.97, 'PS1 w': 0.62, 'LSST u': 0.37, 'LSST g': 0.48, 'LSST r': 0.62, 'LSST i': 0.75,
    'LSST z': 0.87, 'LSST y': 0.97, 'WFPC2 F300W': 0.31, 'WFPC2 F450W': 0.46, 'WFPC2 F555W': 0.54,
    'WFPC2 F606W': 0.60, 'WFPC2 F702W': 0.69, 'WFPC2 F814W': 0.79, 'WFC3 F105W': 1.04, 
    'WFC3 F110W': 1.12, 'WFC3 F125W': 1.23, 'WFC3 F140W': 1.37, 'WFC3 F160W': 1.53, 
    'WFC3 F200LP': 0.55, 'WFC3 F218W': 0.22, 'WFC3 F225W': 0.24, 'WFC3 F275W': 0.27, 
    'WFC3 F300X': 0.29, 'WFC3 F336W': 0.34, 'WFC3 F350LP': 0.59, 'WFC3 F390W': 0.40, 
    'WFC3 F438W': 0.43, 'WFC3 F475W': 0.48, 'WFC3 F475X': 0.50, 'WFC3 F555W': 0.53,
    'WFC3 F600LP': 0.74, 'WFC3 F606W': 0.59, 'WFC3 F625W': 0.62, 'WFC3 F775W': 0.76,
    'WFC3 F814W': 0.80, 'WFC3 F850LP': 0.91, 'ACS clear': 0.62, 'ACS F435W': 0.43, 
    'ACS F475W': 0.48, 'ACS F550M': 0.56, 'ACS F555W': 0.54, 'ACS F606W': 0.59, 'ACS F625W': 0.63,
    'ACS F775W': 0.77, 'ACS F814W': 0.80, 'ACS F850LP': 0.90, 'DES g': 0.48, 'DES r': 0.64,
    'DES i': 0.78, 'DES z': 0.91, 'DES Y': 0.99}

def _get_svo_effective_wavelength_um(filter_code, timeout=10):
    """
    Return effective wavelength (micron) for an SVO filter code.

    Arguments
    ---------
    filter_code (str)
        SVO filter code in the format OBS/INST.FILTER.
    timeout (int)
        Timeout in seconds for the SVO query.

    Returns
    -------
    value (float)
        The effective wavelength in microns.
    """

    # Contruct the SVO URL.
    if ('/' not in filter_code) or ('.' not in filter_code):
        raise ValueError(f'Filter code must be OBS/INST.FILTER, got: {filter_code}')

    observatory = filter_code.split('/')[0]
    instrument = filter_code.split('/')[1].split('.')[0]
    filter_name = filter_code.split('.')[-1]
    ID = f'{observatory}/{instrument}.{filter_name}'
    svo_url = f'http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID={ID}'

    # Query SVO and parse the effective wavelength from the response.
    try:
        with urllib.request.urlopen(svo_url, timeout=timeout) as f:
            root = ElementTree.parse(f).getroot()
    except URLError as exc:
        raise RuntimeError(f'SVO could not be accessed for {filter_code} ({svo_url}).') from exc

    try:
        params = root.findall('.//PARAM')
        target_param = next(
            p
            for p in params
            if p.attrib.get('name', '') in ('WavelengthEff', 'WavelengthPivot', 'WavelengthMean')
        )

        value = float(target_param.attrib.get('value'))
        wavelength_unit = target_param.attrib.get('unit', '')

        # If unit is not provided, try to find it from other parameters.
        if wavelength_unit == '':
            wavelength_unit = next(
                p.attrib.get('value', '')
                for p in params
                if p.attrib.get('name', '') == 'WavelengthUnit'
            )

        # Convert to microns if necessary.
        if wavelength_unit in ('um', 'micron', 'microns'):
            return value
        if wavelength_unit in ('nm', 'nanometer', 'nanometers'):
            return value / 1e3
        if wavelength_unit in ('Angstrom', 'angstrom', 'A'):
            return value / 1e4
    except Exception as exc:
        raise ValueError(f'Wavelength could not be recovered from SVO for {filter_code}.') from exc

    raise ValueError(f'Wavelength could not be recovered from SVO for {filter_code}.')

def _resolve_correction_filter_svo(filter_code):
    """
    Find the NED correction filter closest in wavelength to an SVO code.

    Arguments
    ---------
    filter_code (str)
        SVO filter code in the format OBS/INST.FILTER.
    
    Returns
    -------
    corr_filter (str)
        The name of the matched NED correction filter.
    """

    if '/' not in filter_code:
        raise ValueError(f'Unexpected filter code format: {filter_code}')
    lam_um = _get_svo_effective_wavelength_um(filter_code)

    corr_filter, _ = min(NED_FILTER_WAVELENGTH_UM.items(), key=lambda item: abs(lam_um - item[1]),)
    
    return corr_filter

def correct_extinction(catalogue, flux_err_pairs, to_jy, replace=False, suffix='_EXT',
                       ra_key='ALPHA_SKY', dec_key='DELTA_SKY'):
    """
    Query the NED extinction calculator using mean RA and DEC location
    and apply correction to CREST catalogue.

    Correction filters are selected automatically by querying SVO for each
    filter's effective wavelength and then mapping to the nearest supported
    NED correction band.
    
    Arguments
    ---------
    catalogue (str)
        Path to CREST hdf5 catalogue.
    flux_err_pairs (List[Tuple[str, str/None]]])
        List of (flux_key, error_key) pairs to correct. Can set error_key 
        to None if no error is available.
    to_jy (float)
        Multiplicative factor to convert fluxes to Jy.
    replace (bool)
        Should the original flux values be replaced?
    suffix (str)
        If original values are not replaced, create new dataset using
        this suffix.
    ra_key (str)
        Name of the RA dataset in the catalogue.
    dec_key (str)
        Name of the DEC dataset in the catalogue.
    """

    try:
        from ned_extinction_calc import request_extinctions
    except ImportError as exc:
        raise ImportError(
            'correct_extinction requires the optional dependency "ned_extinction_calc", which is '
            'not installed. Please find it on GitHub and follow the installation instructions.'
        ) from exc

    # Read the catalogue.
    with h5py.File(catalogue, 'r+') as f:

        print(f'Correcting {catalogue} for extinction.')

        # Get list of instruments and associated filters.
        observatories = list(f['photometry'].keys())
        if 'DETECTION' in observatories:
            observatories.remove('DETECTION')

        filters = []
        for observatory in observatories:
            filters_ = list(f[f'photometry/{observatory}'].keys())
            for i in filters_:
                filters_[filters_.index(i)] = f'{observatory}/{i}'
            filters += filters_
        
        # Determine the closest matching dust map.
        for filter in filters:
            corr_filt = _resolve_correction_filter_svo(filter)
            print(corr_filt)
            
            # Get the extinction in magnitudes at the approximate centre 
            # of the image.
            ra = str(np.mean(f[f'photometry/{filter}/{ra_key}'][:]))
            dec = str(np.mean(f[f'photometry/{filter}/{dec_key}'][:]))

            Alam = request_extinctions(ra, dec, filters=corr_filt)

            # Apply the correction to each requested flux key.
            keys = f[f'photometry/{filter}'].keys()
            for flux_key, err_key in flux_err_pairs:

                if flux_key not in keys:
                    raise KeyError(f'Flux key {flux_key} not found in photometry/{filter}.')

                print(f'Working on {flux_key}...')

                # Get the flux in Jy.
                flux = f[f'photometry/{filter}/{flux_key}'][:] 
                flux_corr = flux.copy()
                flux *= to_jy

                # Keep original values if not detected.
                s = (flux > 0) & (~np.isnan(flux)) & (np.isfinite(flux))

                # Convert to magnitude and apply correction.
                mag = (-2.5 * np.log10(flux[s])) + 8.90
                mag -= Alam

                flux_corr[s] = (10**((mag-8.90)/-2.5)) / to_jy

                if replace == True:
                    del f[f'photometry/{filter}/{flux_key}']
                    f[f'photometry/{filter}/{flux_key}'] = flux_corr
                else:
                    if f'{flux_key}{suffix}' in f[f'photometry/{filter}'].keys():
                        del f[f'photometry/{filter}/{flux_key}{suffix}']
                    f[f'photometry/{filter}/{flux_key}{suffix}'] = flux_corr

                # If error is provided, correct by maintaining
                # signal to noise ratio.
                if err_key is not None:
                    if err_key not in keys:
                        raise KeyError(f'Error key {err_key} not found in photometry/{filter}.')

                    # Calculate original signal to noise.
                    err = f[f'photometry/{filter}/{err_key}'][:] * to_jy
                    s_n = flux/err

                    # Scale error to maintain this.
                    err_corr = flux_corr/s_n

                    if replace == True:
                        del f[f'photometry/{filter}/{err_key}']
                        f[f'photometry/{filter}/{err_key}'] = err_corr
                    else:
                        if f'{err_key}{suffix}' in f[f'photometry/{filter}'].keys():
                            del f[f'photometry/{filter}/{err_key}{suffix}']
                        f[f'photometry/{filter}/{err_key}{suffix}'] = err_corr
                
    return

def match_gaia(catalogue, bands, gaia_catalogue, tolerance, ra_name='ALPHA_SKY', 
                       dec_name='DELTA_SKY', angle_unit=u.degree):
    """
    Match and flag sources in a CREST catalogue to those in a GAIA
    star catalogue
    
    Arguments
    ---------
    catalogue (str)
        Filename of CREST hdf5 catalogue.
    bands (List[str])
        List containing the catalogue groups to match.
    gaia_catalogue (str)
        Filename of the fits GAIA star catalogue.
    tolerance (float)
        The matching tolerance in arcseconds.
    ra_name (str):
        Name of the RA dataset in the catalogue.
    dec_name (str)
        Name of the DEC dataset in the catalogue.
    angle_unit (astropy.units.Unit)
        The unit of angular tolerance and RA and DEC.

    """

    tolerance = tolerance*u.arcsec

    # Load the GAIA catalogue and get source positions.
    gaia_cat = Table.read(gaia_catalogue)
    gaia_coord = SkyCoord(ra=np.array(gaia_cat['ra']) * angle_unit,
                          dec=np.array(gaia_cat['dec']) * angle_unit)

    # For each band requested.
    with h5py.File(catalogue, 'r+') as cat:

        for band in bands:

            # Store the flags here..
            star_flag = np.zeros(len(cat[f'photometry/{band}/{ra_name}'][:]))
            quasar_flag = np.zeros(len(cat[f'photometry/{band}/{ra_name}'][:]))
            p_flag = np.zeros(len(cat[f'photometry/{band}/{ra_name}'][:]))

            # Match each source to the GAIA catalogue.
            cat_coord = SkyCoord(ra=cat[f'photometry/{band}/{ra_name}'][:] * angle_unit,
                                 dec=cat[f'photometry/{band}/{dec_name}'][:] * angle_unit)

            # Find the source closest to each gaia source.
            idx, d2d, d3d = gaia_coord.match_to_catalog_sky(cat_coord)
            d2d = d2d.to('arcsec')
            s = (d2d < tolerance)

            # Add the varous flags.
            for index, class_s in zip(idx[s], gaia_cat['classprob_dsc_combmod_star'][s]):
                star_flag[index] = class_s
            for index, class_q in zip(idx[s], gaia_cat['classprob_dsc_combmod_quasar'][s]):
                quasar_flag[index] = class_q
            for index, p_over_e in zip(idx[s], gaia_cat['parallax_over_error'][s]):
                p_flag[index] = p_over_e

            # Add the flag array.
            if 'GAIA_STAR' in cat[f'photometry/{band}'].keys():
                del cat[f'photometry/{band}/GAIA_STAR']
            cat[f'photometry/{band}/GAIA_STAR'] = star_flag

            if 'GAIA_QUASAR' in cat[f'photometry/{band}'].keys():
                del cat[f'photometry/{band}/GAIA_QUASAR']
            cat[f'photometry/{band}/GAIA_QUASAR'] = quasar_flag

            if 'GAIA_POE' in cat[f'photometry/{band}'].keys():
                del cat[f'photometry/{band}/GAIA_POE']
            cat[f'photometry/{band}/GAIA_POE'] = p_flag
    
    return

def gaia_cutouts(catalogue, img, wcs, side_length=200):
    """
    Plot cutouts of an image at the location of GAIA sources.
    
    Arguments
    ---------
    catalogue (astropy.table.Table)
        Astropy table containing the GAIA sources.
    img (np.ndarray)
        2D array from which to extract the cutout.
    wcs (astropy.wcs.WCS)
        Astropy WCS object for image array.
    side_length (int)
        The side length of the cutouts in pixels.
    """

    # For each GAIA source.
    for row in catalogue:

        # Extract useful information.
        ra = row['ra'] 
        dec = row['dec']
        class_s = row['classprob_dsc_combmod_star']
        class_q = row['classprob_dsc_combmod_quasar']
        class_g = row['classprob_dsc_combmod_galaxy']
        id = row['SOURCE_ID']
        parallax = row['parallax']

        # Extract the cutout from the original image.
        coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
        try:
            cutout = Cutout2D(img, coord, (side_length, side_length), wcs=wcs)
            
        # Catalogue can sometimes contain sources outside the bounds of 
        # the image. Account for this.
        except:
            print(str(id)+' is beyond the image boundry.')
            continue

        # Plot the cutout
        print(id)
        fig, ax = plt.subplots(figsize=(3.78, 3.78))
        ax.imshow(cutout.data, origin='lower', cmap='gray', norm = SymLogNorm(linthresh=0.03))
        ax.set_title('ID: ' + str(id) + 
                     ' RA: {:.2f}, DEC: {:.2f}, S: {:.2f}, Q: {:.2f}, G: {:.2f}, P: {:.2f}'.format(
                         ra, dec, class_s, class_q, class_g, parallax))       
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(label='Intensity')
        plt.show()

    return

def inspect_gaia(imgs, gaia_table="gaiadr3.gaia_source"):
    """
    Query the GAIA database and create cutouts from an image to identify
    missclassifications.
    
    Arguments
    ---------
    imgs (List[str])
        List of paths to images in which to search for stars.
    gaia_table (str)
        The GAIA data table to query.
    
    Returns
    -------
    tables (List[astropy.table.Table])
        List of Astropy tables containing the GAIA sources in each image.
    """

    # Select the appropriate GAIA table and return all rows.
    from astroquery.gaia import Gaia
    Gaia.MAIN_GAIA_TABLE = gaia_table
    Gaia.ROW_LIMIT = -1

    # For each image identified in the directory.
    tables = []
    for img_path in imgs:

        img, hdr = fits.getdata(img_path, header = True)
        ny, nx = img.shape

        wcs = WCS(hdr)

        # Convert pixel coordinates to RA and DEC.
        pixel_corners = np.array([[0, 0], [0, ny], [nx, ny], [nx, 0]])
        ra_dec_corners = wcs.pixel_to_world_values(pixel_corners[:, 0], pixel_corners[:, 1])

        # Find the minimum and maximum values of RA and DEC.
        ra_values = ra_dec_corners[0]
        dec_values = ra_dec_corners[1]

        min_ra = np.min(ra_values)
        max_ra = np.max(ra_values)
        min_dec = np.min(dec_values)
        max_dec = np.max(dec_values)

        # Calculate the width and height of the query region.
        width_deg = (max_ra - min_ra)
        height_deg = (max_dec - min_dec)

        # and create a SkyCoord object for its center.
        center_ra_deg = (min_ra + max_ra) / 2
        center_dec_deg = (min_dec + max_dec) / 2
        center_coord = SkyCoord(ra=center_ra_deg, dec=center_dec_deg, unit=u.deg, frame='icrs')

        # Query GAIA database.
        columns = ['source_id', 'ra', 'dec', 'phot_g_mean_mag', 'classprob_dsc_combmod_star',
                   'classprob_dsc_combmod_quasar', 'classprob_dsc_combmod_galaxy', 'parallax',
                   'parallax_error', 'parallax_over_error']        
        result_table = Gaia.query_object_async(center_coord, width=width_deg * u.deg,
                                               height=height_deg * u.deg, columns=columns)
        
        # Store the table for later.
        tables.append(result_table)

        # Plot the cutouts.
        gaia_cutouts(result_table, img, wcs)

    return tables

def gaia_catalogue(tables, spurious=None, outname="gaia_catalogue.fits", append=True):
    """
    Merge GAIA catalogues and remove spurious sources.
    
    Arguments
    ---------
    tables (List[astropy.table.Table])
        List of Astropy tables containing the GAIA sources.
    spurious (List[int])
        List of spurious source IDs.
    append (bool)
        If outname already exists, should the new catalogue be appended.
        If False, overwrite the existing file.
    outname (str)
        Name of the merged GAIA catalogue.
    """

    if spurious is None:
        spurious = []

    # Stack the GAIA catalogues.
    gaia_data = vstack(tables)

    # Convert the spurious IDs to the correct dtype.
    spurious = np.array(spurious, dtype=gaia_data['SOURCE_ID'].dtype)

    # Remove the spurious sources.
    mask = np.isin(gaia_data['SOURCE_ID'], spurious)
    gaia_data_ = gaia_data[~mask]

    # If file already exists append the new one if requested.
    if (os.path.exists(outname)) & (append == True):
        existing_table = Table.read(outname)
        gaia_data_ = vstack([existing_table, gaia_data_])
    
    # Write out the catalogue.
    gaia_data_.write(outname, overwrite=True)

    return