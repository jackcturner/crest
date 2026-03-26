import h5py
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from ned_extinction_calc import request_extinctions

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

def correct_extinction(catalogue, flux_err_pairs, to_jy, replace=False, suffix='_EXT',
                       ra_key='ALPHA_SKY', dec_key='DELTA_SKY'):
    """
    Query the NED extinction calculator using mean RA and DEC location
    and apply correction to FLAGS catalogue.
    
    Arguments
    ---------
    catalogue (str)
        Path to FLAGS hdf5 catalogue.
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

    # Translate commonly used filters to the closest match on NED.
    translate = {'JWST/NIRCam.F070W': 'WFPC2 F702W','JWST/NIRCam.F090W': 'ACS F850LP','JWST/NIRCam.F115W': 'WFC3 F110W',
                    'JWST/NIRCam.F150W': 'WFC3 F160W','JWST/NIRCam.F200W': 'WFC3 F160W','JWST/NIRCam.F140M': 'WFC3 F140W',
                    'JWST/NIRCam.F162M': 'UKIRT H','JWST/NIRCam.F182M': 'UKIRT H','JWST/NIRCam.F210M': 'UKIRT K',
                    'JWST/NIRCam.F277W': 'UKIRT K','JWST/NIRCam.F356W': "UKIRT L",'JWST/NIRCam.F444W': "UKIRT L",
                    'JWST/NIRCam.F250M': 'UKIRT K','JWST/NIRCam.F300M': "UKIRT L",'JWST/NIRCam.F335M': "UKIRT L",
                    'JWST/NIRCam.F360M': "UKIRT L",'JWST/NIRCam.F410M': "UKIRT L",'JWST/NIRCam.F430M': "UKIRT L",
                    'JWST/NIRCam.F460M': "UKIRT L",'JWST/NIRCam.F480M': "UKIRT L",'HST/ACS_WFC.F435W': 'ACS F435W',
                    'HST/ACS_WFC.F475W': 'ACS F475W','HST/ACS_WFC.F555W': 'ACS F555W','HST/ACS_WFC.F606W': 'ACS F606W',
                    'HST/ACS_WFC.F625W': 'ACS F625W','HST/ACS_WFC.F775W': 'ACS F775W','HST/ACS_WFC.F814W': 'ACS F814W',
                    'HST/WFC3_IR.F098M': 'LSST y','HST/WFC3_IR.F105W': 'WFC3 F105W','HST/WFC3_IR.F110W': 'WFC3 F110W',
                    'HST/WFC3_IR.F125W': 'WFC3 F125W','HST/WFC3_IR.F140W': 'WFC3 F140W','HST/WFC3_IR.F160W': 'WFC3 F160W',
                    'HST/ACS_WFC.F850LP': 'ACS F850LP'}

    # Read the catalogue.
    with h5py.File(catalogue, 'r+') as f:

        print(f'Correcting {catalogue} for extinction.')

        # Get list of instruments and associated filters.
        instruments = list(f['photometry'].keys())
        if 'DETECTION' in instruments:
            instruments.remove('DETECTION')

        filters = []
        for instrument in instruments:
            filters_ = list(f[f'photometry/{instrument}'].keys())
            for i in filters_:
                filters_[filters_.index(i)] = f'{instrument}/{i}'
            filters += filters_
        
        # Determine the closest matching dust map.
        for filter in filters:
            corr_filt = translate[filter]

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