import os
import copy
import h5py
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

import emcee

from photutils.utils import ImageDepth

import sep

from crest.utils import measure_curve_of_growth

import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)


class SEP():
    """
    Class for running SEP in dual or single image mode and performing
    Kron or circular aperture photometry.
    """

    def __init__(self, config_file):
        """
        __init__ method for SEP.

        Arguments
        ---------
        config_file (str)
            Path to ".yml" configuration file.
        """
        # Store the configuration file path
        self.configfile = config_file

        # and the content.
        with open(self.configfile, 'r') as file:
            self.config = next(yaml.safe_load_all(file))

        # These are the available outputs.
        self.output_names = ['thresh', 'npix', 'tnpix', 'xmin', 'xmax', 'ymin', 'ymax', 'x', 'y',
                             'x2', 'y2', 'xy', 'errx2', 'erry2', 'errxy', 'a', 'b', 'theta', 'cxx',
                             'cyy', 'cxy', 'cflux', 'flux', 'cpeak', 'peak', 'xcpeak', 'ycpeak',
                             'xpeak', 'ypeak', 'flag', 'ellipse_flag', 'RA', 'DEC', 'FLUX_AUTO', 
                             'FLUXERR_AUTO', 'FLUX_FLAG']
        
        # May need this later.
        self._cat_name = None

    def _update_config(self, parameters):        
        """
        Copy and update the stored config with parameters provided 
        at runtime.

        Arguments
        ---------
        parameters (dict)
            Key-value pairs of parameters to update.
            
        Returns
        -------
        new_config (dict)
            Updated copy of the config file.
        att_config (dict)
            Config with values appropriate for saving to hdf5.
        """

        # Copy the stored parameter file.
        new_config = copy.deepcopy(self.config)

        # Update with the given parameters.
        new_config.update(parameters)

         # Store the config as is for saving as hdf5 attributes.
        att_config = copy.deepcopy(new_config)

        # Expand any environment variables and convert string to None.
        for key, value in new_config.items():
            if type(value) == str:
                new_config[key] = os.path.expandvars(value)
            if value == 'None':
                new_config[key] = None

        return new_config, att_config

    def _measure_background(self, sci, err, config):
        """
        Measure the background of an image using SEP functionality.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D array from which to measure the background.
        err (numpy.ndarray)
            2D array matching the shape of sci, containing the 
            corresponding error values.
        config (dict)
            Dictionary of background configration arguments.

        Returns
        -------
        bkg (sep.Background)
            The measured SEP background object.
        """

        print('Estimating background.')

        # Load a source mask if provided.
        mask = np.isnan(sci)
        if config['background_mask'] != None:
            mask = mask + fits.getdata(config['background_mask'])

        # Also mask off detector regions if we can.
        if isinstance(err, type(None)) == False:
            mask = mask + (err <= 0) + np.isnan(err)

        # Measure the background.
        bkg = sep.Background(sci, mask, 0, config['bw'], config['bh'], config['fw'], 
                             config['fh'], config['fthresh'])

        return bkg
                
    def _detect_sources(self, sci, err, segmap, config):
        """
        Detect and deblend sources and create an initial catalogue.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D array from which to measure the background.
        err (numpy.ndarray)
            2D array matching the shape of sci, containing the 
            corresponding error values.
        segmap (bool, numpy.ndarray)
            The segmap argument to pass to sep.extract. If bool, should
            the segmap be saved. If numpy.ndarray, the 2D segmentation
            map determined from the detection image.
        config (dict)
            Dictionary of background configration arguments.

        Return
        ------
        cat (astropy.table.Table)
            Astropy table storing information on detected sources.
        segmap (numpy.ndarray)
            2D image indicating the locations of detected sources.
        """

        # Mask off detector regions.
        mask = ((err <= 0) | np.isnan(err) | np.isnan(sci) | 
                (~np.isfinite(err)) | (~np.isfinite(sci)))

        # Generate kernel based on provided FWHM and size.
        kernel_map = {
            'Gaussian': Gaussian2DKernel(x_stddev=config['FWHM'] * gaussian_fwhm_to_sigma,
                                         y_stddev=config['FWHM'] * gaussian_fwhm_to_sigma,
                                         x_size=config['SIZE'], y_size=config['SIZE']).array,
            'Tophat': Tophat2DKernel(config['FWHM'] / np.sqrt(2), x_size=config['SIZE'], 
                                     y_size=config['SIZE']).array
        }
        kernel = kernel_map.get(config['FILTER'], None)

        # Set some memory limits.
        sep.set_extract_pixstack(config['pixstack'])
        sep.set_sub_object_limit(config['object_limit'])

        # Do the extraction.
        objects, segmap = sep.extract(
            sci, config['thresh'], err=err, gain=config['gain'], mask=mask, maskthresh=0, 
            minarea=config['minarea'], filter_kernel=kernel, filter_type=config['filter_type'], 
            deblend_nthresh=config['deblend_nthresh'], deblend_cont=config['deblend_cont'], 
            clean=config['clean'], clean_param=config['clean_param'], segmentation_map=segmap)
        
        cat = Table(objects)
    
        # Extract can produce theta values > pi/2, so we need to 
        # correct these before performing photometry.
        cat['theta'][cat['theta'] > np.pi / 2] -= np.pi
        cat['theta'][cat['theta'] < -np.pi / 2] += np.pi

        return cat, segmap
    
    def _measure_photometry(self, sci, err, segmap, cat, config, type='kron', radius=0):
        """
        Measure the photometry of detected sources using Kron apertures.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D science image from which to identify sources.
        error (None, str)
            The path to the map for detection weighting.
        segmap (numpy.ndarray)
            2D image indicating the locations of detected sources.
        cat (astropy.table.Table)
            Astropy table storing information on detected sources.
        config (dict)
            Dictionary of source detection configuration arguments.
        type (str)
            The type of photometry to generate, either 'kron' or 
            'circular'.
        radius (float)
            The radius of the circular aperture in pixels. Only used when
            type = 'circular'.

        Return
        ------
        flux (numpy.ndarray)
            The flux of the detected sources in image counts.
        fluxerr (numpy.ndarray)
            The corresponding flux error.
        flag (numpy.ndarray)
            Flag indicating the quality of the measured photometry.
        ap_radius (float/numpy.ndarray)
            The radius of the circular aperture used or the radius of the
            Kron aperture used for each object.
        """

        # Mask off detector regions.
        mask = ((err <= 0) | np.isnan(err) | np.isnan(sci) | 
                (~np.isfinite(err)) | (~np.isfinite(sci)))

        # The type of nearby pixel masking.

        # Do not mask any nearby pixels.
        if (config['mask_type'] == None) or (config['mask_type'] == 'NONE'):
            seg_id = None
            seg = None
        # Mask pixels belonging to other souces.
        elif config['mask_type'] == 'BLANK':
            seg_id = np.arange(1, len(cat)+1, dtype=np.int32)
            seg = segmap
        # Mask all pixels not identified as part of the segment.
        elif config['mask_type'] == 'SEGMENT':
            seg = segmap
            seg_id = np.arange(1, len(cat)+1, dtype=np.int32) * -1
        else:
            raise ValueError(f"mask_type {config['mask_type']} not recognised.")
        
        # Calculate the kron flux.
        if type == 'kron':
            # First get the kron radius.
            ap_radius, krflag = sep.kron_radius(
                sci, cat['x'], cat['y'], cat['a'], cat['b'], cat['theta'], config['int_radius'], 
                mask=mask, maskthresh=0, seg_id=seg_id, segmap=seg)
            
            # Then measure the flux in an elliptical aperture.
            flux, fluxerr, flag = sep.sum_ellipse(
                sci, cat['x'], cat['y'], cat['a'], cat['b'], cat['theta'], 
                config['kron_factor']*ap_radius, err=err, mask=mask, maskthresh=0, seg_id=seg_id, 
                segmap=seg, gain=config['gain'], subpix=config['subpix'])
            
            # Combine the Kron radius and ellipse flags.
            flag += krflag
        
        # Measure circular aperture photometry.
        elif type == 'circular':            
            ap_radius = radius
            flux, fluxerr, flag = sep.sum_circle(
                sci, cat['x'], cat['y'], ap_radius, err=err, mask=mask, maskthresh=0, seg_id=seg_id,
                  segmap=seg, gain=config['gain'], subpix=config['subpix'])
        
        return flux, fluxerr, flag, ap_radius
    
    def _get_aperture_locations(self, sci, mask, radius, napers, overlap=False, 
                                overlap_maxiters=50000):
        """
        Place random apertures in unmasked regions of an image and return
        their centres.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D science image in which to place the apertures.
        mask (numpy.ndarray)
            The 2D science image source mask.
        radius (float)
            The radius in pixels of the apertures to place.
        napers (int)
            The maximum number of apertures to place.
        overlap (bool)
            Should the apertures be allowed to overlap.
        overlap_maxiters (int)
            The number of attempts at placing a non-overlapping aperture.
        
        Returns
        -------
        x (List[float])
            The x-coordinate of the aperture centres.
        y (List[float])
            The y-coordinate of the aperture centres.
        """
        
        # Get the random aperture locations.
        depth = ImageDepth(radius, nsigma=1.0, napers=napers, niters=1, overlap=overlap,
                           overlap_maxiters=overlap_maxiters)
        limits = depth(sci, mask)
        print(f' Placed {int(depth.napers_used)} apertures.')

        # Get the location of the apertures.
        locations = depth.apertures[0].positions

        # Extract x-y coordinates from apertures.
        x = []
        y = []
        for i in np.round(locations).astype(int):
            x.append(i[0])
            y.append(i[1])

        return x, y
    
    def measure_depth(self, science, psf, mask=None, error=None, parameters=None, radius=3.33, 
                      max_apers=50, max_iters=50000):
        """
        Use randomly placed apertures to measure the average 
        5-sigma depth of an image.
        
        Arguments
        ---------
        science (str)
            Filename of science fits image.
        psf (str)
            Filename of the PSF fits image used to scale the aperture 
            depths to total.
        mask (None, str)
            Filename of the fits image mask. If None, generate and use
            a SE segmentation map.
        error (None, str)
            Filename of fits error map. If None, no weighting will be 
            used if generating a mask and only NaN non-source pixels will
            be masked.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        radius (float)
            Radius of the random apertures to use in pixels.
        max_apers (int)
            The maximum number of apertures to place.
        max_iters (int)
            The maximun attempts at finding a non overlapping location.
        
        Returns
        -------
        depth (float)
            The 5-sigma depth of the image.
        """

        print(f'Measuring 5-sigma depth of {os.path.basename(science)}.')

        if parameters is None:
            parameters = {}

        # Update the config file.
        depth_config, _ = self._update_config(parameters)
        depth_config['background_sub'] = False

        # Open the science image.
        sci, hdr = fits.getdata(science, header=True)
        sci = sci.byteswap(inplace=True).newbyteorder()

        # Load RMS map if available or use background RMS.
        if isinstance(error, type(None)):
            bkg = self._measure_background(sci, None, depth_config)
            err = bkg.rms()
        else:
            err = fits.getdata(error)
            err = err.byteswap(inplace=True).newbyteorder()
        
        # Has a source mask been provided?
        if isinstance(mask, str):
            source_mask = fits.getdata(mask)

        # If not, generate it.
        else:
            print('Generating source mask.')
            _, source_mask = self._detect_sources(sci, err, True, depth_config)
        
        # Construct the full source and coverage mask.
        full_mask = (source_mask != 0) | np.isnan(sci) | np.isnan(err) | (err <= 0)

        print('Placing random apertures...')
        x, y = self._get_aperture_locations(sci, full_mask, radius, max_apers, False, max_iters)
        cat = {'x':x, 'y':y}

        depth_config['mask_type'] = 'NONE'
        flux, _, _, _ = self._measure_photometry(sci, err, None, cat, depth_config, 
                                            'circular', radius)
        flux *= depth_config['flux_conversion']

        # Measure the median absolute deviation.
        s = (flux != 0) & (np.isfinite(flux))
        mad = median_abs_deviation(flux[s], nan_policy='omit', scale='normal')

        # Measure the PSF curve of growth and interpolate.
        psf_ = fits.getdata(psf)
        radii = np.arange(0.1, psf_.shape[0], 1)
        radii, cog, p = measure_curve_of_growth(psf_, radii=radii, position=None, 
                                                norm=False, show=False)
        f = lambda r: np.interp(r, radii, cog)

        # Correct by the fraction of the PSF enclosed within the 
        # aperture used and convert to 5 sigma.
        depth = 5*mad/f(radius)

        print('Depth calculation completed! \n')

        return depth
            
    def _empirical_uncertainty(self, sci, err, seg, cat, config):
        """
        Perform empirical uncertainty estimation by fitting the relation
        between aperture size and noise. Based on Finkelstein+23.

        Arguments
        ---------
        sci (numpy.ndarray)
            2D science image.
        err (numpy.ndarray)
            RMS map matching the shape of sci.
        seg (numpy.ndarray)
            Segmentation map measured from sci.
        cat (astropy.table.table.Table)
            Detection catalogue generated by SEP from sci.
        config (dict)
            Dictionary of SEP configuration parameters.
        
        Returns
        -------
        cat (astropy.table.table.Table)
            SEP detection catalogue updated with empirical uncertainties.
        """

        print('\nBeginning uncertainty estimation:')

        # Make a local copy of the config.
        err_config = copy.deepcopy(config)
        err_config['mask_type'] = 'NONE'

        # Mask off detector regions and sources.
        mask = (err <= 0) + np.isnan(err) + np.isnan(sci)

        # Get the aperture radii.
        if err_config['RADII_SPACING'] == 'linear':
            radii = np.linspace(err_config['MIN_RADIUS'], err_config['MAX_RADIUS'], 
                                err_config['N_RADII'])
        else:
            radii = np.logspace(np.log10(err_config['MIN_RADIUS']),
                                np.log10(err_config['MAX_RADIUS']), err_config['N_RADII'])
        
        # Seperate the radii into small and large components. This way we
        # only need to run SEP twice.
        smaller = radii < np.median(radii)
        larger = radii >= np.median(radii)

        app_runs = {'small':smaller, 'large':larger}
        medians = []
        for run, s in app_runs.items():
            print(f' Placing {run} apertures...')

            # Get the random locations for the apertures.
            x, y = self._get_aperture_locations(
                sci, mask+(seg!=0), max(radii[s]), err_config[f'N_{run.upper()}'], False, 
                err_config['MAX_ITERS'])
            ap_cat = {'x':x, 'y':y}

            # Measure the median flux in each aperture size.
            for r in radii[s]:
                flux, _, _, _ = self._measure_photometry(sci, err, seg, ap_cat, err_config, 
                                                         'circular', round(r,2))
                s_ = flux != 0
                medians.append(median_abs_deviation(flux[s_], nan_policy='omit', scale='normal'))

        # Defining the model to fit. 
        sig1 = sigma_clipped_stats(sci, mask+(seg!=0))[2]   
        Npix = np.pi * (radii**2)    
        def model(theta, Npix=Npix):
            a, b = theta
            return sig1 * a * (Npix**b)
        
        # Using a chi2 log-likelihood function.
        def lnlike(theta, x, y, yerr):
            return -0.5 * np.sum(((y - model(theta, x)) / yerr)** 2)
        
        # Setting allowed ranges for the free parameters.
        def lnprior(theta):
            a, b = theta
            if -1e9 < a < 1e9 and -1e9 < b < 1e9:
                return 0.0
            return -np.inf
        
        # Set up the MCMC.
        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)
    
        # The percentage error to use when fitting. 
        Merr = err_config['P_ERR']*np.array(medians)

        # Collect the x,y and error data.
        data = (Npix, medians, Merr)

        # Set the step methodology.
        initial = np.array(err_config['INITIAL'])
        p0 = [initial + 1e-7 * np.random.randn(len(initial)) for i in range(err_config['WALKERS'])] 
        
        # Begin the MCMC
        sampler = emcee.EnsembleSampler(err_config['WALKERS'], len(initial), lnprob, args = data)

        print(' Running MCMC...')
        p0, _, _ = sampler.run_mcmc(p0, err_config['BURN_IN'])
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(p0, err_config['N_ITERS'])

        # Get most likely parameter values.
        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]
        print(f' Most likely parameter values: {theta_max}')

        # Median error value of the whole map. Will use this to scale 
        # the errors.
        median_err = np.median(err[~mask])

        # We now want the radii of the apertures used for photometry.
        radii = err_config['radii']

        # Expecting a few NaNs so quiet any warnings.
        with np.errstate(invalid='ignore'):

            # Will scale errors by this relative value.
            rel_e = err[cat['y'].astype(int)-1, cat['x'].astype(int)-1] / median_err

            # For each flux column, calculate the area based on the type
            # of aperture and extract the noise from the fit.
            for column in cat.colnames:

                if column == 'FLUX_AUTO':
                    area = np.pi * cat['a'] * cat['b'] * np.power(cat['KRON_RADIUS'] * 
                                                                  err_config['kron_factor'], 2)
                    cat['FLUXERR_AUTO_EMPIRICAL'] = (model(theta_max, area) * rel_e * 
                                                     err_config['flux_conversion'])

                    usec = (cat['KRON_RADIUS'] * err_config['kron_factor'] * 
                            np.sqrt(cat['a'] * cat['b']) < err_config['min_radius'])
                    area = np.pi * np.power(err_config['min_radius'], 2)
                    cat['FLUXERR_AUTO_EMPIRICAL'][usec] = (model(theta_max, area) * rel_e[usec] * 
                                                           err_config['flux_conversion'])

                if 'FLUX_APER_' in column:
                    aper = int(column.split('FLUX_APER_')[1])
                    area = np.pi * np.power(radii[aper], 2)
                    cat[f'FLUXERR_APER_{aper}_EMPIRICAL'] = (model(theta_max, area) * rel_e * 
                                                             err_config['flux_conversion'])

        # Save a plot of noise vs aperture size.
        if err_config['SAVE_FIG'] == True:

            x = np.linspace(0, max(Npix), 10000)
            fig = plt.figure()
            ax = plt.gca()
            plt.scatter(np.sqrt(Npix), medians,s = 15, color = 'white', edgecolors = 'blue',
                        alpha = 0.8)
            plt.plot(np.sqrt(x), model(theta_max, x), color = 'grey', linestyle = '--',
                     linewidth = 1)  
            title = os.path.basename(self._cat_name).removesuffix('.hdf5')
            plt.title(title, fontsize = 10)
            plt.xlabel('sqrt(Number of pixels in aperture)')
            plt.ylabel('Noise in aperture [counts]')
            plt.minorticks_on()
            ax.tick_params(axis = 'both', direction = 'in', which = 'both')
            plt.savefig(self._cat_name.replace('.hdf5', '_noise.png'))
            plt.close()

        print(' Empirical errors calculated! \n')

        return cat

    def extract(self, science, error=None, parameters=None, outputs=None, cat_name=None, outdir='./'):
        """
        Main function for extracting sources and measuring photometry 
        in a science image.
        
        Arguments
        ---------
        science (str, List[str])
            If string, path to single image from which to detect and 
            measure sources. If List[str], path to detection image as the 
            first entry and measurement as the second.
        error (None, str, List[str])
            The corresponding error images for detection.
            If None, use global background RMS.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        outputs (None, List[str])
            The source extraction outputs to save to the catalogue.
            If None, save all available.
        cat_name (None, str)
            The base name for the photometry catalogue. If None, use the 
            base name of the measurement file.
        outdir (str)
            The directory in which to store output files.

        Returns
        -------
        cat_name (str)
            The filepath of the generated hdf5 catalogue.
        """

        # All files will be output here.
        if os.path.isdir(outdir) == False:
            raise NotADirectoryError(f'{outdir} is not a directory. Set "outdir" to an existing'
                                        ' directory.')

        if parameters is None:
            parameters = {}

        # Update the config file with the given parameters.
        config, att_config = self._update_config(parameters)

        # Are we in double image mode?
        single_mode = False
        if isinstance(science, list):
            if len(science) == 2:
                print('Starting extraction in double image mode. \n')
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                    'form [detection, measurement].') 
             
            # Have errors been provided.
            if isinstance(error, list):
                if len(error) != 2:
                    raise ValueError('Double image mode requires a list of weight paths of the '
                                    'form [detection, measurement].')
                
                # If not, warn the user that the background RMS will be
                # used instead.
                s = [i == None for i in error]  
                if sum(s) == 2:
                    print('No RMS maps provided. Will use measured background RMS for weighting. \n')
                elif sum(s) == 1:
                    print(f'No RMS map provided for {np.array(["detection", "measurement"])[s][0]}.'
                          ' Will use measured background RMS for weighting. \n')
            elif isinstance(error, type(None)):
                print('No RMS maps provided. Will use measured background RMS for weighting. \n')
                error = [error] * 2
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                                'form [detection, measurement], or None for no weighting.') 
            
        # If not, we should be in single image mode.
        elif isinstance(science, str):
            print('Starting extraction in single image mode. \n')
            single_mode = True
            
            if isinstance(error, type(None)):
                print('No RMS map provided. Will use measured background RMS for weighting. \n')
            elif isinstance(error, str) == False:
                raise ValueError('Single image mode requires a string path to a weight map or None '
                               'for no weighting.') 
            
            # Duplicate the inputs to match double image format.
            science = [science] * 2
            error = [error] * 2

        # If we get here, the inputs are very wrong.   
        else:
            raise ValueError('Image inputs are not the correct format. Use strings for single image'
                           ' mode and lists of the form [detection, measurement] for double. '
                           'Use None for no weighting.')
        
        # Name the catalogue after the measurement image.
        if isinstance(cat_name, type(None)):
            cat_name = f'{outdir}/{os.path.basename(science[1]).removesuffix(".fits")}_sep.hdf5'
        else:
            cat_name = f'{outdir}/{os.path.basename(cat_name)}.hdf5'
        self._cat_name = cat_name

        # Load detection image.
        if not single_mode: print(f'Processing {os.path.basename(science[0])}:')
        sci_d, hdr_d = fits.getdata(science[0], header=True)
        sci_d = sci_d.byteswap(inplace=True).newbyteorder()

        # Load RMS map if available.
        if isinstance(error[0], type(None)) == False:
            err_d = fits.getdata(error[0])
            err_d = err_d.byteswap(inplace=True).newbyteorder()
        else:
            err_d = None

        # Measure the background if needed.
        if isinstance(err_d, type(None)) or config['background_sub']:
            bkg = self._measure_background(sci_d, err_d, config)
            if isinstance(err_d, type(None)):
                err_d = bkg.rms()
            if config['background_sub']:
                print('Subtracting the background.')
                bkg.subfrom(sci_d)

        # Create a segmentation map and an initial catalogue.
        print('Detecting sources.')
        cat, segmap = self._detect_sources(sci_d, err_d, True, config)

        # Save the segmentation map if requested.
        if config['segmap_name'] != None:
            print(f'Saving segmentation map to {outdir}/{config["segmap_name"]}.fits')
            fits.writeto(f'{outdir}/{config["segmap_name"]}.fits', segmap, hdr_d, overwrite=True)

        # If in single image mode, measurement is detection.
        if single_mode:
            sci_m, err_m = (sci_d, err_d)
        
        # Otherwise need to load and process the measurement images.
        else:
            print(f'Processing {os.path.basename(science[1])}:')
            sci_m = fits.getdata(science[1])
            sci_m = sci_m.byteswap(inplace=True).newbyteorder()

            # Load RMS map if available.
            if isinstance(error[1], type(None)) == False:
                err_m = fits.getdata(error[1])
                err_m = err_m.byteswap(inplace=True).newbyteorder()
            else:
                err_m = None

            # Measure the background if needed.
            if isinstance(err_m, type(None)) or config['background_sub']:
                bkg = self._measure_background(sci_m, err_m, config)
                if isinstance(err_m, type(None)):
                    err_m = bkg.rms()
                if config['background_sub']:
                    print('Subtracting the background.')
                    bkg.subfrom(sci_m)

            # Rerun detect_sources with the previously measured segmap to
            # get a new catalogue.
            cat, _ = self._detect_sources(sci_m, err_m, segmap, config)

        # Can't perform aperture photometry if a or b couldn't be 
        # measured. Will set these to zero and flag.
        s_a = ~np.isfinite(cat['a'])
        cat['a'][s_a] = 0

        s_b = ~np.isfinite(cat['b'])
        cat['b'][s_b] = 0

        cat['ellipse_flag'] = s_a + s_b

        # Calculate RA and DEC.
        wcs = WCS(hdr_d)
        coordinates = pixel_to_skycoord(cat['x'], cat['y'], wcs)
        cat['RA'] = coordinates.ra.degree
        cat['DEC'] = coordinates.dec.degree

        # Measure the photometry in Kron apertures.
        print('Measuring Kron photometry.')
        kflux, kfluxerr, kflag, kron = self._measure_photometry(sci_m, err_m, segmap, cat, config, 
                                                            'kron')

        # Also measure in circular apertures, with the minimum Kron 
        # radius defined in the config.
        r_min = config['min_radius']
        cflux, cfluxerr, cflag, _ = self._measure_photometry(sci_m, err_m, segmap, cat, config, 
                                                            'circular', r_min)

        # Only use this photometry when kron radius is less than minimum.
        use_circle = kron * np.sqrt(cat['a'] * cat['b']) < r_min

        # Replace Kron flux measurements with circular ones.
        kflux[use_circle] = cflux[use_circle]
        kfluxerr[use_circle] = cfluxerr[use_circle]
        kflag[use_circle] = cflag[use_circle]

        # Convert to desired flux unit and add to catalogue.
        cat['FLUX_AUTO'] = kflux * config['flux_conversion']
        cat ['FLUXERR_AUTO'] = kfluxerr * config['flux_conversion']
        cat['FLUX_FLAG'] = kflag
        cat['KRON_RADIUS'] = kron

        # Also measure flux in user defined circular apertures.
        for idx, radius in enumerate(config['radii']):
            if idx == 0: print('Measuring aperture photometry.')
            flux, fluxerr, _, _ = self._measure_photometry(sci_m, err_m, segmap, cat, config, 
                                                          'circular', radius)
            cat[f'FLUX_APER_{idx}'] = flux * config['flux_conversion']
            cat[f'FLUXERR_APER_{idx}'] = fluxerr * config['flux_conversion']

        # Perfrom empirical uncertaninty estimation.
        if config['EMPIRICAL']:
            self._empirical_uncertainty(sci_m, err_m, segmap, cat, config)

        # Need to convert other quantities to chosen flux unit.
        flux_columns = ['cflux', 'flux', 'cpeak', 'peak']
        for name in flux_columns:
            cat[name] = cat[name] * config['flux_conversion']

        # Now add everything to the hdf5 catalogue.
        with h5py.File(cat_name, 'w') as f:

            # Add contents to a "photometry" group.
            f.create_group('photometry')

            # If no outputs requested, use all.
            if outputs == None:
                outputs = cat.colnames
            outputs = set(outputs)
                        
            # Add the outputs to the hdf5 catalogue.
            for output in outputs:
                if output in cat.colnames:
                    f[f'photometry/{output}'] = cat[output]
                else:
                    print(f'Skipping {output} as it is not a recognised output quantity. ' 
                          'Check SEP.output_names for available outputs.')

            # Add the config parameters as attributes.
            for key,value in att_config.items():
                f['photometry'].attrs[key] = value
            f['photometry'].attrs['CODE'] = 'SEP'
            f['photometry'].attrs['VERSION'] = sep.__version__
            
        print(f'Completed extraction and saved to {cat_name} \n')

        return cat_name
