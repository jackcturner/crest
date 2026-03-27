import os
import copy
import h5py
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, convolve_fft
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

import emcee

import photutils
import photutils.background as pb
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog
from photutils.utils import ImageDepth

from crest.utils import measure_curve_of_growth
from crest.utils import _parallel_execute, _tile_worker, _construct_tiles

class Photutils():
    """
    Wrap Photutils, allowing it to be used in a manner consistent with
    Source Extractor and produce hdf5 catalogues.
    """

    def __init__(self, config_file, verbose=True):
        """
        __init__ method for Photutils.

        Arguments
        ---------
        config_file (str)
            Path to YAML configuration file.
        verbose (bool)
            If True, print progress messages.
        """

        # Store the configuration file path
        self.configfile = config_file

        # and the content.
        with open(self.configfile, 'r') as file:
            self.config = next(yaml.safe_load_all(file))

        self.verbose = verbose

        # List of outputs produced by SourceCatalogue.
        self.output_names = [
            'area', 'background_centroid', 'background_mean', 'background_sum', 'bbox_xmax',
              'bbox_xmin', 'bbox_ymax', 'bbox_ymin', 'centroid','centroid_quad', 'centroid_win', 
              'covar_sigx2', 'covar_sigy2', 'covariance', 'covariance_eigvals', 'cutout_centroid', 
              'cutout_centroid_quad', 'cutout_centroid_win', 'cutout_maxval_index', 
              'cutout_minval_index', 'cxx', 'cxy', 'cyy', 'eccentricity', 'ellipticity', 
              'elongation', 'equivalent_radius', 'fwhm', 'gini', 'inertia_tensor', 
              'kron_flux', 'kron_fluxerr', 'kron_radius', 'label', 'labels', 
              'local_background', 'max_value', 'maxval_index', 
              'maxval_xindex', 'maxval_yindex', 'min_value', 'minval_index', 'minval_xindex', 
              'minval_yindex', 'moments', 'moments_central', 'orientation', 'perimeter',
              'segment_area', 'segment_flux', 'segment_fluxerr', 'semimajor_sigma', 
              'semiminor_sigma', 'sky_bbox_ll', 'sky_bbox_lr', 'sky_bbox_ul', 'sky_bbox_ur', 
              'sky_centroid', 'sky_centroid_icrs', 'sky_centroid_quad', 'sky_centroid_win', 
              'xcentroid', 'xcentroid_quad', 'xcentroid_win', 'ycentroid', 
              'ycentroid_quad', 'ycentroid_win', 'RA', 'DEC', 'background', 'convdata', 'data',
              'error', 'segment']
        
        # May need this later.
        self._cat_name = None

    def _vprint(self, *args, **kwargs):
        """
        Print only when verbose output is enabled.
        """

        if self.verbose:
            print(*args, **kwargs)

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

        # Update with the given parameters.
        new_config = copy.deepcopy(self.config)
        new_config.update(parameters)

         # Store as is for saving as hdf5 attributes.
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
        Measure and remove the background from a science image.

        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D array from which to measure the background.
        err (numpy.ndarray)
            The corresponding error map.
        config (dict)
            Dictionary of background configration arguments.

        Returns
        -------
        bkg (photutils.background.Background2D)
            Photutils background object measured from the science image.
        """

        # Translate interpolation, background and RMS estimators.
        interpolators = {'IDW':pb.BkgIDWInterpolator(), 'Zoom':pb.BkgZoomInterpolator()}
        back_est = {'Mean':pb.MeanBackground(), 'Median':pb.MedianBackground(), 
                    'Mode':pb.ModeEstimatorBackground(),'MMM':pb.MMMBackground(),
                    'SExtractor':pb.SExtractorBackground(),
                    'BiweightLocation':pb.BiweightLocationBackground()}
        rms_est = {'Std':pb.StdBackgroundRMS(), 'MADStd':pb.MADStdBackgroundRMS(), 
                   'BiweightScale':pb.BiweightScaleBackgroundRMS()}
        
        # Set up the coverage and source mask.
        coverage_mask = ~np.isfinite(sci)
        if isinstance(err, type(None)) == False:
            coverage_mask += (err <= 0) + (~np.isfinite(err))
        
        mask = None
        if config['SOURCE_MASK'] != None:
            mask = fits.getdata(config['SOURCE_MASK'])

        # Construct the sigma clipping object.
        sigma_clip = None
        if config['SIGMA_CLIP'] == True:
            sigma_clip = SigmaClip(sigma_lower=config['SIGMA'][0], sigma_upper=config['SIGMA'][1], 
                                   maxiters=config['MAX_ITERS'])

        # Get the interpolation, background and RMS estimators.
        bkg_estimator = back_est.get(config['BACK_ESTIMATOR'])
        bkgrms_estimator = rms_est.get(config['RMS_ESTIMATOR'])
        interpolator = interpolators.get(config['INTERPOLATOR'])

        # Calculate the 2D background.
        self._vprint('Measuring the 2D sky background...')
        bkg = pb.Background2D(
            sci, box_size=config['BOX_SIZE'], mask=mask, coverage_mask=coverage_mask, fill_value=0,
            exclude_percentile=config['EXCLUDE_PERCENTILE'], filter_size=config['FILTER_SIZE'],
            filter_threshold=config['FILTER_THRESH'], edge_method=config['EDGE_METHOD'], 
            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, bkgrms_estimator=bkgrms_estimator,
            interpolator=interpolator)
        
        return bkg
    
    def _filter(self, sci, err, bkg, config):
        """
        Filter an image using a Guassian or Tophat kernel.

        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D array from which to measure the background.
        err (numpy.ndarray)
            The corresponding error map.
        bkg (photutils.background.Background2D)
            Photutils background object measured from the science image.
        config (dict)
            Key value pairs defining the filtering parameters.

        Returns
        -------
        sci (numpy.ndarray)
            The filtered science image.
        """

        # Replace off detector regions with median background so 
        # convolution doesn't smear them.
        mask = (err <= 0) + (~np.isfinite(err)) + (~np.isfinite(sci))
        sci = np.where(mask == True, bkg.background_median, sci)

        # Generate kernel and convolve.
        kernel_map = {
            'Gaussian': Gaussian2DKernel(x_stddev=config['FWHM'] * gaussian_fwhm_to_sigma,
                                         y_stddev=config['FWHM'] * gaussian_fwhm_to_sigma,
                                         x_size=config['SIZE'], y_size=config['SIZE']).array,
            'Tophat': Tophat2DKernel(config['FWHM'] / np.sqrt(2), x_size=config['SIZE'], 
                                     y_size=config['SIZE']).array
        }
        kernel = kernel_map[config['FILTER']]

        # If no tiling requested, convolve the full image.
        n_tiles = config.get('N_TILES', 1)
        if n_tiles <= 1:
            sci = convolve_fft(sci, kernel, mask=mask, preserve_nan=True, allow_huge=True)

        else:

            # Otherwise, compute halo size.
            kh = int(np.ceil(max(kernel.shape) // 2))

            # Split into tiles.
            ny, nx = sci.shape
            convolved_sci = np.zeros_like(sci, dtype=np.float32)
            slices = _construct_tiles((ny, nx), int(n_tiles), kh)

            # Prepare tasks for each tile.
            tasks = []
            for s in slices:
                y0, y1, x0, x1, e0, e1, f0, f1 = s
                block = sci[e0:e1, f0:f1]
                mask_block = mask[e0:e1, f0:f1]
                tasks.append({'block': block, 'mask': mask_block,
                              'kernel': kernel, 'slices': s})

            # Execute in parallel and stitch tiles back together.   
            workers = max(int(config.get('N_WORKERS', 1)), 1)
            results = _parallel_execute(_tile_worker, tasks, workers)
            for res in results:
                y0, y1, x0, x1, interior = res
                convolved_sci[y0:y1, x0:x1] = interior

            sci = convolved_sci

        sci = np.where(mask == True, 0, sci)
                
        return sci
    
    def _segmentation(self, sci, err, config):
        """
        Segment and deblend sources in a sicence image.

        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D array from which to measure the background.
        err (numpy.ndarray)
            The corresponding error map.
        config (dict)
            Key value pairs defining the detection parameters.

        Returns
        -------
        seg_image (photutils.segmentation.SegmentationImage)
            A segmentation image, with the same shape as sci, where 
            sources are marked by different positive integer values. 
        """

        # Mask off detector regions.
        mask = (err <= 0) + (~np.isfinite(err)) + (~np.isfinite(sci))

        # Generate the segmentation image.
        self._vprint('Detecting sources...')
        seg_image = detect_sources(
            sci, threshold=config['N_SIGMA'] * err, npixels=config['N_PIXELS'],
            connectivity=config['CONNECTIVITY'], mask=mask)
        
        # and then deblend it.
        self._vprint('Deblending sources...')
        seg_image = deblend_sources(sci, seg_image, config['N_PIXELS'], nlevels=config['N_LEVELS'],
                                    contrast=config['CONTRAST'], mode=config['MODE'],
                                    connectivity=config['CONNECTIVITY'], relabel=True,
                                    nproc=1, progress_bar=False)
        
        return seg_image
    
    def _get_aperture_locations(self, sci, mask, radius, napers, overlap=False, 
                                overlap_maxiters=50000):
        """
        Place random apertures in unmasked regions of an image and create
        a detection image based on their centres.
        
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
            Should the apertures be allowed to overlap?
        overlap_maxiters (int)
            The number of attempts at placing a non-overlapping aperture.
        
        Returns
        -------
        det (numpy.ndarray)
            Array matching the shape of sci with value 1 at aperture 
            centres and zero otherwise.
        """
                        
        # Get the random aperture locations.
        depth = ImageDepth(radius, nsigma=1.0, napers=napers, niters=1, overlap=overlap,
                           overlap_maxiters=overlap_maxiters)
        limits = depth(sci, mask)
        self._vprint(f' Placed {int(depth.napers_used)} apertures.')

        locations = depth.apertures[0].positions

        # Construct the detection image.
        det = np.zeros(sci.shape)
        for i in np.round(locations).astype(int):
            det[i[1], i[0]] = 1

        return det
    
    def measure_depth(self, science_path, psf_path, mask_path=None, error_path=None, 
                      parameters=None, radius=3.33, max_apers=50, max_iters=50000):
        """
        Use randomly placed apertures to measure the average 
        5-sigma depth of an image.
        
        Arguments
        ---------
        science_path (str)
            Filename of science fits image.
        psf_path (str)
            Filename of the PSF fits image used to scale the aperture 
            depths to total.
        mask_path (None/str)
            Filename of the fits image mask. If None, generate and use
            a SEP segmentation map.
        error_path (None/str)
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

        self._vprint(f'Measuring 5-sigma depth of {os.path.basename(science_path)}.')

        if parameters is None:
            parameters = {}
        
        # Update the config file with given parameters.
        depth_config, _ = self._update_config(parameters)
        depth_config['BKG_SUB'] = False

        sci, hdr = fits.getdata(science_path, header=True)

        # Load RMS map if available.
        bkg = None
        if isinstance(error_path, type(None)):
            bkg = self._measure_background(sci, None, depth_config)
            err = bkg.background_rms
        else:
            err = fits.getdata(error_path)

        # Has a source mask been provided?
        if isinstance(mask_path, str):
            source_mask = fits.getdata(mask_path)

        # If not, generate it.
        else:
            self._vprint('Generating source mask.')

            # Filter the image if required.
            if depth_config['FILTER'] != None:

                # Measure background if we haven't already.
                if isinstance(bkg, type(None)):
                    bkg = self._measure_background(sci, err, depth_config)

                sci_filt = self._filter(sci, err, bkg, depth_config)
                source_mask = self._segmentation(sci_filt, err, depth_config).data

            else:
                source_mask = self._segmentation(sci, err, depth_config).data

        # Construct the full source and coverage mask.
        mask = (~np.isfinite(sci)) | (~np.isfinite(err)) | (err <= 0)
        full_mask = mask | (source_mask != 0)  

        # Get the random aperture locations and construct and image with
        # ones at these coordinates.
        self._vprint('Placing random apertures...')
        det = self._get_aperture_locations(sci, full_mask, radius, max_apers, False, max_iters) 

        # Perform aperture photometry.
        det_seg = detect_sources(det, threshold=1E-12, npixels=1, mask=mask)
        ap_cat = SourceCatalog(sci, det_seg, error=err, mask=mask)
        ap_cat.circular_photometry(radius, 'APER_0')

        # Calculate the Gaussian-like MAD of the fluxes.
        flux = getattr(ap_cat, f'APER_0_flux')*depth_config['CONVERSION']
        s = (flux != 0) & (np.isfinite(flux))
        mad = median_abs_deviation(flux[s], nan_policy='omit', scale='normal')

        # Measure the PSF curve of growth and interpolate.
        psf_ = fits.getdata(psf_path)
        radii = np.arange(0.1, psf_.shape[0], 1)
        radii, cog, p = measure_curve_of_growth(psf_, radii=radii, position=None, 
                                                norm=False, show=False)
        f = lambda r: np.interp(r, radii, cog)

        # Correct by the fraction of the PSF enclosed within the 
        # aperture used and convert to 5 sigma.
        depth = 5*mad/f(radius)

        self._vprint('Depth calculation completed! \n')

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
            The corresponding error map.
        seg (numpy.ndarray)
            2D image indicating the extent of detected sources.
        cat (photutils.segmentation.catalog.SourceCatalog)
            SourceCatalogue storing information on detected sources.
        config (dict)
            Dictionary of configuration parameters.
        
        Returns
        -------
        cat (photutils.segmentation.catalog.SourceCatalog)
            SourceCatalogue updated with empirical uncertainties.
        """

        self._vprint('\nBeginning uncertainty estimation:')

        # Make a local copy of the config.
        err_config = copy.deepcopy(config)
        err_config['FILTER'] = None
        err_config['N_SIGMA'] = 1E-12
        err_config['N_PIXELS'] = 1
        err_config['APERMASK_METHOD'] = None
        err_config['BKG_SUB'] = False 

        # Mask off detector regions.
        mask = (err <= 0) + (~np.isfinite(err)) + (~np.isfinite(sci))

        # Separate the radii into small and large components.        
        if err_config['RADII_SPACING'] == 'linear':
            radii = np.linspace(err_config['MIN_RADIUS'], err_config['MAX_RADIUS'], 
                                err_config['N_RADII'])
        else:
            radii = np.logspace(np.log10(err_config['MIN_RADIUS']),
                                np.log10(err_config['MAX_RADIUS']), err_config['N_RADII'])
        
        smaller = radii < np.median(radii)
        larger = radii >= np.median(radii)

        # For each component.
        app_runs = {'small':smaller, 'large':larger}
        medians = []
        for run, s in app_runs.items():

            # Measure the median flux in each aperture size.
            det = self._get_aperture_locations(
                sci, mask+(seg!=0), max(radii[s]), err_config[f'N_{run.upper()}'], False, 
                err_config['MAX_ITERS'])

            det_seg = detect_sources(det, threshold=1E-12, npixels=err_config['N_PIXELS'],
                                        connectivity=err_config['CONNECTIVITY'], mask = mask)
    
            ap_cat = SourceCatalog(
                sci, det_seg, convolved_data=None, error=err, mask=mask,
                background=None, wcs=None, localbkg_width=config['LOCALBKG_WIDTH'],
                apermask_method=config['APERMASK_METHOD'], kron_params=config['KRON_PARAMS'],
                detection_cat=None, progress_bar=False)
        
            for i, r in enumerate(radii[s]):
                ap_cat.circular_photometry(r, f'APER_{i}', overwrite=False)
                medians.append(median_abs_deviation(getattr(ap_cat, f'APER_{i}_flux'), 
                                                    nan_policy='omit', scale='normal'))

        # The noise model to fit. 
        sig1 = sigma_clipped_stats(sci, mask+(seg!=0))[2]   
        Npix = np.pi * (radii**2)    
        def model(theta, Npix=Npix):
            a, b = theta
            return sig1 * a * (Npix**b)
        
        # Define the likelihood function and priors.
        def lnlike(theta, x, y, yerr):
            return -0.5 * np.sum(((y - model(theta, x)) / yerr)** 2)
        
        def lnprior(theta):
            a, b = theta
            if -1e9 < a < 1e9 and -1e9 < b < 1e9:
                return 0.0
            return -np.inf
        
        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)
        
        # Set the step methodology.
        initial = np.array(err_config['INITIAL'])
        p0 = [initial + 1e-7 * np.random.randn(len(initial)) for i in range(err_config['WALKERS'])] 
    
        # The percentage error to use when fitting. 
        Merr = err_config['P_ERR']*np.array(medians)
        
        # Begin the MCMC
        sampler = emcee.EnsembleSampler(err_config['WALKERS'], len(initial), lnprob,
                                        args=(Npix, medians, Merr))

        self._vprint(' Running MCMC...')
        p0, _, _ = sampler.run_mcmc(p0, err_config['BURN_IN'])
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(p0, err_config['N_ITERS'])

        # Get most likely parameter values.
        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]
        self._vprint(f' Most likely parameter values: {theta_max}.')

        # Median error value of the whole map. 
        median_err = np.median(err[~mask])

        # Expecting a few NaNs so quiet any warnings.
        labels = []
        with np.errstate(invalid='ignore'):

            # Will scale errors by this relative value.
            rel_e = err[cat.ycentroid.astype(int), cat.xcentroid.astype(int)] / median_err

            # Scale Kron flux and minimum circular apertue.
            k_radius = (err_config['KRON_PARAMS'][0] * cat.kron_radius * 
                        np.sqrt(cat.semimajor_sigma * cat.semiminor_sigma))
            area = np.pi * np.power(k_radius, 2)
            area_min = np.pi * (err_config['KRON_PARAMS'][2]**2)

            kron_emprical = model(theta_max, area) * rel_e
            if area_min > 0:
                 kron_emprical = np.where(k_radius > err_config['KRON_PARAMS'][1], kron_emprical, 
                                          model(theta_max, area_min) * rel_e)
            else:
                kron_emprical = np.where(k_radius > err_config['KRON_PARAMS'][1], kron_emprical, 0)
                 
            cat.add_extra_property('kron_fluxerr_empirical', kron_emprical)
            labels.append('kron_fluxerr_empirical')

            # Segment flux,
            area = cat.segment_area
            cat.add_extra_property('segment_fluxerr_empirical',  model(theta_max, area) * rel_e)
            labels.append('segment_fluxerr_empirical')

            # and any aperture fluxes.
            for idx, radius in enumerate(err_config['RADII']):
                area = np.pi * np.power(radius, 2)
                cat.add_extra_property(f'APER_{idx}_fluxerr_empirical', 
                                       model(theta_max, area) * rel_e)
                labels.append(f'APER_{idx}_fluxerr_empirical')

        # Save a plot of noise vs aperture size.
        if err_config['SAVE_FIG'] == True:

            x = np.linspace(0, max(Npix), 10000)
            fig, ax  = plt.subplots(figsize=(3.78, 3.78))
            ax.scatter(np.sqrt(Npix), medians, s=15, color='white', edgecolors='blue',
                       alpha = 0.8)
            ax.plot(np.sqrt(x), model(theta_max, x), color='grey', linestyle='--',
                    linewidth=1)  
            title = os.path.basename(self._cat_name).removesuffix('.hdf5')
            ax.set_title(title, fontsize = 10)
            ax.set_xlabel('sqrt(Number of pixels in aperture)')
            ax.set_ylabel('Noise in aperture [counts]')
            plt.savefig(self._cat_name.replace('.hdf5', '_noise.png'))
            plt.close()

        return labels
    
    def extract(self, science_path, error_path, parameters=None, outputs=None, 
                cat_name=None, outdir='./'):
        """
        Main function for extracting sources and measuring photometry 
        in a science image.

        Arguments
        ---------
        science_path (str/List[str])
            If str, the filename of the image to extract.
            If a List[str] filename of detection and measurement images.
        error_path (None/str/List[str])
            If None, use global background RMS.
            If str, path to corresponding error map.
            If List[str], paths to error maps for detection 
            and measurement.
        parameters (dict/None)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        outputs (None/List[str])
            The source extraction outputs to save to the catalogue.
            If None, save all available.
        cat_name (None/str)
            The base name for the photometry catalogue. If None, use the 
            base name of the measurement file.
        outdir (str)
            The directory in which to store output files.

        Returns
        -------
        cat_name (str)
            Path to the hdf5 file containing the measured photometry.
        """

        if os.path.isdir(outdir) == False:
            raise NotADirectoryError(f'{outdir} is not a directory. Set "outdir" to an existing'
                                        ' directory.')

        # Update the config file with the provided parameters.
        if parameters is None:
            parameters = {}
        config, att_config = self._update_config(parameters)
 
        # Are we in double image mode?
        single_mode = False
        if isinstance(science_path, list):
            if len(science_path) == 2:
                self._vprint('Starting extraction in double image mode.')
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                                 'form [detection, measurement].') 
             
            # Have errors been provided.
            if isinstance(error_path, list):
                if len(error_path) != 2:
                    raise ValueError('Double image mode requires a list of weight paths of the '
                                     'form [detection, measurement].')
                
                # If not, warn the user that the background RMS will be 
                # used instead.
                s = [i == None for i in error_path]  
                if sum(s) == 2:
                    self._vprint('No RMS maps provided. Will use measured '
                                 'background RMS for weighting. \n')
                elif sum(s) == 1:
                    i = np.array(["detection", "measurement"])[s][0]
                    self._vprint(f'No RMS map provided for {i}. Will use measured'
                                 'background RMS for weighting. \n')
            elif isinstance(error_path, type(None)):
                self._vprint('No RMS maps provided. Will use measured'
                             'background RMS for weighting. \n')
                error_path = [error_path] * 2
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                                'form [detection, measurement], or None for no weighting.') 
            
        # If not, we should be in single image mode.
        elif isinstance(science_path, str):
            self._vprint('Starting extraction in single image mode.')
            single_mode = True
            
            if isinstance(error_path, type(None)):
                self._vprint('No RMS map provided. Will use measured background RMS for weighting. \n')
            elif isinstance(error_path, str) == False:
                raise ValueError('Single image mode requires a string path to a weight map or None '
                               'for no weighting.') 
            
            # Duplicate the inputs to match double image format.
            science_path = [science_path] * 2
            error_path = [error_path] * 2

        # If we get here, the inputs are very wrong.   
        else:
            raise ValueError('Image inputs are not the correct format. Use strings for single image '
                           'mode and lists of the form [detection, measurement] for double. '
                           'Use None for no weighting.')

        # Name the catalogue after the measurement image.
        if isinstance(cat_name, type(None)):
            cat_name = f'{outdir}/{os.path.basename(science_path[1]).removesuffix(".fits")}_photutils.hdf5'
        else:
            cat_name = f'{outdir}/{cat_name.split(".")[0]}.hdf5'
        self._cat_name = cat_name

        # Load detection image.
        self._vprint(f'Processing {os.path.basename(science_path[0])}:')
        sci, hdr = fits.getdata(science_path[0], header=True)

        # Load RMS map if available.
        if isinstance(error_path[0], type(None)) == False:
            err = fits.getdata(error_path[0])
        else:
            err = None

        # Measure the background.
        bkg = self._measure_background(sci, err, config)
        if isinstance(err, type(None)):
            err = bkg.background_rms
        if config['BKG_SUB']:
            sci -= bkg.background

        # Filter the image if required.
        if config['FILTER'] != None:
            sci_filt = self._filter(sci, err, bkg, config)
        else:
            sci_filt = sci

        # Identify the sources and save segmentation map.
        segmap = self._segmentation(sci_filt, err, config)
        if config['SEGMAP'] != None:
            self._vprint(f' Saving segmentation map to {outdir}/{config["SEGMAP"]}.fits')
            fits.writeto(f'{outdir}/{config["SEGMAP"]}.fits', segmap.data, hdr, overwrite=True)

        # Get the WCS information from the header.
        wcs = WCS(hdr)

        # Mask the off detector regions.
        mask = (err <= 0) + (~np.isfinite(err)) + (~np.isfinite(sci))

        # Should convolved data be used to measure properties?
        convolved_data = None
        if config['CONVOLVED'] == True:
            if config['FILTER'] != None:
                convolved_data = sci_filt
            else:
                raise ValueError('Requested filtered image be used to measure source properties'
                                ' but filtering is turned off')

        # Measure the properties of the sources.
        cat = SourceCatalog(
            sci, segmap, convolved_data=convolved_data, error=err, mask=mask,
            background=bkg.background, wcs=wcs, localbkg_width=config['LOCALBKG_WIDTH'],
            apermask_method=config['APERMASK_METHOD'], kron_params=config['KRON_PARAMS'],
            detection_cat=None, progress_bar=False)

        # If in double image mode, repeat with the measurement images.
        if not single_mode :
            self._vprint(f'Processing {os.path.basename(science_path[1])}:')
            sci = fits.getdata(science_path[1])

            # Load RMS map if available.
            if isinstance(error_path[1], type(None)) == False:
                err = fits.getdata(error_path[1])
            else:
                err = None

            # Measure the background.
            bkg = self._measure_background(sci, err, config)
            if isinstance(err, type(None)):
                err = bkg.background_rms
            if config['BKG_SUB']:
                sci -= bkg.background

            # Filter the image if required.
            if config['FILTER'] != None:
                sci_filt = self._filter(sci, err, bkg, config)
            else:
                sci_filt = sci

            # Mask the off detector regions.
            mask = (err <= 0) + (~np.isfinite(err)) + (~np.isfinite(sci))

            # Should convolved data be used to measure properties?
            convolved_data = None
            if config['CONVOLVED'] == True:
                if config['FILTER'] != None:
                    convolved_data = sci_filt
                else:
                    raise ValueError('Requested filtered image be used to measure source properties'
                                    ' but filtering is turned off')

            # Measure the photometry.
            self._vprint('Measuring source properties...')
            cat = SourceCatalog(
                sci, segmap, convolved_data=convolved_data, error=err, mask=mask,
                background=bkg.background, localbkg_width=config['LOCALBKG_WIDTH'], 
                apermask_method=config['APERMASK_METHOD'], kron_params=config['KRON_PARAMS'],
                detection_cat=cat, progress_bar=False)
        
        # Selection array to only keep objects with a position 
        # measurement.
        s = np.isfinite(cat.xcentroid) & np.isfinite(cat.ycentroid)
        
        # Calculate RA and DEC of the centroids.
        ra_dec = wcs.pixel_to_world(cat.xcentroid, cat.ycentroid)
        cat.add_extra_property('RA', ra_dec.ra.deg)
        cat.add_extra_property('DEC', ra_dec.dec.deg)
        
        # Measure circular aperture photometry if requested.
        labels = []
        for idx, radius in enumerate(config['RADII']):
            cat.circular_photometry(radius, f'APER_{idx}', overwrite=False)
            labels += [f'APER_{idx}_flux', f'APER_{idx}_fluxerr']

        if config['EMPIRICAL']:
            labels_ = self._empirical_uncertainty(sci, err, segmap.data, cat, config)
            labels += labels_

        # Get the full list of avilable outputs.
        output_names = set(self.output_names + labels)

        # Now add everything to the hdf5 catalogue.
        with h5py.File(cat_name, 'w') as f:
            f.create_group('photometry')

            # If no outputs requested, use all bar the image cutouts.
            if outputs == None:
                outputs = output_names
                outputs.difference_update(['background', 'convdata', 'data', 'error', 'segment'])
            outputs = set(outputs)

            # Flux values need to be converted to desired units.
            flux_quantities = ['background_centroid', 'background_mean', 'background_sum', 
                               'kron_flux', 'kron_fluxerr', 'local_background', 'max_value',
                               'min_value', 'segment_flux', 'segment_fluxerr'] + labels
            
            # Add quantities to hdf5 catalogue.
            for output in outputs:
                if output in output_names:

                    # Need to get things in the right format.
                    attr = getattr(cat, output)
                    if isinstance(attr, SkyCoord):
                        attr = np.array([list(coord) for coord in zip(attr.ra.deg, attr.dec.deg)])

                    # Add to catalogue.
                    if output in flux_quantities:
                        f[f'photometry/{output}'] = attr[s] * config['CONVERSION']
                    else:
                        f[f'photometry/{output}'] = attr[s]
                else:
                      self._vprint(f'Skipping {output} as it is not a recognised output quantity. '
                                   'Check Photutils.output_names for available outputs.')

            # Add the config parameters as attributes.
            for key, value in att_config.items():
                f['photometry'].attrs[key] = value
            f['photometry'].attrs['CODE'] = 'Photutils'
            f['photometry'].attrs['VERSION'] = photutils.__version__

        self._vprint(f'Completed extraction and saved to {cat_name}')

        return cat_name