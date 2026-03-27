import os
import re
import copy
import h5py
import yaml
import traceback
import subprocess
import warnings
import shutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation

from astropy.table import Table
from astropy.io import ascii, fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import FITSFixedWarning

import emcee
from photutils.utils import ImageDepth

from crest.utils import TempFileManager, measure_curve_of_growth

warnings.filterwarnings('ignore', category=FITSFixedWarning)

class SourceExtractor():
    """
    Wrap Source Extractor (SE) maintaining its key functionality 
    and producing hdf5 catalogues.
    """

    def __init__(self, config_path, sexpath=None, verbose=True):
        """
        __init__ method for SourceExtractor.

        Arguments
        ---------
        config_path (str)
            Path to YAML configuration file.
        sexpath (str/None)
            Path to SE executable. If not provided, find on PATH.
        verbose (bool)
            If True, print progress messages.
        """

        # Read the configuration file and split into SE and CREST
        # specific parts.
        self.configfile = config_path
        with open(self.configfile, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.SEconfig, self.config = content

        # Fix the catalogue type.
        self.SEconfig['CATALOG_TYPE'] = 'ASCII_HEAD'

        # Raise a warning if SE version can't be determined.
        self.sexpath = self._resolve_sexpath(sexpath)
        self.version = self.get_version()
        self.verbose = verbose

        # Keep track of temporary files.
        self._temp_manager = TempFileManager()

        self._outdir = None
        self._prefix = None

    def _resolve_sexpath(self, sexpath):
        """
        Resolve the SE executable path.

        Arguments
        ---------
        sexpath (str/None)
            Explicit path to the SE executable.

        Returns
        -------
        resolved_path (str)
            Executable path to SE.
        """

        # If a path or command was explicitly passed, try to 
        # resolve it first.
        if sexpath:
            resolved = shutil.which(sexpath)
            if resolved is not None:
                return resolved
            raise FileNotFoundError(
                f'Could not find Source Extractor executable: {sexpath}. '
                'Check that the path is correct or that the command is on PATH.'
            )

        # Fall back to common executable names on PATH.
        resolved = shutil.which('sex')
        if resolved is None:
            resolved = shutil.which('source-extractor')
        if resolved is not None:
            return resolved

        raise FileNotFoundError(
            'Could not find Source Extractor on PATH. Install Source Extractor '
            'or pass the executable path with the sexpath argument.'
        )

    def _vprint(self, *args, **kwargs):
        """
        Print only when verbose output is enabled.
        """
        if self.verbose:
            print(*args, **kwargs)

    def _generate_default(self):
        """
        Generate the default SE configuration file.

        Returns
        -------
        config_path (str)
            Path to the generated configuration file.
        """

        # Pipe the SExtractor output. -d requests the default parameters.
        p = subprocess.Popen([self.sexpath, "-d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        # Write these parameters to a file.
        config_path = f'{self._outdir}/{self._prefix}default.temp.sex'
        self._temp_manager.register(config_path)

        with open(config_path, 'w') as f:
            f.write(out.decode(encoding='UTF-8'))

        return config_path
    
    def get_version(self):
        """
        Retrieve the SE version.

        Returns
        -------
        version (str)
            The SE version.
        """

        # Run SE with no inputs.
        p = subprocess.Popen([self.sexpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        stdout_text = out.decode(encoding='UTF-8', errors='ignore')
        stderr_text = err.decode(encoding='UTF-8', errors='ignore')
        combined_text = '\n'.join([stdout_text, stderr_text])

        # Search for version token first, then a numeric fallback.
        version_match = re.search(r"[Vv]ersion\s+([0-9]+(?:\.[0-9]+)*)", combined_text)
        if version_match is None:
            version_match = re.search(r"\b([0-9]+(?:\.[0-9]+){1,})\b", combined_text)

        if version_match is None:
            raise RuntimeError('Could not determine Source Extractor version. Check the output of'
                               f' running {self.sexpath}')
        version = version_match.group(1)

        return version
     
    def _write_params(self, params):
        """
        Write output parameters to a text file in SE format.

        Arguments
        ---------
        params (set[str])
            Set of parameters to be written to the file.

        Returns
        -------
        parameter_path (str)
            Path to the generated parameter file.
        """

        # Write the parameters to a file.
        parameter_path = f'{self._outdir}/{self._prefix}temporary_parameters.temp.params'
        self._temp_manager.register(parameter_path)

        with open(parameter_path, 'w') as f:
            f.write("\n".join(params))
            f.write("\n")

        return parameter_path
        
    def _convert_to_hdf5(self, catalogue, config):
        """
        Convert a SE ascii catalogue to HDF5.

        Arguments
        ---------
        catalogue (str)
            Path to SE catalogue file to be converted.
        config (dict)
            Config containing all parameters to be added as attributes.

        Returns
        -------
        hdf5_name (str)
            Path to the generated hdf5 file.
        """

        self._vprint('Saving to hdf5 catalogue.')

        # Read the ascii catalogue.
        cat = Table.read(catalogue, format='ascii')

        # Create HDF5 file with the same name.
        hdf5_name = catalogue.replace(".temp.cat", '.hdf5')
        with h5py.File(hdf5_name, 'w') as f:

            # Add contents to a "photometry" group.
            f.create_group('photometry')
            for column in cat.colnames:

                # Convert fluxes to the requested units.
                if ('FLUX' in column) and (column not in ['FLUX_GROWTHSTEP', 'FLUX_RADIUS']):
                    f[f'photometry/{column}'] = cat[column] * config['TO_FLUX']
                else:
                    f[f'photometry/{column}'] = cat[column]
            
            # Add the parameters used as attributes.
            for key in config:
                f['photometry'].attrs[key] = config[key]
            f['photometry'].attrs['CODE'] = 'Source Extractor'
            f['photometry'].attrs['VERSION'] = self.version
        
        return hdf5_name
    
    def _run_SExtractor(self, basecmd, SEconfig):
        """
        Passes a command to SE on the command line.

        Arguments
        ---------
        basecmd (str)
            String containing the base command line arguments.
        SEconfig (dict)
            SE config containing the parameter updates to be added.
        """

        # Copy the base SE command.
        SEcmd = copy.deepcopy(basecmd)

        # Add parameters given in the config.
        for (key, value) in SEconfig.items():
            SEcmd.append("-" + str(key))
            SEcmd.append(str(value).replace(' ',''))

        # Run SExtractor and collect outputs.
        self._temp_manager.register(SEconfig['CATALOG_NAME'])
        res = subprocess.run(SEcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
        if res.stderr:
            for line in res.stderr.decode(encoding="UTF-8", errors='ignore').splitlines():
                clean_line = ansi_escape.sub('', line)
                print(clean_line, end='\n', flush=True)
        if res.returncode != 0:
            raise RuntimeError('Source Extractor encountered an error. Check the '
                               'output for further information.')

        return
    
    def _update_config(self, parameters):
        """
        Copy and update the stored SE and CREST config dictionaries 
        with parameters provided at runtime.

        Arguments
        ---------
        parameters (dict)
            Key-value pairs of parameters to update.
            
        Returns
        -------
        new_SEconfig (dict)
            An updated Source Extractor config.
        new_config (dict)
            An updated CREST config.
        att_config (dict)
            Config with values appropriate for saving to hdf5.
        """

        # Create new copies of the config files.
        new_SEconfig = copy.deepcopy(self.SEconfig)
        new_config = copy.deepcopy(self.config)
                    
        # Overwrite parameters with those given at run time. 
        for (key, value) in parameters.items():
            if key in new_SEconfig:
                new_SEconfig[key] = value
            elif key in new_config:
                new_config[key] = value
            else:
                raise KeyError(f'Parameter {key} not found in config file. It either doesn\'t '
                            'exist, or needs to be defined in the config file before '
                            'overwriting.')
            
        # This is always fixed.
        new_SEconfig['CATALOG_TYPE'] = 'ASCII_HEAD'
        
        # We don't want expanded environment variables in the catalogues.
        # Create combined config file without these.
        att_config = copy.deepcopy(new_SEconfig)
        att_config.update(new_config)
            
        # Expand any environment variables.
        for key, value in new_SEconfig.items():
            if isinstance(value, str):
                new_SEconfig[key] = os.path.expandvars(value)   
            if isinstance(value, list):
                new_SEconfig[key] = ','.join(value)

        return new_SEconfig, new_config, att_config
        
    def _get_aperture_config(self, SEconfig):
        """
        Return an updated config with parameters appropriate for
        measuring in apertures around set locations.
        
        Arguments
        ---------
        SEconfig (dict)
            Dictonary containing the SE config to be updated.
        
        Returns
        -------
        SEconfig (dict)
            The updated config.
        """

        # Set the parameters.
        SEconfig['DETECT_MINAREA'] = 1
        SEconfig['DETECT_THRESH'] = 1E-12
        SEconfig['WEIGHT_TYPE'] = 'NONE'
        SEconfig['FILTER'] = 'N'
        SEconfig['CLEAN'] = 'N'
        SEconfig['MASK_TYPE'] = 'NONE'
        SEconfig['BACK_TYPE'] = 'MANUAL'
        SEconfig['BACK_VALUE'] = 0.0
        SEconfig['CHECKIMAGE_TYPE'] = 'NONE'
        SEconfig['BACKPHOTO_TYPE'] = 'GLOBAL'

        return SEconfig
    
    def _get_aperture_locations(self, sci, hdr, mask, radius, napers=10000, overlap=False,
                               overlap_maxiters=50000, outname='aperture_image.fits'):
        """
        Create a detection image with value one at random aperture
        centres.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D science image in which to place the apertures.
        hdr (astropy.io.fits.Header)
            Header containing image WCS information.
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
        outname (str)
            The name of the output detection image file.
        
        Returns
        -------
        outname (str)
            The name of the output detection image file.
        """

        # Get the random aperture locations.
        depth = ImageDepth(radius, nsigma=1.0, napers=napers, niters=1, overlap=overlap,
                           overlap_maxiters=overlap_maxiters)
        limits = depth(sci, mask)
        self._vprint(f' Placed {depth.napers_used[0]} apertures.')

        locations = depth.apertures[0].positions

        # Construct the detection image.
        det = np.zeros(sci.shape, dtype=np.uint8)
        for i in np.round(locations).astype(int):
            det[i[1], i[0]] = 1

        self._temp_manager.register(outname)
        fits.writeto(outname, det.astype(np.uint8), header=hdr, overwrite=True)

        return outname

    def measure_depth(self, science_path, psf_path, mask_path=None, weight_path=None, 
                      parameters=None, radius=3.33, max_apers=50, max_iters=50000, outdir='./'):
        """
        Use randomly placed apertures to measure the average total
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
            a SE segmentation map.
        weight_path (None/str)
            Filename of fits weight map. If None, no weighting will be 
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
        outdir (str)
            The directory in which to save temporary files.
        
        Returns
        -------
        depth (float)
            The average total 5-sigma depth of the image.
        """

        if parameters is None:
            parameters = {}
        if os.path.isdir(outdir):
            self._outdir = outdir
        else:
            raise NotADirectoryError(f'{outdir} is not a directory. Set "outdir" to an '
                                     'existing directory.')  

        try:

            self._vprint(f'Measuring 5-sigma depth of {os.path.basename(science_path)}.')
            self._prefix = os.path.splitext(os.path.basename(science_path))[0]

            # Update the config files.
            sexfile = self._generate_default()
            SEconfig_depth, config_depth, _ = self._update_config(parameters)
        
            # Open the science image.
            sci, hdr = fits.getdata(science_path, header=True)
                
            # Use provided source mask
            if isinstance(mask_path, str):
                source_mask = fits.getdata(mask_path)

            # or run SE to generate one.
            else:
                self._vprint('No source mask provided, will use generated segmentation map.')
                SEconfig_depth['CATALOG_NAME'] = f'{self._outdir}/{self._prefix}depth_mask.temp.cat'

                # Set up the command.
                basecmd = [self.sexpath, "-c", sexfile, science_path]

                # Ensure weight map is provided correctly.
                if isinstance(weight_path, str):
                    basecmd += ['-WEIGHT_IMAGE', weight_path]
                    
                    # Are we dealing with relative weights or RMS?
                    if len(SEconfig_depth['WEIGHT_TYPE'].split(',')) > 1:
                        raise ValueError('Single image mode but WEIGHT_TYPE = '
                                         f'{SEconfig_depth["WEIGHT_TYPE"]}.')
                    elif (('NONE' in SEconfig_depth['WEIGHT_TYPE']) or 
                          ('BACKGROUND' in SEconfig_depth['WEIGHT_TYPE'])):
                        raise ValueError('Weight map provided but WEIGHT_TYPE = '
                                         f'{SEconfig_depth["WEIGHT_TYPE"]}.')

                # If no weights are not being used, ensure types are set 
                # correctly.         
                elif isinstance(weight_path, type(None)):
                    if (SEconfig_depth['WEIGHT_TYPE'].count('NONE') + 
                        SEconfig_depth['WEIGHT_TYPE'].count('BACKGROUND')) != 1:
                        raise ValueError('No weight image provided but WEIGHT_TYPE = '
                                         f'{SEconfig_depth["WEIGHT_TYPE"]}.')

                # Run SE to get the mask.
                SEconfig_depth['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
                check_name = SEconfig_depth["CATALOG_NAME"].replace(".temp.cat", ".seg.temp.fits")
                SEconfig_depth['CHECKIMAGE_NAME'] = check_name
                self._temp_manager.register(check_name)
                SEconfig_depth['PARAMETERS_NAME'] = self._write_params(['NUMBER'])

                self._run_SExtractor(basecmd, SEconfig_depth)
                source_mask = fits.getdata(check_name)

            # Full mask includes sources and off detector regions.
            full_mask = (source_mask != 0) + (~np.isfinite(sci))
            if weight_path is not None:
                wht = fits.getdata(weight_path)
                full_mask += (~np.isfinite(wht)) + (wht <= 0)

            # Place random apertures and save as detection image.
            SEconfig_depth = self._get_aperture_config(SEconfig_depth) 
            det_filename = f'{self._outdir}/{self._prefix}depth_apertures.temp.fits'
            self._get_aperture_locations(sci, hdr, full_mask, radius, max_apers, overlap=False,
                                        overlap_maxiters=max_iters, outname=det_filename)

            # Run SE.
            cat_name = f'{self._outdir}/{self._prefix}depth_apertures.temp.cat'
            SEconfig_depth['CATALOG_NAME'] = cat_name
            SEconfig_depth['PHOT_APERTURES'] = str(round(radius*2, 2))
            SEconfig_depth['PARAMETERS_NAME'] = self._write_params(['FLUX_APER', 'NUMBER'])

            detcmd = [self.sexpath, "-c", sexfile, det_filename, science_path]
            self._run_SExtractor(detcmd, SEconfig_depth)

            # Get the aperture fluxes.
            apps = ascii.read(SEconfig_depth['CATALOG_NAME'])
            flux = apps['FLUX_APER'] * config_depth['TO_FLUX']

            # Measure the median absolute deviation.
            s = (flux != 0) & (np.isfinite(flux))
            mad = median_abs_deviation(flux[s], nan_policy='omit', scale='normal')

            # Measure the PSF curve of growth and interpolate.
            psf_ = fits.getdata(psf_path)
            radii = np.arange(0.1, psf_.shape[0], 1)
            radii, cog, = measure_curve_of_growth(psf_, radii=radii, position=None)
            f = lambda r: np.interp(r, radii, cog)

            # Correct by the enclosed energy.
            depth = 5*mad/f(radius)

            return depth
        
        except Exception:
            traceback.print_exc()
            raise

        finally:
            self._temp_manager.cleanup()
    
    def _empirical_uncertainty(self, science_path, weight_path, weight_type, segmap_path, 
                               SEconfig, config):
        """
        Perform empirical uncertainty estimation by fitting the relation
        between aperture size and noise. Based on Finkelstein+23.

        Arguments
        ---------
        science_path (str)
            Filename of the science image.
        weight_path (str)
            Filename of the corresponding weight map.
        weight_type (str)
            The type of the weight map, either MAP_WEIGHT, 
            MAP_VAR or MAP_RMS.
        segmap_path (str)
            Path to the segmentation map generated by SE.
        SEconfig (dict)
            SE configuration parameters specific to this image.
        config (dict)
            CREST parameters specific to this image.
        """

        self._vprint('\nMeasuring empirical uncertainties...')

        # Update the config to deal with aperture locations.
        err_SEconfig = copy.deepcopy(SEconfig)
        err_config = copy.deepcopy(config)

        err_SEconfig = self._get_aperture_config(err_SEconfig)

        # Generate a default configuration file.
        sexfile = self._generate_default()

        # Open images and construct mask. 
        sci, hdr = fits.getdata(science_path, header = True)
        seg = fits.getdata(segmap_path)

        # Convert to RMS.
        err = fits.getdata(weight_path)
        if weight_type == 'MAP_WEIGHT':
            err = 1/np.sqrt(err)
        elif weight_type == 'MAP_VAR':
            err = np.sqrt(err)
        elif weight_type == 'MAP_RMS':
            pass
        else:
            raise ValueError(f'Invalid weight type {weight_type}, only'
                             'MAP_WEIGHT, MAP_VAR, and MAP_RMS are supported.')
            
        mask = (err <= 0) + (~np.isfinite(sci)) + (~np.isfinite(err))
        
        # Seperate the radii into small and large components.
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

            # Get the aperture locations.
            app_filename = f'{self._outdir}/{self._prefix}{run}_apertures.temp.fits'
            self._get_aperture_locations(
                sci, hdr, mask+(seg!=0), max(radii[s]), err_config[f'N_{run.upper()}'], False, 
                err_config['MAX_ITERS'], app_filename)

            err_SEconfig['CATALOG_NAME'] = app_filename.replace('.fits', '.cat')

            # Tell SE the aperture sizes to use.
            apertures = ''
            for radius in radii[s]:
                apertures += str(round(radius, 2)*2) + ','
            apertures = apertures[:-1]
            err_SEconfig['PHOT_APERTURES'] = apertures

            parameter_filename = self._write_params([f'FLUX_APER({sum(s)})'])
            err_SEconfig['PARAMETERS_NAME'] = parameter_filename

            # Run SE.
            basecmd = [self.sexpath, "-c", sexfile, app_filename, science_path]
            self._run_SExtractor(basecmd, err_SEconfig)

            # Calculate the MAD in each aperture.
            app_cat = ascii.read(err_SEconfig['CATALOG_NAME'])
            s = app_cat['FLUX_APER'] != 0
            for column in app_cat.colnames:
                medians.append(median_abs_deviation(app_cat[column][s], nan_policy='omit', 
                                                    scale='normal'))

            self._temp_manager.delete(app_filename)

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
            if -1e9 < a < 1e9 and -1e9 < b< 1e9:
                return 0.0
            return -np.inf
        
        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, x, y, yerr)
        
        # The step methodology.
        initial = np.array(err_config['INITIAL'])
        p0 = [initial + 1e-7 * np.random.randn(len(initial)) for i in range(err_config['WALKERS'])] 
        
        # The percentage error to use when fitting. 
        Merr = err_config['P_ERR']*np.array(medians)
        
        # Begin the MCMC.
        sampler = emcee.EnsembleSampler(err_config['WALKERS'], len(initial), lnprob, 
                                        args=(Npix, medians, Merr))

        p0, _, _ = sampler.run_mcmc(p0, err_config['BURN_IN'])
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(p0, err_config['N_ITERS'])

        # Get most likely parameter values.
        samples = sampler.get_chain(flat=True)
        theta_max  = samples[np.argmax(sampler.get_log_prob(flat=True))]

        # Median error value of the whole map.
        median_err = np.median(err[~mask])

        # We now want the radii of apertures used for photometry.
        radii = SEconfig['PHOT_APERTURES']
        radii = [float(i) for i in radii.split(',')]        

        # Read original catalogue produced by SE.
        cat = ascii.read(SEconfig['CATALOG_NAME'])

        # Expecting a few NaNs so quiet any warnings.
        with np.errstate(invalid='ignore'):

            # Will scale errors by this relative value.
            rel_e = err[cat['Y_IMAGE'].astype(int)-1, cat['X_IMAGE'].astype(int)-1] / median_err

            # For each flux column, calculate the area based on the type
            # of aperture and extract the noise.
            for column in cat.colnames:

                if column == 'FLUX_AUTO':
                    area = np.pi * cat['A_IMAGE'] * cat['B_IMAGE'] * (cat['KRON_RADIUS']**2)
                    cat['FLUXERR_AUTO_EMPIRICAL'] = model(theta_max, area) * rel_e 
                    
                if column == 'FLUX_APER':
                    area = np.pi * np.power(radii[0], 2)
                    cat['FLUXERR_APER_EMPIRICAL'] = model(theta_max, area) * rel_e

                if 'FLUX_APER_' in column:
                    aper = int(column.split('FLUX_APER_')[1])
                    area = np.pi * np.power(radii[aper], 2)
                    cat[f'FLUXERR_APER_{aper}_EMPIRICAL'] = model(theta_max, area) * rel_e 

            cat.write(SEconfig['CATALOG_NAME'], format='ascii', overwrite=True)

        # Save a plot of noise vs aperture size.
        if err_config['SAVE_FIG'] == True:

            x = np.linspace(0, max(Npix), 10000)
            fig, ax = plt.subplots(1, 1, figsize=(3.78, 3.78))

            ax.errorbar(np.sqrt(Npix), medians, yerr=np.array(medians)*err_config['P_ERR'], 
                        fmt='none', color='blue', alpha=0.8)
            ax.scatter(np.sqrt(Npix), medians, color='blue', alpha=0.8, s=5)
            ax.plot(np.sqrt(x), model(theta_max, x), color = 'grey', linestyle = '--',
                     linewidth = 1)  
            ax.set_xlabel('sqrt(Number of pixels in aperture)')
            ax.set_ylabel('Noise in aperture [counts]')
            title = os.path.basename(SEconfig["CATALOG_NAME"]).removesuffix('.temp.cat')
            ax.set_title(f'{title.split(".cat")[0]}', fontsize = 10)
            fig.savefig(SEconfig["CATALOG_NAME"].replace('.temp.cat', '_noise.png'))
            plt.close()
        
        return

    def extract(self, science_path, weight_path=None, parameters=None, output=None, 
                cat_name=None, outdir='./'):
        """
        Run Source Extractor in any of its standard modes.

        Arguments
        ---------
        science_path (str/List[str])
            If str, the filename of the image to extract.
            If a List[str] filename of detection and measurement images.
        weight_path (None/str/List[str])
            If None, ignore weighting.
            If str, path to corresponding weight map.
            If List[str], paths to weight maps for detection 
            and measurement.
        parameters (None/dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        output (None/list)
            List of output parameters to save. If None, return some key 
            values.
        cat_name (None/str)
            The base name for the photometry catalogue. If None, use the 
            base name of the measurement file.
        outdir (str)
            Directory in which to store outputs. 

        Returns
        -------
        outname (str)
            The name of the output hdf5 photometry catalogue.
        """
		
        try:

            if parameters is None:
                parameters = {}
            
            # All files will be output here.
            if os.path.isdir(outdir):
                self._outdir = outdir
            else:
                raise NotADirectoryError(f'{outdir} is not a directory. Set "outdir" to an existing'
                                         ' directory.')

            # Create a copy of the config for this run.
            img_SEconfig, img_config, att_config = self._update_config(parameters)

            # Check each checkimage name to ensure the directory exists.
            # The SE error for this is not very helpful.
            check_images = {}
            if img_SEconfig['CHECKIMAGE_TYPE'].strip() != 'NONE':

                for check_type, check_name in zip(img_SEconfig['CHECKIMAGE_TYPE'].split(','), 
                                                  img_SEconfig['CHECKIMAGE_NAME'].split(',')):
                    check_type = check_type.strip()
                    check_name = check_name.strip()

                    dir_name = os.path.dirname(check_name)
                    if dir_name == '':
                        check_name = f'{outdir}/{check_name}'
                    elif os.path.isdir(dir_name) == False:
                        raise NotADirectoryError(f'{dir_name} does not exist. Use an existing '
                                                 ' directory when defining CHECKIMAGE_NAME.')
                    check_images[check_type] = check_name

            # If no output requested, return only key quantities. 
            if isinstance(output, type(None)):
                output = {'NUMBER', 'X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO'}
            output = set(output)            

            # Check for double mode.
            if isinstance(science_path, list):
                self._vprint('Starting extraction in dual image mode.')
                
                # Get the file prefix and catalogue name.
                self._prefix = os.path.splitext(os.path.basename(science_path[1]))[0]
                if isinstance(cat_name, type(None)):
                    img_SEconfig['CATALOG_NAME'] = f'{outdir}/{self._prefix}_sextractor.temp.cat'
                else:
                    img_SEconfig['CATALOG_NAME'] = f'{outdir}/{os.path.basename(cat_name)}.temp.cat'
                    self._prefix = cat_name
                
                # Generate the base SE parameter file and command.
                sexfile = self._generate_default() 
                basecmd = [self.sexpath, "-c", sexfile, science_path[0], science_path[1]]  

                # Check if weights are being used.
                if isinstance(weight_path, list):
                    if len(weight_path) == 2:

                        # Are we dealing with relative weights or RMS?
                        split_weight = img_SEconfig['WEIGHT_TYPE'].split(',')
                        if len(split_weight) < 2:
                            raise ValueError('When using double image mode, WEIGHT_TYPE should have'
                                             ' the form MAP_{type},MAP_{type}.')
                        weight_type = split_weight[1].strip()

                        # How we set up the command depends on how many 
                        # weight paths were provided. 
                        s = [i is None for i in weight_path]

                        # Two weight images.
                        if sum(s) == 0:
                            if (('NONE' in img_SEconfig["WEIGHT_TYPE"]) or 
                                ('BACKGROUND' in img_SEconfig["WEIGHT_TYPE"])):
                                raise ValueError('Two weight maps provided but WEIGHT_TYPE = '
                                                 f'{img_SEconfig["WEIGHT_TYPE"]}.')
                            basecmd += ['-WEIGHT_IMAGE', f'{weight_path[0]},{weight_path[1]}']

                        # One weight image.
                        elif sum(s) == 1:
                            if (img_SEconfig['WEIGHT_TYPE'].count('NONE') + 
                                img_SEconfig['WEIGHT_TYPE'].count('BACKGROUND')) != 1:
                                raise ValueError('One weight map provided but WEIGHT_TYPE = '
                                                 f'{img_SEconfig["WEIGHT_TYPE"]}.')
                            
                            if s[0] and (('NONE' not in split_weight[0]) and 
                                         ('BACKGROUND' not in split_weight[0])):
                                raise ValueError('No detection weight provided but WEIGHT_TYPE = '
                                                 f'{img_SEconfig["WEIGHT_TYPE"]}.')
                            
                            if s[1] and (('NONE' not in split_weight[1]) and 
                                         ('BACKGROUND' not in split_weight[1])):
                                raise ValueError('No measurement weight provided but WEIGHT_TYPE = '
                                                 f'{img_SEconfig["WEIGHT_TYPE"]}.')
                            
                            weight_path = [item if item is not None else '' for item in weight_path]
                            basecmd += ['-WEIGHT_IMAGE', f'{weight_path[0]},{weight_path[1]}']
                            
                        # If no weight images, we don't need to update 
                        # the command.
                        elif sum(s) == 2:
                            if (img_SEconfig['WEIGHT_TYPE'].count('NONE') + 
                                img_SEconfig['WEIGHT_TYPE'].count('BACKGROUND')) != 2:
                                raise ValueError('No weight images provided but WEIGHT_TYPE = '
                                                 f'{img_SEconfig["WEIGHT_TYPE"]}.')

                    else:
                        raise ValueError('Double image mode requires a list of weight paths of the '
                                         'form [detection, measurement].')  
                    
                # Also allow passing a single None.       
                elif isinstance(weight_path, type(None)):
                    weight_type = 'NONE'
                    if (img_SEconfig['WEIGHT_TYPE'].count('NONE') + 
                        img_SEconfig['WEIGHT_TYPE'].count('BACKGROUND')) != 2:
                                    raise ValueError('No weight images provided but WEIGHT_TYPE = '
                                                     f'{img_SEconfig["WEIGHT_TYPE"]}.')
                else:
                    raise ValueError('Double image mode requires a list of weight paths of the '
                                     'form [detection, measurement].') 

            # If not double, hopefully we are in single image mode.
            elif isinstance(science_path, str):
                self._vprint('Starting extraction in single image mode.')

                # Get the file prefix and catalogue name.
                self._prefix = os.path.splitext(os.path.basename(science_path))[0]
                if isinstance(cat_name, type(None)):
                    img_SEconfig['CATALOG_NAME'] = f'{outdir}/{self._prefix}_sextractor.temp.cat'
                else:
                    img_SEconfig['CATALOG_NAME'] = f'{outdir}/{os.path.basename(cat_name)}.temp.cat'
                    self._prefix = cat_name

                # Generate the base SE parameter file and command.
                sexfile = self._generate_default() 
                basecmd = [self.sexpath, "-c", sexfile, science_path]

                # Check if weights are being used.
                if isinstance(weight_path, str):

                    # Update the base command.
                    basecmd += ['-WEIGHT_IMAGE', weight_path]
                    
                    # Are we dealing with relative weights or RMS?
                    if len(img_SEconfig['WEIGHT_TYPE'].split(',')) > 1:
                        raise ValueError('Single image mode but WEIGHT_TYPE = '
                                         f'{img_SEconfig["WEIGHT_TYPE"]}.')
                    elif (('NONE' in img_SEconfig['WEIGHT_TYPE']) or 
                          ('BACKGROUND' in img_SEconfig['WEIGHT_TYPE'])):
                        raise ValueError('Weight map provided but WEIGHT_TYPE = '
                                         f'{img_SEconfig["WEIGHT_TYPE"]}.')
                    weight_type = img_SEconfig['WEIGHT_TYPE'].strip()

                # If no weights are not being used, ensure types are set 
                # correctly.         
                elif isinstance(weight_path, type(None)):
                    weight_type = 'NONE'
                    if (img_SEconfig['WEIGHT_TYPE'].count('NONE') + 
                        img_SEconfig['WEIGHT_TYPE'].count('BACKGROUND')) != 1:
                        raise ValueError('No weight image provided but WEIGHT_TYPE = '
                                         f'{img_SEconfig["WEIGHT_TYPE"]}.')
                else:
                    raise ValueError('Single image mode requires a string path to a weight map or '
                                     'None')

            # If we get here, the inputs are very wrong.   
            else:
                raise ValueError('Image inputs are not the correct format. Use strings for single '
                                 'image mode and lists of the form [detection, measurement] for '
                                 'double. Use None for background based or noweighting.')

            # Will uncertainties be estimated empirically?
            if img_config['EMPIRICAL'] == True:
            
                # we will need these quantitites.
                for i in ['A_IMAGE','B_IMAGE','KRON_RADIUS', 'X_IMAGE', 'Y_IMAGE']:
                    output.add(i)

                # Will also need a segmentation map.
                if 'SEGMENTATION' in check_images.keys():
                    segmap = check_images['SEGMENTATION']
                else:
                    segmap = img_SEconfig["CATALOG_NAME"].replace(".temp.cat", ".seg.temp.fits")
                    self._temp_manager.register(segmap)
                    check_images['SEGMENTATION'] = segmap
                
                # And possibly a weight map.
                if (weight_type == 'NONE') or (weight_type == 'BACKGROUND'):
                    if 'BACKGROUND_RMS' in check_images.keys():
                        weight_path = check_images['BACKGROUND_RMS']
                    else:
                        weight_path = img_SEconfig["CATALOG_NAME"].replace(".temp.cat", ".rms.temp.fits")
                        self._temp_manager.register(weight_path)
                        check_images['BACKGROUND_RMS'] = weight_path
                    weight_type = 'MAP_RMS'
                elif isinstance(weight_path, list):
                    weight_path = weight_path[1]
            
            # Add all the checkimage requests to the config.
            img_SEconfig['CHECKIMAGE_TYPE'] = ','.join(check_images.keys())
            img_SEconfig['CHECKIMAGE_NAME'] = ','.join(check_images.values())
                    
            # Always need at least 2 outputs.
            if len(output) < 2:
                output.add('NUMBER')
            if len(output) < 2:
                output.add('FLUX_AUTO')

            # Write the full set of output parameters to a text file.
            parameter_filename = self._write_params(output)
            img_SEconfig['PARAMETERS_NAME'] = parameter_filename

            # Run SE using this command and the config parameters.
            self._run_SExtractor(basecmd, img_SEconfig)

            # Begin uncertainty estimation.
            if img_config['EMPIRICAL'] == True:
                if isinstance(science_path, list):
                    self._empirical_uncertainty(science_path[1], weight_path, weight_type, segmap, 
                                                img_SEconfig, img_config)
                else:
                    self._empirical_uncertainty(science_path, weight_path, weight_type, segmap, 
                                                img_SEconfig, img_config)                   
            
            # Combine the two config files and save everything to hdf5.
            outname = self._convert_to_hdf5(img_SEconfig['CATALOG_NAME'], att_config)

            self._vprint(f'Completed extraction and saved to {outname} \n')

            return outname
        
        except Exception:
            traceback.print_exc()
            raise

        finally:
            self._temp_manager.cleanup()