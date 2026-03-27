# Adapted from the Bagley+23 code available at 
# https://github.com/ceers/ceers-nircam 

import os
import yaml
import copy
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import stats as astrostats
from astropy.wcs import WCS
from astropy.convolution import convolve_fft, Ring2DKernel, Gaussian2DKernel
from scipy.ndimage import median_filter, binary_dilation, distance_transform_edt

from photutils.background import Background2D, BiweightLocationBackground, BkgIDWInterpolator
from photutils.background import BkgZoomInterpolator
from photutils.segmentation import detect_sources
from photutils.utils import circular_footprint

from crest.utils import _parallel_execute, _tile_worker, _construct_tiles

class Background():
    """
    Perform background subtraction on multiple observations based on 
    individual or merged tiered source masks. Largely based on the 
    Bagley+2023 approach, but now includes additional scaling in low
    weight regions and tiling to increase speed.
    """

    def __init__(self, config_path, verbose=True):
        """
        __init__ method for Background class.
        
        Arguments
        ---------
        config_path (str)
            Path to YAML configuration file specifying parameters to use
            at each step.
        verbose (bool)
            If True, print progress messages.
        """

        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
        self.config = cfg
        self.config_filepath = config_path
        self.verbose = verbose

    def _vprint(self, *args, **kwargs):
        """
        Print only when verbose output is enabled.
        """
        if self.verbose:
            print(*args, **kwargs)

    def _replace_mask(self, sci, mask):
        """
        Replace masked regions of an image with a biweight background 
        estimate.
        
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D image to replace.
        mask (numpy.ndarray)
            The 2D image mask where True regions are to be replaced.

        Returns
        -------
        sci_filled (numpy.ndarray)
            The image with masked regions replaced by the biweight 
            background.
        """

        # Replace the masked regions with an approximate background.
        robust_mean_background = astrostats.biweight_location(sci[~mask], c=6., ignore_nan=True)
        sci_filled = np.where(mask, robust_mean_background, sci).astype(np.float32)

        return sci_filled

    def _clipped_ring_median_filter(self, sci, mask, config):
        """
        Remove ring median filtered signal from an image.
            
        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D image to be filtered.
        mask (numpy.ndarray)
            A 2D image mask where True regions are to be masked.
        config (dict)
            The background config dictionary.
            
        Returns
        -------
        rmf_image (numpy.ndarray)
            The 2D image after subtracting the ring median filtered 
            signal.
        """

        # First make a smooth background.
        bkg = Background2D(
            sci, box_size = config["RING_CLIP_BOX_SIZE"], 
            sigma_clip = astrostats.SigmaClip(sigma=config["BG_SIGMA"]),
            filter_size = config["RING_CLIP_FILTER_SIZE"],
            bkg_estimator = BiweightLocationBackground(),
            exclude_percentile = 90,
            mask = mask,
            interpolator = BkgZoomInterpolator())
        
        # Apply a floating ceiling based on the RMS.
        background_rms = astrostats.biweight_scale((sci - bkg.background)[~mask]) 

        ceiling = config["RING_CLIP_MAX_SIGMA"] * background_rms + bkg.background
        ceiling_mask = sci > ceiling

        sci_filled = self._replace_mask(sci, mask | ceiling_mask).astype(np.float32)

        # Filter with a ring kernel.
        self._vprint(f'Ring median filtering with radius, width = '
                     f'{config["RING_RADIUS_IN"]}, {config["RING_WIDTH"]}')

        ring = Ring2DKernel(config["RING_RADIUS_IN"], config["RING_WIDTH"])
        halo = int(config['RING_RADIUS_IN'] + config['RING_WIDTH'])

        # If no tiling requested, filter the full image.       
        n_tiles = config.get('N_TILES', 1)
        if n_tiles <= 1:
            filtered = median_filter(sci_filled, footprint=ring.array)
            rmf_image = (sci - filtered).astype(np.float32)
            return rmf_image

        # Otherwise split into tiles.
        ny, nx = sci.shape
        rmf_image = np.zeros_like(sci, dtype=np.float32)            

        slices = _construct_tiles((ny, nx), int(n_tiles), halo)

        # Prepare tasks for each tile.
        tasks = []
        for s in slices:
            y0, y1, x0, x1, e0, e1, f0, f1 = s
            block = sci[e0:e1, f0:f1]
            filled_block = sci_filled[e0:e1, f0:f1]
            tasks.append({'block': block, 'filled_block': filled_block, 
                          'slices': s, 'ring_footprint': ring.array})

        # Execute in parallel and stitch tiles back together.
        workers = max(int(config.get('N_WORKERS', 1)), 1)
        results = _parallel_execute(_tile_worker, tasks, workers)
        for res in results:
            y0, y1, x0, x1, interior = res
            rmf_image[y0:y1, x0:x1] = interior

        return rmf_image
    
    def _tier_mask(self, img, mask, scaling, config, tiernum=0):
        """
        Update a source mask using parameters dependent on the tier of 
        masking.
            
        Arguments
        ---------
        img (numpy.ndarray)
            The 2D image to tier mask.
        mask (numpy.ndarray)
            A 2D image mask where True regions are to be masked.
        scaling (numpy.ndarray)
            The 2D image defining the detection threshold scaling of 
            each pixel.
        config (dict)
            The background config dictionary.
        tiernum (int)
            The tier of source masking.
            
        Returns
        -------
        mask (numpy.ndarray)
            The updated 2D source mask.
        """

        self._vprint(f"Tier #{tiernum}:")
        self._vprint(f'  Kernel size = {config["TIER_KERNEL_SIZE"][tiernum]}')
        self._vprint(f'  N-sigma = {config["TIER_NSIGMA"][tiernum]}')
        self._vprint(f'  N-pixels = {config["TIER_NPIXELS"][tiernum]}')
        self._vprint(f'  Dilate size = {config["TIER_DILATE_SIZE"][tiernum]}')

        # Calculate a robust RMS.
        background_rms = astrostats.biweight_scale(img[~mask])

        # Replace the masked pixels by the robust background level/
        background_level = astrostats.biweight_location(img[~mask])
        replaced_img = np.where(mask, background_level, img).astype(np.float32)

        self._vprint(f"  Median of ring-median-filtered image = {np.median(img[~mask])}")
        self._vprint(f"  Biweight RMS of ring-median-filtered image  = {background_rms}")

        # Convolve the image with a Gaussian kernel.
        gauss_kernel = Gaussian2DKernel(config["TIER_KERNEL_SIZE"][tiernum])

        # If no tiling requested, convolve the full image.       
        n_tiles = config.get('N_TILES', 1)
        if n_tiles <= 1:
            convolved_difference = convolve_fft(
                replaced_img, gauss_kernel, allow_huge=True).astype(np.float32)

        # Otherwise split into tiles.       
        else:

            kh = int(max(gauss_kernel.array.shape) // 2)

            ny, nx = img.shape
            convolved_difference = np.zeros_like(img, dtype=np.float32)

            slices = _construct_tiles((ny, nx), int(n_tiles), kh)

            # Prepare tasks for each tile.
            tasks = []
            for s in slices:
                y0, y1, x0, x1, e0, e1, f0, f1 = s
                block = replaced_img[e0:e1, f0:f1]
                tasks.append({'block': block, 'kernel': gauss_kernel, 'slices': s})

            # Execute in parallel and stitch tiles back together.   
            workers = max(int(config.get('N_WORKERS', 1)), 1)
            results = _parallel_execute(_tile_worker, tasks, workers)
            for res in results:
                y0, y1, x0, x1, interior = res
                convolved_difference[y0:y1, x0:x1] = interior

        # Construct a source mask from detections in the convolved image.
        seg_detect = detect_sources(
            convolved_difference, 
            threshold=config["TIER_NSIGMA"][tiernum] * background_rms * scaling, 
            npixels=config["TIER_NPIXELS"][tiernum],
            mask=mask)
        
        mask = seg_detect.make_source_mask()

        # If dilation requested.
        dilate_r = int(config["TIER_DILATE_SIZE"][tiernum])
        if dilate_r == 0:
            pass
        else:

            # Construct the footprint.
            footprint = circular_footprint(radius=dilate_r)
            n_tiles = config.get('N_TILES', 1)

            # Dilate the full mask if no tiling requested.
            if n_tiles <= 1:
                mask = binary_dilation(mask, structure=footprint)
            
            # Otherwise construct the tiles.
            else:
                ny, nx = img.shape
                mask_out = np.zeros_like(mask, dtype=bool)
                slices = _construct_tiles((ny, nx), int(n_tiles), dilate_r)

                # Prepare tasks for each tile.
                tasks = []
                for s in slices:
                    y0, y1, x0, x1, e0, e1, f0, f1 = s
                    block = mask[e0:e1, f0:f1]
                    tasks.append({'block': block, 'dilate_footprint': footprint, 'slices': s})

                # Execute in parallel and stitch tiles back together.
                workers = max(int(config.get('N_WORKERS', 1)), 1)
                results = _parallel_execute(_tile_worker, tasks, workers)
                for res in results:
                    y0, y1, x0, x1, interior = res
                    mask_out[y0:y1, x0:x1] = interior
                mask = mask_out

        return mask

    def _mask_sources(self, img, bitmask, scaling, config, starting_bit=1): 
        """
        Iteratively mask sources using _tier_mask.

        Arguments
        ----------
        img (numpy.ndarray)
            The 2D image to be masked.
        bitmask (numpy.ndarray)
            The 2D starting bitmask.
        scaling (numpy.ndarray)
            The 2D image defining the detection threshold scaling of 
            each pixel.
        config (dict)
            The background config dictionary.
        starting_bit (int)
            The bit of the bitmask to start on for the first tier.
        
        Returns
        -------
        bitmask (numpy.ndarray)
            The final 2D combined bitmask after all tiers of source
            masking.
        """

        # Iterate over the tiers and combine masks.
        for tiernum in range(len(config["TIER_NSIGMA"])):
            mask = self._tier_mask(img, (bitmask != 0), scaling, config, tiernum=tiernum)
            bitmask = np.bitwise_or(
                bitmask, np.left_shift(mask.astype(np.uint32), tiernum + starting_bit))
        return bitmask
    
    def _estimate_background(self, img, mask, config):
        """
        Estimate an image background using 'zoom' interpolation.
        
        Arguments
        ---------
        img (numpy.ndarray)
            The 2D image to be background subtracted.
        mask (numpy.ndarray)
            A 2D image mask where True regions are to be masked.
        config (dict)
            The background config dictionary.

        Returns
        -------
        bkg (photutils.Background2D)
            Background object measured from the image.
        """

        bkg = Background2D(img, 
                    box_size = config["BG_BOX_SIZE"],
                    sigma_clip = astrostats.SigmaClip(sigma=config["BG_SIGMA"]),
                    filter_size = config["BG_FILTER_SIZE"],
                    bkg_estimator = BiweightLocationBackground(),
                    exclude_percentile = config["BG_EXCLUDE_PERCENTILE"],
                    mask = mask,
                    interpolator = BkgZoomInterpolator())
        return bkg
    
    def _estimate_background_IDW(self, img, mask, config):
        """
        Estimate an image background using 'IDW' interpolation.
        
        Arguments
        ---------
        img (numpy.ndarray)
            The 2D image to be background subtracted.
        mask (numpy.ndarray)
            A 2D image mask where True regions are to be masked.
        config (dict)
            The background config dictionary.

        Returns
        -------
        bkg (photutils.Background2D)
            Background object measured from the image.
        """

        bkg = Background2D(img, 
                    box_size = config["BG_BOX_SIZE"],
                    sigma_clip = astrostats.SigmaClip(sigma=config["BG_SIGMA"]),
                    filter_size = config["BG_FILTER_SIZE"],
                    bkg_estimator = BiweightLocationBackground(),
                    exclude_percentile = config["BG_EXCLUDE_PERCENTILE"],
                    mask = mask,
                    interpolator = BkgIDWInterpolator())
        return bkg

    def _evaluate_bias(self, bkgd, detector_mask, mask):
        """
        Evaluate the bias between masked and unmasked pixels.
        
        Arguments
        ---------
        bkgd (numpy.ndarray)
            The 2D background to evaluate.
        detector_mask (numpy.ndarray)
            A 2D image mask where True regions are off the detector.
        mask (numpy.ndarray)
            A 2D image mask where True regions are to be masked.

        Returns
        -------
        diff (float)
            The difference in mean values under masked and unmasked
            pixels.
        significance (float)
            The significance of the difference in mean values.
        """

        on_detector = np.logical_not(detector_mask)
    
        # Mean and deviation of background under masked pixels.
        mean_masked = bkgd[mask & on_detector].mean()
        std_masked = bkgd[mask & on_detector].std()
        stderr_masked = mean_masked / (np.sqrt(len(bkgd[mask]))*std_masked)
    
        # Mean and deviation of background in unmasked regions.
        mean_unmasked = bkgd[~mask & on_detector].mean()
        std_unmasked = bkgd[~mask & on_detector].std()
        stderr_unmasked = mean_unmasked / (np.sqrt(len(bkgd[~mask]))*std_unmasked)
        
        # Calculate the significance of the difference in mean values.
        diff = mean_masked - mean_unmasked
        significance = diff / np.sqrt(stderr_masked**2 + stderr_unmasked**2)

        self._vprint(f"Mean under masked pixels   = {mean_masked:.4f} +- {stderr_masked:.4f}")
        self._vprint(f"Mean under unmasked pixels = "
                 f"{mean_unmasked:.4f} +- {stderr_unmasked:.4f}")
        self._vprint(f"Difference = {diff:.4f} at {significance:.2f} sigma significance")

        return diff, significance

    def individual_background(self, science_paths, weight_paths, parameters=None, suffix='bkgsub',
                              store_mask=True):
        """
        Perform individual background subtraction with tiered source masking.
        
        Arguments
        ---------
        science_paths (str/List[str])
            Filenames of science images to subtract the background
            from.
        weight_paths (str/List[str])
            Filenames of the corresponding weight images.
        parameters (None/dict)
            Key-value pairs overwritting parameters given in the config
            file for this run.
        suffix (str)
            Suffix to append to the science filenames when saving 
            subtracted versions.
        store_mask (bool)
            Whether to store the tiered source mask as an extension.
            Required for merged masking.
        
        Returns
        -------
        bkgsub_filenames (List[str])
            Filenames of the generated background subtracted images.
        """

        # Process the inputs.
        if parameters is None:
            parameters = {}

        if type(science_paths) == str:
            science_paths = [science_paths]
        if type(weight_paths) == str:
            weight_paths = [weight_paths]

        if len(science_paths) != len(weight_paths):
            raise KeyError('There should be corresponding images of each type.')
        
        # Overwrite some parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.',
                                  stacklevel=2)
                    
        # Load each pair of images.
        bkgsub_filenames = []
        for sci_filename, weight_filename in zip(science_paths, weight_paths):

            self._vprint(f'Measuring background of {sci_filename}...')

            sci, hdr = fits.getdata(sci_filename, header = True)   
            sci = np.asarray(sci, dtype=np.float32)
            wht = np.asarray(fits.getdata(weight_filename), dtype=np.float32)

            # Set up a bitmask
            bitmask = np.zeros(sci.shape,np.uint32)

            # First level is for masking pixels off the detector.
            off_detector_mask = (~np.isfinite(wht)) | (wht <= 0) | (~np.isfinite(sci))
            mask = off_detector_mask.copy()
            bitmask = np.bitwise_or(bitmask, np.left_shift(off_detector_mask.astype(np.uint32), 0))

            # Scale the detection threshold for low weight regions.
            scaling = np.ones(sci.shape, dtype=np.float32)

            # Find the ratio of weight to median weight.
            med_wht = np.median(wht[~off_detector_mask])
            ratio = np.where(off_detector_mask, np.nan, med_wht / wht)

            # Turn on scaling above the provided threshold.
            if config['SCALE_THRESH'] != 'None':

                above_thresh = (ratio > config['SCALE_THRESH'])
                if np.sum(above_thresh) > 0:
                    
                    self._vprint("Scaling threshold based on weight ratios.")

                    # Limit scaling to the 99th percentile of these ratios.
                    percentile = np.percentile(ratio[above_thresh], 99)
                    ratio_capped = np.minimum(ratio[above_thresh], percentile)

                    scaling[above_thresh] = (1 + (ratio_capped - 1) * (config['SCALE_MAX'] - 1) / 
                                            (percentile - 1))

            # Ring-median filter the image.
            filtered = self._clipped_ring_median_filter(sci, mask, config)
            
            # Mask sources iteratively in tiers.
            bitmask = self._mask_sources(filtered, bitmask, scaling, config, starting_bit = 1)
            source_mask = (bitmask != 0) 

            # Estimate the background using just unmasked regions
            if config["INTERPOLATOR"] == 'IDW':
                bkg = self._estimate_background_IDW(sci, source_mask, config)
            else:
                bkg = self._estimate_background(sci, source_mask, config)

            # Subtract the background.
            bkgd_subtracted = (sci - bkg.background).astype(np.float32)
            bkgd_subtracted = np.where(
                off_detector_mask, np.float32(0.), bkgd_subtracted).astype(np.float32)

            # Evaluate the bias under all sources
            self._vprint("Bias under bright sources:")
            bias, sig = self._evaluate_bias(bkg.background, off_detector_mask, source_mask)
            hdr[f'BIAS_B'] = (bias, 'Bias under all sources.')
            hdr[f'SIG_B'] = (sig, 'Significance of bias under all sources.')

            # and just under the faintest sources.
            self._vprint("\nBias under fainter sources")
            faintmask = np.zeros(sci.shape, bool)
            for t in [len(config["TIER_NSIGMA"])-1, len(config["TIER_NSIGMA"])]:
                faintmask = faintmask | (np.bitwise_and(bitmask, 2**t) != 0)

            bias, sig = self._evaluate_bias(bkg.background, off_detector_mask, faintmask)
            hdr[f'BIAS_F'] = (bias, 'Bias under faint sources.')
            hdr[f'SIG_F'] = (sig, 'Significance of bias under faint sources.')

            # Save the file and append tier mask if needed.
            out_filename = sci_filename.replace(".fits", f"_{suffix}.fits")
            self._vprint(f'Saving background subtracted image to {out_filename}...')

            # Add parameters and function used to header.
            hdr['HIERARCH MASK_TYPE'] = 'Individual'
            for (key, value) in config.items():
                hdr[f'HIERARCH {key}'] = str(value)

            # Write primary HDU and mask.
            primary_hdu = fits.PrimaryHDU(bkgd_subtracted.astype(np.float32), header=hdr)
            if store_mask:
                wcs = WCS(hdr)
                mask_hdu = fits.ImageHDU(bitmask.astype(np.int32), header=wcs.to_header(), 
                                         name='TIERMASK')
                hdul = fits.HDUList([primary_hdu, mask_hdu])
            else:
                hdul = fits.HDUList([primary_hdu])
            hdul.writeto(out_filename, overwrite=True)
            hdul.close()
            
            bkgsub_filenames.append(out_filename)

        return bkgsub_filenames

    def merged_background(self, science_paths, bkgsub_images, parameters=None, WCS_filter=0,
                          suffix=None, merged_name=None):
        """
        Perform background subtraction using a mask merged from multiple 
        images.
        
        Arguments
        ---------
        science_paths (List[str])
            Filenames of science images to subtract the background
            from.
        bkgsub_images (List[str])
            Filenames of background subtracted images using individual 
            masks.
        parameters (None/dict)
            Key-value pairs overwritting parameters given in the config 
            file for this run.
        WCS_filter (int):
            Index into science_paths. Take the WCS information from this
            image.
        suffix (None, str)
            Suffix to append to the science filenames when saving 
            subtracted versions.
        merged_name (str, None):
            Filename for the output merged source mask.
            If None, don't save.
        """

        self._vprint('Calculating background using merged mask:')

        # Process the inputs.
        if parameters is None:
            parameters = {}
            
        if len(science_paths) == 1:
            raise KeyError('Only one science image given so not possible to create a merged mask.')
        if len(science_paths) != len(bkgsub_images):
            raise KeyError('There should be corresponding images of each type.')
        if (WCS_filter >= len(science_paths)) or (WCS_filter < 0):
            raise ValueError(f'WCS_filter should index science images but has value {WCS_filter}'
                             f' for {len(science_paths)} images.')

        # Overwrite some parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.', 
                                  stacklevel=2)

        mask = None
        self._vprint('Generating mask...')

        # Iterate over each image and get the stored tiered mask.
        for i, bkgimage in enumerate(bkgsub_images):

            with fits.open(bkgimage) as hdu:

                # Get header information from specified filter.
                if i == WCS_filter:
                    wcs = WCS(hdu[0].header)

                input_tiermask = hdu['TIERMASK'].data
                this_source_mask = np.left_shift(np.right_shift(input_tiermask, 1), 1)

                # Merge the masks.
                if mask is None:
                    mask = this_source_mask
                else:
                    mask = mask | this_source_mask 

        merged_mask = mask.astype(bool)

        # Save the merged mask if requested.
        basedir = os.path.dirname(bkgsub_images[0])
        if merged_name is not None:
            if '.fits' not in merged_name:
                merged_name = f'{merged_name}.fits'
            if os.path.dirname(merged_name) != basedir:
                merged_name = f'{basedir}/{os.path.basename(merged_name)}'
            hduout = fits.PrimaryHDU(merged_mask.astype(np.uint32), header=wcs.to_header())
            hduout.writeto(merged_name, overwrite=True)

        # Run final background subtraction using this merged mask
        for (image, bkgimage) in zip(science_paths, bkgsub_images):
            self._vprint(f'Measuring final background for {bkgimage}...')

            # Get tiermask from bgk-subtracted image to get bordermask
            # specific to this image.
            with fits.open(bkgimage) as hdumask:
                bordermask = hdumask['TIERMASK'].data == 1 

            # Combine the merged and border masks.
            sourcemask = merged_mask | bordermask
            mask = sourcemask != 0

            # Measure the background.
            sci, hdr = fits.getdata(image, header = True)
            sci = np.asarray(sci, dtype=np.float32)
            wcs = WCS(hdr)

            if config["INTERPOLATOR"] == 'IDW':
                bkg = self._estimate_background_IDW(sci, mask, config)
            else:
                bkg = self._estimate_background(sci, mask, config)

            bkgsub = (sci - bkg.background).astype(np.float32)
            bkgsub = np.where(bordermask, np.float32(0.), bkgsub).astype(np.float32)
            bkgsub = np.where(sci == 0, 0, bkgsub)

            # Add parameters and function used to header.
            hdr['HIERARCH MASK_TYPE'] = 'Merged'
            for (key, value) in config.items():
                hdr[f'HIERARCH {key}'] = str(value)

            # Save the new image.
            self._vprint(f'Saving background subtracted image to {bkgimage}...')

            if suffix == None:
                fits.writeto(bkgimage, bkgsub.astype(np.float32), header=hdr, overwrite=True)
            else:
                fits.writeto(image.replace(".fits", f"_{suffix}.fits"), bkgsub.astype(np.float32),
                             header=hdr, overwrite=True) 

        return        

    def full_background(self, science_paths, weight_paths, parameters=None, suffix='bkgsub', 
                        suffix_merged='merged', WCS_filter=0, merged_name=None):
        """
        Perform iterative source masking on individual images and 
        measure final background from a merged mask.
        
        Arguments
        ---------
        science_paths (List[str])
            Filenames of science images from which to subtract the merged 
            background.
        weight_paths (List[str])
            Filenames of the corresponding weight images.
        parameters (None/dict)
            Key-value pairs overwritting parameters given in the config
            file.
        suffix (str)
            Suffix to append to the science filenames when saving
            individual backgrounds.
        suffix_merged (str)
            Suffix to append to the science filenames when saving
            merged backgrounds.
        WCS_filter (int)
            Index into science_paths. The merged mask will borrow 
            WCS information from this image.
        merged_name (str/None)
            The filepath to save the merged mask to. If None, don't save.
        """

        # Measure the individual backgrounds.
        bkgsub_images = self.individual_background(science_paths, weight_paths, 
                                                   parameters, suffix, True)

        # Measure the merged background.
        self.merged_background(science_paths, bkgsub_images, parameters, WCS_filter,
                               suffix_merged, merged_name)

        return
    
def _block_sum_masked(img, mask, N):
    """
    Return block-sums of image and mask validity for NxN blocks.

    Arguments
    ---------
    img (numpy.ndarray)
        The 2D image to be summed in blocks.
    mask (numpy.ndarray)
        A 2D image mask where True regions are to be masked.
    N (int)
        The block size (NxN) to sum over.
    
    Returns
    -------
    block_sums (numpy.ndarray)
        The 1D array of block sums for valid blocks.
    """

    # Trim the image and mask to be divisible by N.
    ny, nx = img.shape
    ny_trim = ny - ny % N
    nx_trim = nx - nx % N
    
    img_crop = img[:ny_trim, :nx_trim].astype(np.float32)
    mask_crop = mask[:ny_trim, :nx_trim]
    
    # Reshape.
    img_blocks = img_crop.reshape(ny_trim//N, N, nx_trim//N, N)
    mask_blocks = mask_crop.reshape(ny_trim//N, N, nx_trim//N, N)
    
    # Sum the image within each block
    block_sums = img_blocks.sum(axis=(1, 3), dtype=np.float32)
    
    # Return only valid block sums
    valid_blocks = (mask_blocks.sum(axis=(1, 3)) == 0)
    
    return block_sums[valid_blocks]

def block_validate(science_path, bkgsub_path, weight_path, mask_path=None, max_block=101):
    """
    Validate background subtraction by comparing the standard deviation 
    of block summed images to the ideal.

    Arguments
    ---------
    science_path (str)
        Filename of the original science image.
    bkgsub_path (str)
        Filename of the background subtracted image.
    weight_path (str)
        Filename of the weight image corresponding to the science image.
    mask_path (str/None)
        Filename of the merged background mask. If None, use individual 
        mask stored in the bkgsub image.
    max_block (int)
        The maximum block size to test when validating the background.

    Returns
    -------
    fig (matplotlib.figure.Figure)
        The figure object containing the validation plot.
    ax (matplotlib.axes.Axes)
        The axes object containing the validation plot.
    """

    # Block sizes
    N_vals = np.arange(1, int(max_block))

    # Construct a detector mask.
    with fits.open(weight_path) as hdul:
        weight = np.asarray(hdul[0].data, dtype=np.float32)
        mask = (weight <= 0) | (~np.isfinite(weight))

        # also mask sources.
        if mask_path is not None:
            with fits.open(mask_path) as hdul:
                mask |= (hdul[0].data > 0)
        else:
            with fits.open(bkgsub_path) as hdul:
                mask |= (hdul[1].data > 0)

        # Compute the ideal rms.
        ideal_rms = np.nanmean(1 / np.sqrt(weight[~mask]))
    del weight

    # For the native image, compute the block sums and 
    # their standard deviation.
    std_per_N = []
    with fits.open(science_path) as hdul:
        img = np.asarray(hdul[0].data, dtype=np.float32)

        for N in N_vals:
            valid_block_sums = _block_sum_masked(img, mask, N)
            if valid_block_sums.size > 0:
                std = np.nanstd(valid_block_sums)
                std_per_N.append(std / N)
            else:
                std_per_N.append(np.nan)

    # Do the same for the background subtracted image.
    std_per_N_merged = []
    with fits.open(bkgsub_path) as hdul:
        img = np.asarray(hdul[0].data, dtype=np.float32)

        for N in N_vals:
            valid_block_sums = _block_sum_masked(img, mask, N)
            if valid_block_sums.size > 0:
                std = np.std(valid_block_sums)
                std_per_N_merged.append(std / N)
            else:
                std_per_N_merged.append(np.nan)

    # Plot the results.
    fig, ax = plt.subplots(1, 1)

    ax.scatter(N_vals, np.log10(std_per_N), marker='o', s=5, label = 'Original')
    ax.scatter(N_vals, np.log10(std_per_N_merged), marker='o', s=5, label = 'Subtracted')

    ax.set_xlabel("Block Size (NxN)")
    ax.set_ylabel('RMS / N')

    ax.axhline(np.log10(ideal_rms), ls='--', color='green', label='Ideal RMS')
    ax.legend()

    return fig, ax

def distance_validate(science_path, bkgsub_path, weight_path, max_dist=50):
    """
    Validate background subtraction by comparing the biweight mean of pixels
    at different distances from the top-level source mask.
    
    Arguments
    ---------
    science_path (str)
        Filename of the original science image.
    bkgsub_path (str)
        Filename of the background subtracted image.
    weight_path (str)
        Filename of the weight image corresponding to the science image.
    max_dist (int)
        The maximum distance from the mask in pixels to test.

    Returns
    -------
    fig (matplotlib.figure.Figure)
        The figure object containing the validation plot.
    ax (matplotlib.axes.Axes)
        The axes object containing the validation plot.
    """

    # Also mask sources.
    with fits.open(bkgsub_path) as hdul:
        img_bkgsub = np.asarray(hdul[0].data, dtype=np.float32)
        if len(hdul) < 2:
            raise KeyError('TIERMASK extension not found in bkgsub image; cannot run distance_validate.')
        tiermask = hdul[1].data

    with fits.open(weight_path) as hdul:
        weight = np.asarray(hdul[0].data, dtype=np.float32)
        combined_mask = (tiermask > 0) | (weight <= 0) | (~np.isfinite(weight))
    del weight

    with fits.open(science_path) as hdul:
        img_orig = np.asarray(hdul[0].data, dtype=np.float32)

    # Distance to nearest top-level mask pixel.
    distances = distance_transform_edt(~(tiermask == 2)).astype(np.float32)

    orig_means = []
    bkg_means = []
    dist_vals = np.arange(1, int(max_dist) + 1)

    # Compute the biweight mean of pixels at each distance from the mask.
    for d in dist_vals:
        sel = (np.floor(distances) == d) & (~combined_mask)
        vals_orig = img_orig[sel]
        vals_bkg = img_bkgsub[sel]

        if vals_orig.size > 0:
            m_orig = astrostats.biweight_location(vals_orig, ignore_nan=True)
        else:
            m_orig = np.nan

        if vals_bkg.size > 0:
            m_bkg = astrostats.biweight_location(vals_bkg, ignore_nan=True)
        else:
            m_bkg = np.nan

        orig_means.append(m_orig)
        bkg_means.append(m_bkg)

    # Plot the results
    fig, ax = plt.subplots(1, 1)

    ax.plot(dist_vals, orig_means, marker='o', linestyle='-', label='Original')
    ax.plot(dist_vals, bkg_means, marker='o', linestyle='-', label='Subtracted')

    ax.set_xlabel('Distance from top-level source mask (pixels)')
    ax.set_ylabel('Biweight mean')

    ax.legend()

    return fig, ax
