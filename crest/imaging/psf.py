# Adapted from the aperpy code available at https://github.com/astrowhit/aperpy
# See also Skelton+2014,  Whitaker+2019 and Weaver+2023.

import os
import copy
import yaml
import subprocess
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import zoom, binary_dilation

from astropy.io import fits
from astropy.table import Table, hstack
from astropy.nddata import block_reduce, Cutout2D
from astropy.stats import mad_std, sigma_clip
from astropy.convolution import convolve_fft
from astropy.modeling.fitting import LinearLSQFitter, FittingWithOutlierRemoval
from astropy.modeling.models import Linear1D
from astropy.visualization import ImageNormalize, LinearStretch

from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_com
from photutils.detection import find_peaks
from photutils.utils import circular_footprint

from crest.utils import _parallel_execute, _tile_worker, _construct_tiles, TempFileManager

class PSF():
    """
    Measure empirical PSFs from a set of images using the Aperpy /
    Skelton+2014 / Whitaker+2019 approach. Generate matching kernels
    with PyPHER and match images to a common PSF using tiles.

    """

    def __init__(self, config_path, verbose=True):
        """
        __init__ method for PSF class.
        
        Arguments
        ---------
        config_path (str)
            Path to .yml configuration file specifying parameters to use
            at each step.
        verbose (bool)
            If True, print progress messages.
        """

        # Load the config file.
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
        self.config = cfg
        self.config_filepath = config_path
        self.verbose = verbose
        
        # Initalise dictionaries for storing science filenames, PSFs and
        # kernels.
        self.filenames = {}
        self.PSFs = {}
        self.Kernels = {}

        self._temp_manager = TempFileManager()

    def _vprint(self, *args, **kwargs):
        """
        Print only when verbose output is enabled.
        """

        if self.verbose:
            print(*args, **kwargs)
    
    def _measure_curve_of_growth(self, image, radii, position=None):
        """
        Measure the Curve Of Growth (COG) of an image within 
        provided radii.
        
        Arguments
        ---------
        image (numpy.ndarray)
            The 2D image from which to measure the COG.
        radii (List[float])
            The radii in pixels within which to measure the flux.
        position (None/list[float]) 
            The x,y position of the source centre. If None, measure from 
            moments.

        Returns
        -------
        radii (List[float])
            The radii at which the enclosed energy was measured.
        cog (numpy.ndarray)
            The value of the COG at each radius.
        profile (numpy.ndarray)
            The value of the profile at each radius.
        """

        # Calculate the centroid of the source.
        if type(position) == type(None):
            position = centroid_com(image)

        # Perform aperture photometry to compute the COG.
        apertures = [CircularAperture(position, r=r) for r in radii]
        phot_table = aperture_photometry(image, apertures)

        cog = np.array([phot_table['aperture_sum_'+str(i)][0] for i in range(len(radii))])

        # Compute the normalised profile.
        area = np.pi*radii**2 
        area_cog = np.insert(np.diff(area), 0, area[0])

        profile = np.insert(np.diff(cog), 0, cog[0]) / area_cog 
        profile /= profile.max()

        return radii, cog, profile
    
    def _imshow(self, imgs, crosshairs=False, **kwargs):
        """
        Display a number of PSF images as a single figure.

        Arguments
        ---------
        imgs (List[numpy.ndarray])
            The 2D images to be plotted.
        crosshairs (bool)
            Should crosshairs be plotted at the images centres?
        **kwargs (dict)
            Additional keyword arguments for customising the figure.
            - width (float): The width of the figure in inches.
            - ncol (int): The number of columns in the figure.  
            - nsig (float): The number of MADs to use for normalisation.
            - stretch (ImageStretch): The normalisation stretch to use.
            - title (List[str]): A list of titles for each image.
            - norm_radius (float): A circle with this radius will be 
                                   plotted at the image centre.

        Returns
        -------
        fig (pyplot.figure)
            Pyplot figure object.
        ax (pyplot.axes)
            Pyplot axes object.
        """

        nimgs = len(imgs)

        # Set some plotting keywords with sensible defaults.
        width = kwargs.get('width', 30)
        ncol = kwargs.get('ncol', int(np.ceil(np.sqrt(nimgs))) + 1)
        nsig = kwargs.get('nsig', 5)
        stretch = kwargs.get('stretch', LinearStretch())

        # Set up the figure.
        nrow = int(np.ceil(nimgs / ncol))
        panel_width = width / ncol
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol,
                               figsize=(ncol * panel_width, nrow * panel_width))
        axes = np.atleast_1d(ax).ravel()

        # For each image.
        for idx, axi in enumerate(axes):
            if idx < nimgs:
                img = imgs[idx]
                finite_mask = (img != 0) & np.isfinite(img)

                # Normalise with the MAD.
                if (not img[finite_mask].size):
                    axi.set_axis_off()
                    continue
                sig = mad_std(img[finite_mask], ignore_nan=True)
                if (not np.isfinite(sig)) or (sig == 0):
                    axi.set_axis_off()
                    continue
                norm = ImageNormalize(img, vmin=-nsig * sig, vmax=nsig * sig, stretch=stretch)

                # Plot.
                axi.imshow(img, norm=norm, origin='lower', interpolation='nearest')

                # Draw a circle at image centre with given radius
                norm_radius = kwargs.get('norm_radius', None)
                cy = img.shape[0] // 2
                cx = img.shape[1] // 2
                if norm_radius is not None:
                    circ = Circle((cx, cy), norm_radius,
                                  edgecolor='red',
                                  linewidth=1.5, fill=False)
                    axi.add_patch(circ)
                axi.set_axis_off()
                if crosshairs:
                    axi.plot(cx, cy, color='red', marker='+', ms=10, mew=1)
            else:
                fig.delaxes(axi)

        title = kwargs.get('title')
        if title is not None:
            for ti, axi in zip(title, axes[:nimgs]):
                axi.set_title(ti)

        return fig, ax
    
    def _find_stars(self, sci, err, config, save_figs=True, 
                    science_path='science_image.fits', outdir='./'):
        """
        Identify stars in an image using peak finding and quality cuts.

        Arguments
        ---------
        sci (numpy.ndarray)
            The 2D image from which to find stars.
        err (numpy.ndarray)
            Corresponding error map.
        config (dict)
            The PSF config dictionary.
        save_figs (bool)
            Should diagnostic figures be saved?
        science_path (str)
            Base name of the science image. Used to name saved figures.
        outdir (str)
            The directory in which to save figures.

        Returns
        -------
        peaks[accept] (astropy.table.table.QTable)
            Information associated with each of the acceptable measured
            peaks.
        cutouts[accept] (numpy.ndarray):
            3D-array containing the cutouts of the acceptable peaks.
        """

        # Generate a catalogue of image peaks.
        peaks = find_peaks(sci, threshold=config["NSIG_THRESHOLD"]*err,
                           npeaks=config["N_PEAKS"])
        if len(peaks) == 0:
            raise RuntimeError('No peaks found. Try reducing the detection threshold.')
        
        peaks.rename_column('x_peak','x')
        peaks.rename_column('y_peak','y')

        # Will store offset from cutout centre and minimum pixel value
        peaks['x0'] = 0.0
        peaks['y0'] = 0.0
        peaks['minv'] = 0.0 

        # and the COG and profile within each radius.
        for ir in np.arange(len(config["RADII"])): peaks['r'+str(ir)] = 0.
        for ir in np.arange(len(config["RADII"])): peaks['p'+str(ir)] = 0.
        
        # For each peak.
        cutouts = []
        for index, peak in enumerate(peaks):

            # Create cutout around the measured position.
            co = Cutout2D(sci, (peak['x'], peak['y']), config["STAR_SIZE"], mode='partial').data
            cutouts.append(co)

            # Measure offset and minimum value.
            position = centroid_com(co)
            peaks['x0'][index] = position[0] - config["STAR_SIZE"] // 2
            peaks['y0'][index] = position[1] - config["STAR_SIZE"] // 2
            peaks['minv'][index] = np.nanmin(co)

            # Measure the the COG and profile.
            radii, cog, profile = self._measure_curve_of_growth(
                co, radii=np.array(config["RADII"]), position=position)
            for ir in np.arange(len(config["RADII"])): 
                peaks['r'+str(ir)][index] = cog[ir]
            for ir in np.arange(len(config["RADII"])): 
                peaks['p'+str(ir)][index] = profile[ir]

        cutouts = np.array(cutouts, dtype=np.float32)

        # Sselect only robust star candidates.

        # Magnitude within desired range.
        peaks['mag'] = (config["MAG_ZP"] - 2.5*np.log10(peaks[f'r{len(config["RADII"])-1}']))
        accept_mag = (peaks['mag'] < config["MAG_MIN"]) & (peaks['mag'] > config["MAG_MAX"])

        # Minimum value above threshold.
        accept_min = (peaks['minv'] > config["THRESHOLD_MIN"])
        # COG is well defined.
        accept_phot = ((np.isfinite(peaks[f'r{len(config["RADII"])-1}']))
                       & (np.isfinite(peaks['r0'])))
        # Offset from cutout centre is acceptable.
        accept_shift = ((
            np.sqrt(peaks['x0']**2 + peaks['y0']**2) < config["SHIFT_LIM"])
            & (np.abs(peaks['x0']) < np.sqrt(config["SHIFT_LIM"]))
            & (np.abs(peaks['y0']) < np.sqrt(config["SHIFT_LIM"])))

        # Ratio of COG at maxmium and middle value.
        ratio = (peaks[f'r{len(config["RADII"])-1}'] / peaks[f'r{len(config["RADII"])//2}'])

        # Bin these values and find the modal radius.
        bins = np.arange(config["RANGE"][0], config["RANGE"][1], config["WIDTH"])
        hist = np.histogram(ratio[(accept_mag)], bins=bins)

        i_mode = np.argmax(hist[0])
        ratio_mode = (hist[1][i_mode] + hist[1][i_mode+1]) / 2

        # Candidate must be within an acceptable range.
        accept_mode = ((ratio/ratio_mode > config["THRESHOLD_MODE"][0])
                       & (ratio/ratio_mode < config["THRESHOLD_MODE"][1]))
            
        # Full selection array.
        accept = (accept_mag & accept_min & accept_phot & accept_shift & accept_mode)

        # Fit the magnitude-ratio relation with outlier removal.
        fitter = FittingWithOutlierRemoval(
            LinearLSQFitter(), sigma_clip, sigma=config["SIGMA_FIT"], 
            niter=config['ITERATIONS_FIT'])
        lfit, outlier = fitter(Linear1D(), x=peaks['mag'][accept], y=ratio[accept])

        # Flag and remove outliers.
        i_outlier = np.where(accept)[0][outlier]
        accept[i_outlier] = False

        # Set new ids for the accepted objects.
        peaks['id'] = 1
        peaks['id'][accept] = np.arange(1, len(peaks[accept]) + 1)

        self._vprint(f' Selected {sum(accept)} candidate stars.')

        # Produce diagnostic figures.
        if save_figs == True:

            # Construct the main diagnostic plot.
            fig, ax = plt.subplots(2, 3, figsize=(14, 8))

            # Use this magnitude limit for all plots.
            mags = peaks['mag']
            mlim_plot = np.nanpercentile(mags, [5, 95]) + np.array([-2, 1])

            # All sources.
            ax[0, 0].scatter(mags, ratio, alpha=0.3, color='grey', s=2, label='All peaks')

            # Removed.
            ax[0, 0].scatter(mags[~accept_shift], ratio[~accept_shift], label='Bad shift', c='C1',
                        alpha=0.8, s=6)
            ax[0, 0].scatter(mags[i_outlier], ratio[i_outlier], label='Outlier', c='darkred',
                        alpha=0.8, s=6)
            
            # Accepted.
            ax[0, 0].scatter(mags[accept], ratio[accept], label='Accepted', c='C2', alpha=0.8, s=6)
            ax[0, 0].plot(np.arange(14,30), lfit(np.arange(14,30)), '--', c='k', alpha=0.3,
                     label='Slope={:.3f}'.format(lfit.slope.value))
            
            ax[0, 0].set_ylim(min(ratio) - 1, max(ratio) + 1)
            ax[0, 0].set_xlim(mlim_plot[0], mlim_plot[1])
            ax[0, 0].set_xlabel(rf'm$_{{\mathrm{{A}}{len(config["RADII"]) - 1}}}$')
            ax[0, 0].set_ylabel(f'A{len(config["RADII"]) // 2} / '
                       f'A{len(config["RADII"]) - 1}')
            ax[0, 0].legend()

            # The same plot, but zoomed in to the fit region.
            ratio_median = np.nanmedian(ratio[accept])

            ax[0, 1].scatter(mags, ratio, alpha=0.3, color='grey', s=2, label='All peaks')
            ax[0, 1].scatter(mags[~accept_shift], ratio[~accept_shift], label='Bad shift', c='C1',
                        alpha=0.8, s=6)
            ax[0, 1].scatter(mags[i_outlier], ratio[i_outlier], label='Outlier', c='darkred',
                        alpha=0.8, s=6)
            ax[0, 1].scatter(mags[accept], ratio[accept], label='Accepted', c='C2', alpha=0.8, s=6)
            ax[0, 1].plot(np.arange(14,30), lfit(np.arange(14,30)), '--', c='k', alpha=0.3,
                     label='Slope={:.3f}'.format(lfit.slope.value))
        
            ax[0, 1].set_ylim(ratio_median - 1, ratio_median + 1)
            ax[0, 1].set_xlim(mlim_plot[0], mlim_plot[1])
            ax[0, 1].set_xlabel(fr'm$_{{\mathrm{{A}}{len(config["RADII"]) - 1}}}$')
            ax[0, 1].set_ylabel((f'A{len(config["RADII"]) // 2} / A{len(config["RADII"]) - 1}'))

            # Histogram showing aperture ratios.
            bins = np.arange(config["RANGE"][0], config["RANGE"][1], config["WIDTH"])
            ax[0, 2].hist(ratio, bins=bins, alpha=0.7, color='grey')
            ax[0, 2].hist(ratio[accept], bins=bins, color='C2', alpha=1)

            ax[0, 2].set_xlabel((f'A{len(config["RADII"]) // 2} / A{len(config["RADII"]) - 1}'))
            ax[0, 2].set_ylabel('N')

            # Ratio of peak value to total
            ax[1, 0].scatter(
                config["MAG_ZP"] - 2.5*np.log10(peaks[f'r{len(config["RADII"])-1}'][accept]),
                (peaks['peak_value']/peaks[f'r{len(config["RADII"])-1}'])[accept], 
                color='C2', s=10, alpha=0.8)
            ax[1, 0].scatter(
                config["MAG_ZP"] - 2.5*np.log10(peaks[f'r{len(config["RADII"])-1}'])[i_outlier],
                (peaks['peak_value'] /peaks[f'r{len(config["RADII"])-1}'])[i_outlier],
                c='darkred', s=10, alpha=0.8)
            
            ax[1, 0].set_ylim(0, 1)
            ax[1, 0].set_xlabel(rf'm$_{{\mathrm{{A}}{len(config["RADII"]) - 1}}}$')
            ax[1, 0].set_ylabel(f'Peak / A{len(config["RADII"]) - 1}')

            # The offset of each source from the cutout centre.
            ax[1, 1].scatter(peaks['x0'][accept], peaks['y0'][accept], c='C2', alpha=0.8, s=10)
            ax[1, 1].scatter(peaks['x0'][i_outlier], peaks['y0'][i_outlier], c='darkred', 
                        alpha=0.8, s =10)
            
            ax[1, 1].set_xlim(-config["SHIFT_LIM"], config["SHIFT_LIM"])
            ax[1, 1].set_ylim(-config["SHIFT_LIM"], config["SHIFT_LIM"])
            ax[1, 1].set_xlabel('X-offset [pix]')
            ax[1, 1].set_ylabel('Y-offset [pix]')

            # The position of the sources in the image.
            ax[1, 2].scatter(peaks['x'][accept], peaks['y'][accept], c='C2', alpha=0.8, s=10)
            ax[1, 2].scatter(peaks['x'][i_outlier], peaks['y'][i_outlier], c='darkred', 
                        alpha=0.8, s=10)
            ax[1, 2].axis('scaled')
            ax[1, 2].set_xlabel('X [pix]')
            ax[1, 2].set_ylabel('Y [pix]')
            fig.tight_layout()

            # Save the diagnostic plot.
            outname = os.path.basename(science_path.replace(".fits", "_diagnostic.pdf"))
            fig.savefig(f'{outdir}/{outname}')
            plt.close()

            # Show all of the PSFs that will be used in stacking.
            title = ['{}: {:.1f} AB, ({:.1f}, {:.1f})'.format(ii, mm, xx, yy) for ii, mm, xx, yy in 
                     zip(peaks['id'][accept], mags[accept], peaks['x0'][accept], peaks['y0'][accept]
                         )]
            fig, ax = self._imshow(cutouts[accept], nsig=30, title=title)
            fig.tight_layout()

            outname = os.path.basename(science_path.replace(".fits", "_star_stamps.pdf"))
            fig.savefig(f'{outdir}/{outname}')
            plt.close()

        return peaks[accept], cutouts[accept]
    
    def _imshift(self, img, ddx, ddy, interpolation=cv2.INTER_CUBIC):
        """
        Recentre an image using an affine transformation.

        Arguments
        ---------
        img (numpy.ndarray)
            The image array to be recentred.
        ddx (float)
            Shift in the x direction.
        ddy (float)
            Shift in the y direction.
        interpolation (cv2 interpolator)
            Interpolation approach.
        
        Returns
        -------
        recentred (numpy.ndarray)
            The recentred image.
        """

        # Create the transformation matrix.
        M = np.float32([[1,0,ddx],[0,1,ddy]])

        # Apply the transformation.
        wxh = img.shape[::-1]
        recentred = cv2.warpAffine(img, M, wxh, flags=interpolation)

        return recentred
    
    def _centre(self, star_catalogue, cutouts, config, interpolation=cv2.INTER_CUBIC):
        """
        Recentre cutouts based on the centre of mass.

        Arguments
        ---------
        star_catalogue (astropy.table.table.QTable)
            Catalogue containing candidate star information.
        cutouts (numpy.ndarray)
            3D-array contaning star candidate cutouts.
        config (dict)
            The PSF config dictionary.
        interpolation (cv2 interpolator)
            Interpolation approach.

        Returns
        -------
        star_catalogue (astropy.table.table.QTable)
            Star catalogue updated with recentering information.
        cutouts (numpy.ndarray)
            The recentred cutouts.
        """

        # Get the window width and cutout centre.
        window = config['WINDOW']
        cw = window // 2
        c0 = config["PSF_SIZE"] // 2

        pos = []
        # Iterate over the different point sources.
        for i in np.arange(len(cutouts)):

            cutout = cutouts[i,:,:]

            # Measure the COM of the source within the window.
            co_window = Cutout2D(cutout, (c0,c0), window, mode='partial', fill_value=0).data
            co_window[~np.isfinite(co_window)] = 0
            x0, y0 = centroid_com(co_window)

            # Recentre the cutout.
            cutout = self._imshift(cutout, (cw-x0), (cw-y0), interpolation=interpolation)

            # Now measure COM using small window and positive definite in
            # case of strong ying-yang residuals.
            co_window = Cutout2D(cutout, (c0,c0), window, mode='partial', fill_value=0).data

            x1,y1 = centroid_com(co_window)
            x2,y2 = centroid_com(np.maximum(cutout,0))

            # Record difference in shift between the two methods.
            dsh = np.sqrt(((c0-x2)-(cw-x1))**2 + ((c0-y2)-(cw-y1))**2)
            pos.append([cw-x0,cw-y0,cw-x1,cw-y1,dsh])

            # Store the shifted cutout in place of the old one.
            cutout = np.ma.array(cutout, mask=~np.isfinite(cutout) | (cutout==0))
            cutouts[i,:,:] = cutout

        # Add these measurements to the star catalogue.
        star_catalogue = hstack(
            [star_catalogue, Table(np.array(pos), names=['x0','y0','x1','y1','dshift'])])
    
        return star_catalogue, cutouts
    
    def _measure(self, star_catalogue, cutouts, config):
        """
        Measure the photometric properties of stellar sources.

        Arguments
        ---------
        star_catalogue (astropy.table.table.QTable)
            Catalogue containing candidate star information.
        cutouts (numpy.ndarray)
            3D-array contaning star candidate cutouts.
        config (dict)
            The PSF config dictionary.

        Return
        ------
        star_catalogue (astropy.table.table.QTable)
            Star catalogue updated with photometry information.
        cutouts (numpy.ndarray)
            Star cutouts with saturated regions masked.
        """

        # Find the peak value in each cutout.
        peaks = np.array([cutout.max()for cutout in cutouts])
        peaks[~np.isfinite(peaks) | (peaks == 0)] = 0

        # Create a mask around the centre.
        norm_aper = CircularAperture((config["PSF_SIZE"]//2, config["PSF_SIZE"]//2),
                                     r=config["NORM_RADIUS"])
        norm_mask = Cutout2D(norm_aper.to_mask(), (config["NORM_RADIUS"], config["NORM_RADIUS"]),
                             self.config["PSF_SIZE"], mode='partial').data

        # Measure the flux within the norm radius for each star.
        phot = [aperture_photometry(cutout, norm_aper)['aperture_sum'][0] for cutout in cutouts]

        # Measurement on unmasked cutout (by casting to array) used for
        # saturation.
        sat =  [aperture_photometry(cutout, norm_aper)['aperture_sum'][0]
                for cutout in np.array(cutouts)]
        
        # Minimum unmasked value.
        cmin = [np.nanmin(cutout*norm_mask) for cutout in cutouts]

        # Combine with mask.
        for i in np.arange(len(cutouts)):
            cutouts[i].mask |= (cutouts[i] * norm_mask) < 0.0

        # Measure the RMS.
        rms_array = []
        for cutout in cutouts:
            rms = mad_std(cutout, ignore_nan=True)
            rms_array.append(rms)

        # Save some information to the catalogue.

        # Fraction of cutout that is masked.
        star_catalogue['frac_mask'] = 0.0
        # Fraction of flux that is within the normalisation radius.
        star_catalogue['phot_frac_mask'] = 1.0

        # New peak value
        star_catalogue['peak'] = peaks
        # and minimum value.
        star_catalogue['cmin'] = np.array(cmin)
        # Photometry measured in aperture.
        star_catalogue['phot'] = np.array(phot)
        # Is the cutout saturated?
        star_catalogue['saturated'] = np.int32(~np.isfinite(np.array(sat)))
        # The signal to noise ratio.
        star_catalogue['snr'] = 2*np.array(phot) / np.array(rms_array)

        return star_catalogue, cutouts
    
    def _select(self, star_catalogue, snr_lim=800, dshift_lim=3, mask_lim=0.99,
               phot_frac_mask_lim=0.99):
        """
        Select objects satisfying given conditions from the catalogue.

        Arguments
        ---------
        star_catalogue (astropy.table.table.QTable)
            Catalogue containing candidate star information.
        snr_lim (float)
            Minimum required SNR.
        dshift_lim (float)
            Maximum allowed difference in shift when recentering.
        mask_lim (float)
            Maximum allowed fraction of masked pixels.
        phot_frac_mask_lim (float)
            Minimum allowed ratio between flux measured within 
            normalisation radius before and after masking.

        Return
        ------
        star_catalogue (astropy.table.table.QTable)
            Star catalogue with updated selection column.
        """

        # Check which objects in the catalogue satisfy all conditions.
        accept = ((star_catalogue['dshift'] < dshift_lim) & (star_catalogue['snr'] > snr_lim)
                  & (star_catalogue['frac_mask'] < mask_lim)
                  & (star_catalogue['phot_frac_mask'] > phot_frac_mask_lim))
        star_catalogue['accept'] = np.int32(accept)

        star_catalogue['accept_shift'] = (star_catalogue['dshift'] < dshift_lim)
        star_catalogue['accept_snr'] = (star_catalogue['snr'] > snr_lim)
        star_catalogue['accept_frac_mask'] = (star_catalogue['frac_mask'] < mask_lim)
        star_catalogue['accept_phot_frac_mask'] = (star_catalogue['phot_frac_mask']
                                                   > phot_frac_mask_lim)

        # Format the columns to 3 DP.
        for c in star_catalogue.colnames:
            if 'id' not in c: star_catalogue[c].format = '.3g'

        return star_catalogue
    
    def _stack(self, star_catalogue, cutouts, masked_cutouts, config, save_figs=True,
              science_path='science_path', outdir ='./'):
        """
        Stack individual PSFs based on a pixelwise sigma clipped mean.

        Arguments
        ---------
        star_catalogue (astropy.table.table.QTable)
            Catalogue containing candidate star information.
        cutouts (numpy.ndarray)
            3D-array of unmasked star cutouts.
        masked_cutouts (numpy.ndarray)
            3D-array of star cutouts with saturated regions masked.
        config (dict)
            The PSF config dictionary.
        save_figs (bool)
            Save figure showing the masked cutouts used in the stack.
        science_path (str)
            Name of science image file. Only used for the figure name.
        outdir (str)
            Directory in which to save figure.

        Returns
        -------
        star_catalogue (astropy.table.table.QTable)
            Star catalogue updated with stacking information.
        masked_cutouts (numpy.ndarray)
            Star cutouts with masks updated by sigma clipping.
        stack (numpy.ndarray)
            Average 2D PSF measured by sigma-clipped stacking.
        """

        # Get indices of acceptable objects.
        i_accept = np.where(star_catalogue['accept'])[0]

        # Normalise by flux within the normalisation radius.
        norm = star_catalogue['phot'][i_accept]

        unmasked_cutouts = cutouts[i_accept].copy()
        for i in np.arange(len(unmasked_cutouts)): 
            unmasked_cutouts[i] = unmasked_cutouts[i] / norm[i]

        # Stack the images based on the pixel-wise sigma clipped mean.
        clipped_data = unmasked_cutouts.copy()

        # Perform required number of sigma clipping iterations.
        for i in range(config['MAX_ITERS']):
            clipped_data, lo, hi = sigma_clip(
                clipped_data, sigma=config['STACK_SIGMA'], maxiters=0, axis=0, masked=True, 
                grow=False, return_bounds=True)
            
            # Grow the mask
            for j in range(len(clipped_data.mask)): 
                clipped_data.mask[j,:,:] = binary_dilation(
                    clipped_data.mask[j,:,:], structure=circular_footprint(config['DILATE_RADIUS']),
                    iterations=1)

        for i in np.arange(len(unmasked_cutouts)):

            # Does object have its central pixel masked after clipping?
            star_catalogue['accept'][i_accept[i]] = (
                star_catalogue['accept'][i_accept[i]] and 
                ~clipped_data[i].mask[config["PSF_SIZE"]//2, config["PSF_SIZE"]//2])
            
            # Update the cutout mask.
            masked_cutouts[i_accept[i]].mask = clipped_data[i].mask
            mask = masked_cutouts[i_accept[i]].mask

            # What fraction of the pixels are masked after clipping?
            star_catalogue['frac_mask'][i_accept[i]] = np.size(mask[mask]) / np.size(mask)

        # Calculate the fraction of the flux remaining within the
        # normalisation radius after masking.
        aper = CircularAperture((config["PSF_SIZE"]//2, config["PSF_SIZE"]//2), 
                                r=config["NORM_RADIUS"])
        phot = [aperture_photometry(cutout, aper)['aperture_sum'][0] for cutout in masked_cutouts]
        star_catalogue['phot_frac_mask'] = phot/star_catalogue['phot']

        # Stack robust candidates.
        robust = ((star_catalogue['frac_mask'][i_accept] < config["MASK_FRAC_LIM"]) & 
                  (star_catalogue['phot_frac_mask'][i_accept] > config["PHOT_FRAC_LIM"]) &
                  (star_catalogue['accept'][i_accept] == True))
        
        self._vprint(f' Stacking {np.sum(robust)} robust candidates...')
        stack = np.mean(clipped_data[robust], axis=0)    
 
        i_accept = i_accept[robust]
        if save_figs == True:

            # Save the masked cutouts of all the stacked sources.
            title = ['{}: Mask - {:.1f}%'.format(ii, 100*frac) for ii, frac in 
                     zip(star_catalogue['id'][i_accept], star_catalogue['frac_mask'][i_accept])]
                    
            fig, ax = self._imshow(masked_cutouts[i_accept], title=title, nsig=30, 
                                   norm_radius=config['NORM_RADIUS'])

            outname = os.path.basename(science_path.replace(".fits", "_masked_cutouts.pdf"))
            fig.savefig(f'{outdir}/{outname}')
            plt.close()

        return star_catalogue, masked_cutouts, stack
    
    def measure_PSF(self, science_paths, error_paths, bands=None, parameters=None,
                    save_PSF=False, save_figs=False, outdir='./'):
        """
        Run star identification and stacking methods to obtain average
        PSF(s). Add generated PSF(s) to internal storage for later use.

        Arguments
        ---------
        science_paths (str, list)
            Paths to fits files containing science image from which to 
            identify stars.
        error_paths (str, list)
            Paths to corresponding error maps.
        bands (str, list, None)
            The photometric filters that these images correspond to.
            If None, use zero based indexing.
        parameters (dict)
            Key-value pairs overwriting parameters given in the config
            file.
        save_PSF (bool)
            Should the PSF be saved to a fits file?
        save_figs (bool)
            Should diagnostic figures be saved?
        outdir (str)
            Directory in which to save figures.
        """

        if parameters == None:
            parameters = {}

        # If single image given, convert to list.
        if type(science_paths) == str:
            science_paths = [science_paths]
            error_paths = [error_paths]
        if type(bands) == str:
            bands = [bands]
            
        # If bands are not defined, just use index.
        if bands == None:
            bands = np.arange(0, len(science_paths))

        # Overwrite some config parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.', 
                                  stacklevel=2)         

        for science_path, error_path, band in zip(science_paths, error_paths, bands):

            self._vprint(f'Measuring empirical PSF from {science_path}...')

            # Get images and corresponding header.
            sci, hdr = fits.getdata(science_path, header=True)
            err = fits.getdata(error_path)

            # Get information and cutouts of stars in the image.
            stars, cutouts = self._find_stars(sci, err, config, save_figs=save_figs,
                                             science_path=science_path, outdir=outdir)

            # Generate new cutouts at the full PSF size.
            psfs = np.array([Cutout2D(sci, (stars['x'][i], stars['y'][i]), config["PSF_SIZE"],
                                      mode='partial').data for i in np.arange(len(stars))])
            psfs_masked = np.ma.array(psfs, mask=~np.isfinite(psfs) | (psfs == 0))

            del sci, err

            # Move stars to the centre of the cutouts.
            stars, psfs_masked = self._centre(stars, psfs_masked, config)

            # Measure their flux and SNR.
            stars, psfs_masked = self._measure(stars, psfs_masked, config)

            # Stack, clip and mask objects with acceptable shift and SNR.
            stars = self._select(stars, config["SNR_LIM"], config["DSHIFT_LIM"], 0.99, 0.99)
            stars, psfs_masked, psf_average = self._stack(stars, psfs, psfs_masked, config, 
                                                           save_figs, science_path, outdir)

            # Normalise the PSF and remove mask.
            psf_average = np.array(psf_average)/np.sum(np.array(psf_average))

            for key, value in config.items():
                hdr[f'HIERARCH {key}'] = str(value)

            if save_PSF == True:
                outname = os.path.basename(science_path.replace(".fits", "_EPSF.fits"))
                fits.writeto(f'{outdir}/{outname}', psf_average, header=hdr, overwrite=True)

            # Save PDF image of the PSF.
            if save_figs == True:

                fig, ax = plt.subplots()
                sig = mad_std(psf_average[(psf_average != 0) & np.isfinite(psf_average)])
                norm = ImageNormalize(np.float32(psf_average), vmin=-50*sig, vmax=50*sig,
                                      stretch=LinearStretch())
                plt.imshow(psf_average, norm=norm, origin='lower', interpolation='none')
                ax.set_xticks([])
                ax.set_yticks([])
                outname = os.path.basename(science_path.replace(".fits", "_EPSF.pdf"))
                plt.savefig(f'{outdir}/{outname}')
                plt.close()
            
            # Store the PSF and corresponding filenames in the internal storage.
            if band != None:
                if band in self.PSFs.keys():
                    warnings.warn(f'Previously measured {band} PSF overwritten.', stacklevel=2)
                self.PSFs[band] = psf_average
                self.filenames[band] = [science_path, error_path]

        return
    
    def compare_COG(self, radii, bands=None):
        """
        Create a diagnostic plot, comparing the COGs of measured PSFs.
        
        Arguments
        ---------
        radii (array-like)
            The radii at which to measure the enclosed energy.
        bands (array-like)
            The PSFs of these bands will be plotted.
            If None, plot all measured PSFs.

        Returns
        -------
        fig (pyplot.figure)
            Pyplot figure object.
        ax (pyplot.axes)
            Pyplot axes object.
        """

        # If no bands indicated, use all.
        if bands == None:
            bands = self.PSFs.keys()

        # Plot the COG for each PSF.
        fig, ax = plt.subplots(1, 1, figsize=(3.78, 3.78))

        for band, psf in self.PSFs.items():
            if band in bands:
                radii, cog, profile = self._measure_curve_of_growth(psf, radii)
                ax.plot(radii, cog, label=band, alpha=0.8)
        
        ax.set_xlabel('Radius [pix]')
        ax.set_ylabel('Enclosed Energy')
        ax.legend()

        return fig, ax
    
    def plot_profile(self, source, target, radii_pix):
        """Plot the profiles of two PSFs.
        
        Arguments
        ---------
        source (np.ndarray)
            The first PSF for which to measure profile.
        target (np.ndarray)
            The second PSF for which to measure profile.
        radii_pix (array-like)
            The radii in pixels at which to measure the enclosed energy.
        
        Returns
        -------
        radii_pix (array-like)
            The radii in pixels at which the enclosed energy was
            measured.
        flux_source (numpy.ndarray)
            Profile of the first PSF.
        flux_target (numpy.ndarray)
            Profile of the second PSF.
        """

        # Use explicit aperture_sum column names (robust against column order)
        radii = np.array(radii_pix)
        center = (source.shape[1] // 2, source.shape[0] // 2)
        apertures = [CircularAperture(center, r=r) for r in radii]

        phot_table = aperture_photometry(source, apertures)
        flux_source = np.array([
            phot_table[f'aperture_sum_{i}'][0] for i in range(len(radii))
        ])

        phot_table = aperture_photometry(target, apertures)
        flux_target = np.array([
            phot_table[f'aperture_sum_{i}'][0] for i in range(len(radii))
        ])

        return radii, flux_source, flux_target
    
    def generate_kernel(self, target_band, bands=None, parameters=None, save_kernel=True,
                        save_figs=True, outdir='./'):
        """Create a kernel to match the measured PSF to a target PSF.

        Arguments
        ---------
        target_band (str)
            Key of the target PSF to match to as defined when measured.
        bands (list, None)
            The bands to generate matching kernels for. Will ignore
            target.
            If None, use measured PSFs.
        parameters (dict)
            Parameter key-value pairs to update in the config file just
            for this run.
        save_kernel (bool)
            Should the kernel be saved as a fits file?
        save_figs (bool)
            Should diagnostic figures be saved?
        outdir (str)
            The directory in which to store temporary and requested
            outputs.
        """

        self._vprint(f'Generating matching kernels for {target_band}.')

        if parameters == None:
            parameters = {}

        # Overwrite some config parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.',
                                  stacklevel=2)

        hdr = fits.Header()
        hdr['PIXSCALE'] = (config['PIXEL_SCALE'], 'Pixel scale in arcsec')

        # Get the target PSF and create a sub-dict in kernels.
        target = self.PSFs[target_band]
        self.Kernels[target_band] = {}
        
        # Oversample if required.
        if config['OVERSAMPLE'] > 1:
            self._vprint(f' Oversampling by {config["OVERSAMPLE"]}x...')
            target = zoom(target, config['OVERSAMPLE'])

        # Renormalise and save for passing to PyPHER.

        try:
            target /= target.sum()
            target_name = f'{outdir}/target.temp.fits'
            self._temp_manager.register(target_name)
            fits.writeto(target_name, target, header=hdr, overwrite=True)

            # If no bands indicated, measure kernel for all.
            if bands == None:
                bands = self.PSFs.keys()

            for band, source in self.PSFs.items():

                # Skip the target or omitted bands.
                if (band == target_band) or (band not in bands):
                    continue

                self._vprint(f' Working on {band}...')

                # Oversample if required.
                if config['OVERSAMPLE'] > 1:
                    source = zoom(source, config['OVERSAMPLE'])
        
                # Renormalise and save.
                source /= source.sum()
                source_name = f'{outdir}/source.temp.fits'
                self._temp_manager.register(source_name)
                fits.writeto(source_name, source, header=hdr, overwrite=True)

                # Filename of matching kernel. PyPHER will not overwrite.
                match_name = f'{outdir}/{band}_to_{target_band}_kernel.fits'
                log_name = match_name.replace('.fits', '.log')
                if os.path.isfile(match_name):
                    os.remove(match_name)
                self._temp_manager.register(match_name)
                self._temp_manager.register(log_name)

                # Run pypher
                pypherCMD = ['pypher', source_name, target_name, match_name, '-r', 
                             str(config["R_PARAMETER"]), '-s', str(config["ANGLE_SOURCE"]), '-t', 
                             str(config["ANGLE_TARGET"])]
                try:
                    res = subprocess.run(pypherCMD, check=True, capture_output=True, text=True, timeout=300)
                    if res.stderr:
                        self._vprint(res.stderr)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    stderr = getattr(e, 'stderr', '')
                    raise RuntimeError(
                        f'PyPHER failed for {band} -> {target_band}. {stderr}'.strip()) from e

                # Remove the temporary source file.
                self._temp_manager.delete(source_name)

                # Load the generated kernel and delete the PyPHER file.
                kernel = fits.getdata(match_name)
                self._temp_manager.delete(match_name)
                self._temp_manager.delete(log_name)

                # If oversampled, renormalise and overwrite saved and stored
                # kernels.
                if config['OVERSAMPLE'] > 1:

                    kernel = block_reduce(kernel, block_size=config['OVERSAMPLE'], func=np.sum)
                    kernel /= kernel.sum()
                    kernel = np.float32(np.array(kernel))

                # Store for later use
                self.Kernels[target_band][band] = kernel

                # and save to fits file if requested.
                if save_kernel == True:

                    # Create the header.
                    hdr_ = fits.Header()
                    hdr_['SOURCE'] = (band, 'Source PSF')
                    hdr_['TARGET'] = (target_band, 'Target PSF')
                    hdr_['OVERSAMP'] = (config["OVERSAMPLE"], 'Degree of oversampling')
                    hdr_['ANGLE_S'] = (config["ANGLE_SOURCE"], 'Angle of source PSF')
                    hdr_['ANGLE_T'] = (config["ANGLE_TARGET"], 'Angle of target PSF')

                    outname = os.path.basename(self.filenames[band][0]).replace(
                        ".fits", f"_kernel_{target_band}.fits")
                    fits.writeto(f'{outdir}/{outname}', kernel, header = hdr_, overwrite = True)

                # Construct the diagnostic figure.
                if save_figs == True:
                
                    fig, ax = plt.subplots(1, 7, figsize=(32, 4))

                    # Normalisation function.
                    sig = mad_std(source[(source != 0) & np.isfinite(source)])
                    norm = ImageNormalize(np.float32(source), vmin=-50*sig, vmax=50*sig,
                                        stretch=LinearStretch())

                    # Show the source, target and kernel images.
                    ax[0].set_title('Source: ' + band)
                    ax[0].imshow(source, norm=norm, origin='lower', interpolation='none')
                    ax[1].set_title('Target: ' + target_band)
                    ax[1].imshow(target, norm=norm, origin='lower', interpolation='none')
                    ax[2].set_title('Kernel')
                    ax[2].imshow(kernel, norm=norm, origin='lower', interpolation='none')

                    # Convolve the source with the kernel.
                    filt_psf_conv = convolve_fft(source, kernel)

                    # Show convolved PSF.
                    ax[3].set_title("Convolved " + band)
                    ax[3].imshow(filt_psf_conv, norm=norm, origin='lower', interpolation='none')

                    # Show the residual after convolution.
                    ax[4].set_title('Residual')
                    res = filt_psf_conv - target
                    ax[4].imshow(res, norm=norm, origin='lower', interpolation='none')

                    for ax_ in ax[0:5]:
                        ax_.set_xticks([])
                        ax_.set_yticks([])

                    # Show the COGs of convolved and target PSFs and the 
                    # ratio.
                    radii = np.arange(1, 40, 1)
                    centre = (target.shape[1] // 2, target.shape[0] // 2)
                    apertures = [CircularAperture(centre, r=r) for r in radii]

                    phot_table = aperture_photometry(filt_psf_conv, apertures)
                    flux_source = np.array([
                        phot_table[f'aperture_sum_{i}'][0] for i in range(len(radii))])
                    phot_table = aperture_photometry(target, apertures)
                    flux_target = np.array([
                        phot_table[f'aperture_sum_{i}'][0] for i in range(len(radii))])
                    
                    #r, pf, pt = self.plot_profile(filt_psf_conv, target, np.arange(1, 40, 1))
                    ax[6].plot(radii*self.config["PIXEL_SCALE"], flux_source/flux_target)
                    ax[6].set_xlabel('Radius [arcsec]')
                    ax[6].set_ylabel('EE convolved source / EE target')
                    ax[6].set_ylim(0.95, 1.05)

                    ax[5].plot(radii*self.config["PIXEL_SCALE"], flux_source, lw=3, label='Convolved source')
                    ax[5].plot(radii*self.config["PIXEL_SCALE"], flux_target, '--', alpha=0.7, lw=3,
                            label='Target')
                    ax[5].set_xlabel('Radius [arcsec]')
                    ax[5].set_ylabel('Enclosed energy')
                    ax[5].legend()
                    
                    fig.tight_layout()
                    outname = os.path.basename(self.filenames[band][0]).replace(
                        ".fits", f"_match_{target_band}_diagnostic.pdf")
                    fig.savefig(f'{outdir}/{outname}')
                    plt.close()

        except Exception as e:
            raise RuntimeError(
                f'Failed to generate matching kernels for target band {target_band}.') from e
        finally:
            self._temp_manager.cleanup()
        
        return
    
    def convolve_image(self, target_band, bands=None, parameters=None, outdir='./'):
        """
        Convolve images used to measure PSF with generated matching
        kernels.

        Arguments
        ---------
        target_band (str)
            The target band for convolution. Matching kernels must
            already be generated.
        bands (list, None)
            The bands on which to perform convolution. Will ignore
            target. If None, use all measured PSFs.
        parameters (dict)
            Parameters to update in the config file just for this run.
        outdir (str)
            Directory in which to store convolved images.
        """

        if parameters == None:
            parameters = {}

        # Check that matching kernels have been generated.
        if target_band not in self.Kernels.keys():
            raise KeyError(f'Matching kernels for {target_band} have not been generated.'
                           f'Run generate_kernel first.')
                
        # Overwrite some config parameters just for this run.
        config = copy.deepcopy(self.config)
        for (key, value) in parameters.items():
                if key in config:
                    config[key] = value
                else:
                    warnings.warn(f'{key} is not a valid parameter. Continuing without updating.',
                                  stacklevel=2)

        # If no bands indicated, convolve all.
        if bands == None:
            bands = self.PSFs.keys()

        # For each band.
        for band, kernel in self.Kernels[target_band].items():

            if band not in bands:
                continue

            self._vprint(f'Matching {band} to {target_band}...')

            # Load in the science and error images.
            sci, sci_hdr = fits.getdata(self.filenames[band][0], header=True)
            err, err_hdr = fits.getdata(self.filenames[band][1], header=True)

            mask = (np.isnan(err) | ~np.isfinite(err) | (err <= 0) | np.isnan(sci) | 
                    ~np.isfinite(sci))

            n_tiles = config.get('N_TILES', 1)
            if n_tiles <= 1:
                convolved_sci = convolve_fft(sci, kernel, allow_huge=True, 
                                             preserve_nan=True, mask=mask)
                convolved_err = convolve_fft(err, kernel, allow_huge=True, 
                                             preserve_nan=True, mask=mask)

            else:

                # Compute halo size.
                kh = np.ceil(max(kernel.shape) // 2)

                # Split into tiles.
                ny, nx = sci.shape
                convolved_sci = np.zeros_like(sci, dtype=np.float32)
                convolved_err = np.zeros_like(sci, dtype=np.float32)

                slices = _construct_tiles((ny, nx), int(n_tiles), kh)

                # Prepare science tasks for each tile.
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

                # Do the same for the error map.
                tasks = []
                for s in slices:
                    y0, y1, x0, x1, e0, e1, f0, f1 = s
                    block = err[e0:e1, f0:f1]
                    mask_block = mask[e0:e1, f0:f1]        
                    tasks.append({'block': block, 'mask': mask_block,
                                  'kernel': kernel, 'slices': s})
                results = _parallel_execute(_tile_worker, tasks, workers)
                for res in results:
                    y0, y1, x0, x1, interior = res
                    convolved_err[y0:y1, x0:x1] = interior
            
            # Add some header keywords.
            sci_hdr['KERNEL'] = (target_band, 'Convolved with this kernel')
            err_hdr['KERNEL'] = (target_band, 'Convolved with this kernel')

            # Ensure off detector region values don't change.
            convolved_sci = np.where(mask, sci, convolved_sci)
            convolved_err = np.where(mask, err, convolved_err)

            # Save the convloved images.
            outname = os.path.basename(self.filenames[band][0]).replace(
                ".fits", f"_match{target_band}.fits")
            fits.writeto(f'{outdir}/{outname}', convolved_sci.astype(np.float32), sci_hdr, overwrite=True)
            outname = os.path.basename(self.filenames[band][1]).replace(
                ".fits", f"_match{target_band}.fits")
            fits.writeto(f'{outdir}/{outname}', convolved_err.astype(np.float32), err_hdr, overwrite=True)

            self._vprint(' Done.')
        
        return