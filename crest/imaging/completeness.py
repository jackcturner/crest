import os
import h5py

import numpy as np
import math

from astropy.io import fits
from astropy.table import Table

from scipy.spatial import cKDTree
import scipy.ndimage as nd

from crest.cataloguing import SourceExtractor
from crest.imaging.masking import create_edge_mask
from crest.utils import TempFileManager, poisson_confidence_interval, _parallel_execute

def find_matches(small_cat, large_cat):
    """
    Return indicies into a larger catalogue from X-Y matches to a smaller
    catalogue. Matches are not unique.
    
    Arguments
    ---------
    small_cat (numpy.ndarray)
        The X-Y coordinates of sources in the smaller catalogue.
    large_cat (numpy.ndarray)
        The X-Y coordinates of sources in the larger catalogue.
        
    Returns
    -------
    indices (numpy.ndarray)
        For each object in small_cat, the index of the closest match in 
        large_cat.
    distances (numpy.ndarray)
        The distance between matches in pixels."""

    # Create KD-tree for the larger catalogue.
    large_tree = cKDTree(large_cat)
    
    # Query the KD-tree with the positions from the smaller catalogue.
    distances, indices = large_tree.query(small_cat)

    # Sort indices and reorder distances accordingly.
    sorted_indices = np.argsort(indices)
    indices = indices[sorted_indices]
    distances = distances[sorted_indices]
    
    return indices, distances

def _completeness_worker(task):
    """
    Worker to insert sources into a mosaic and measure the recovery
    rate.

    Arguments
    ---------
    task (dict)
        A dictionary containing key-value pairs required to execute
        the task. See measure_completeness for details.

    Returns
    -------
    bin_idx (int)
        The index of the bin for which this task was executed.
    n_recovered (int)
        The number of sources recovered in this task.
    """

    # Load the image within which to insert sources.
    img, hdr = fits.getdata(task['science_path'], header=True)

    # Create a table to store the source properties.
    source_table = Table(names=['INDEX', 'X_IMAGE', 'Y_IMAGE', 'FLUX'])

    # Sample possible source positions.
    rng = np.random.default_rng(task['seed'])
    sampled = rng.choice(task['unmasked_y'].size, size=task['n_sources'], replace=False)
    sampled_y = task['unmasked_y'][sampled]
    sampled_x = task['unmasked_x'][sampled]

    # Insert a source at each position with a flux sampled from the bin.
    for i, (row, col) in enumerate(zip(sampled_y, sampled_x)):

        flux_psf = rng.uniform(task['bin_low'], task['bin_high'])
        psf_ = task['psf'] * flux_psf

        # Clip image bounds and apply matching PSF slice near edges.
        x0 = row - psf_.shape[0]//2
        x1 = x0 + psf_.shape[0]
        y0 = col - psf_.shape[1]//2
        y1 = y0 + psf_.shape[1]

        ix0 = max(x0, 0)
        ix1 = min(x1, img.shape[0])
        iy0 = max(y0, 0)
        iy1 = min(y1, img.shape[1])

        if (ix0 >= ix1) or (iy0 >= iy1):
            continue

        px0 = ix0 - x0
        px1 = px0 + (ix1 - ix0)
        py0 = iy0 - y0
        py1 = py0 + (iy1 - iy0)

        img[ix0:ix1, iy0:iy1] += psf_[px0:px1, py0:py1]
        source_table.add_row([i, col, row, flux_psf])

    # Track temporary files.
    temp_manager = TempFileManager()

    try:

        # Save the injected image.
        uid = f'bin{task["bin_idx"]}_mos{task["mosaic_idx"]}_seed{task["seed"]}'
        img_name = f'{task["outdir"]}/{task["temp_name"]}_completeness_{uid}.fits'

        fits.writeto(img_name, img, hdr, overwrite=True)
        temp_manager.register(img_name)

        se_run = SourceExtractor(task['config'], task['sex_path'], verbose=False)
        cat = se_run.extract(img_name, task['weight_path'], parameters = task['parameters'], 
                             output=['FLUX_AUTO', 'FLUXERR_AUTO', 'X_IMAGE', 'Y_IMAGE'],
                             outdir=task['outdir'])
        temp_manager.register(cat)

        with h5py.File(cat) as f:

            # Match true to recovered sources.
            cat_xy = np.column_stack((f['photometry/X_IMAGE'][:], f['photometry/Y_IMAGE'][:]))
            syn_xy = np.column_stack((source_table['X_IMAGE'], source_table['Y_IMAGE']))

            indices, distances = find_matches(syn_xy, cat_xy)

            # Apply distance criterion.
            s = distances < task['offset']

            # If there are duplicate matches, keep the closest.
            unique_indices, unique_pos = np.unique(indices, return_inverse=True)
            duplicate_mask = np.zeros_like(indices, dtype=bool)

            for i in range(len(unique_indices)):
                duplicate_indices = np.where(unique_pos == i)[0]
                if len(duplicate_indices) > 1:
                    min_dist_idx = np.argmin(distances[duplicate_indices])
                    for j in range(len(duplicate_indices)):
                        if j != min_dist_idx:
                            duplicate_mask[duplicate_indices[j]] = True

            s[duplicate_mask] = False

            # Apply the distance criterion to indices and sort them.
            filtered_indices = indices[s]
            sorted_order = np.argsort(filtered_indices)
            sorted_indices = filtered_indices[sorted_order]

            # Apply flux and S/N criteria.
            flux = f['photometry/FLUX_AUTO'][sorted_indices]
            err = f['photometry/FLUXERR_AUTO'][sorted_indices]
            sn = flux / err

            true_flux = source_table['FLUX'][s]
            true_flux = true_flux[sorted_order]

            s_ = ((flux / true_flux < task['flux_limits'][1]) & 
                  (flux / true_flux > task['flux_limits'][0]) &
                  (sn > task['min_sn']))

        return task['bin_idx'], int(np.sum(s_))

    finally:
        temp_manager.cleanup()

def measure_completeness(science_path, weight_path, psf_path, bins, config_path, sex_path='sex', 
                         mask_path=None, dilate=0, border=50, min_sources=1500, density=5, 
                         offset=6.66, flux_limits=[0.5, 1.5], min_sn=2, n_workers=1, 
                         random_seed=None, outdir='./', ):
    """
    Measure the completeness of an image by inserting synthetic sources
    in provided flux bins. 

    Arguments
    ---------
    science_path (str)
        Path to fits image for which to measure completeness.
    weight_path (str)
        Path to the fits image to use for weighting. Can be an error map
        assuming the config value is set appropriately.
    psf_path (str)
        Path to the fits PSF image to insert as a synthetic source.
    bins (numpy.ndarray)
        1D array of flux bin edges in image units.
    config_path (str)
        Path to the SourceExtractor configuration file to use.
    sex_path (str)
        Path to the SourceExtractor executable.
    mask_path (str/None)
        Path to a fits source mask. 
        If None, compute with SourceExtractor.
    dilate (int)
        The number of pixels by which to dilate the source mask.
    border (int)
        The number of pixels by which to mask the border.
    min_sources (int)
        The minimum number of sources to inject.
    density (int)
        Insert one source per this many unmasked pixels.
    offset (float)
        The maximum offset in pixels allowed between the inserted and
        recovered source.
    flux_limits (List[float])
        The minimum and maximum flux ratio of recovered and inserted 
        sources.
    min_sn (float)
        The minimum S/N of recovered sources.
    n_workers (int)
        Number of workers to use for parallel execution.
    random_seed (int/None)
        The random seed from which to generate random streams for 
        each mosaic.
    outdir (str)
        Directory in which to save temporary files.

    Returns
    -------
    complete (List[float])
        The estimated completeness within each bin.
    error (List[numpy.ndarray])
        The 1-sigma upper and lower confidence limits.
    """
    
    temp_name = os.path.basename(science_path).removesuffix('fits')
    print(f'Measuring completeness in {temp_name}...')

    if density <= 0:
        raise ValueError('density must be > 0.')
    if min_sources <= 0:
        raise ValueError('min_sources must be > 0.')

    # Collate low/high bin edges.
    bin_low = np.minimum(bins[:-1], bins[1:])
    bin_high = np.maximum(bins[:-1], bins[1:])
    n_bins = bin_low.size

    # Create an edge mask to remove noisy regions.
    unmasked = create_edge_mask(science_path, n_pixels=border).data == 0

    # If no mask provided, generate a segmentation map.
    delete = False
    if mask_path == None:

        mask_path = f'{outdir}/{temp_name}_completeness_mask_temp.fits'
        delete = True

        parameters = {}
        parameters['CHECKIMAGE_TYPE'] = 'SEGMENTATION'
        parameters['CHECKIMAGE_NAME'] = mask_path
        parameters['VERBOSE_TYPE'] = 'QUIET'

        se_run = SourceExtractor(config_path, sex_path, verbose=False)
        cat = se_run.extract(science_path, weight_path, parameters, outdir=outdir)
        os.remove(cat)

    # Apply all the additional masks.
    with fits.open(weight_path) as wht:
            unmasked = unmasked & ((wht[0].data > 0) & (~np.isnan(wht[0].data)) & 
                                   (np.isfinite(wht[0].data)))
    with fits.open(mask_path) as seg:
            seg_mask = (seg[0].data != 0)
            if dilate > 0:
                seg_mask = nd.binary_dilation(seg_mask, iterations=dilate)
            unmasked = unmasked & (seg_mask == 0)
    if delete == True:
        os.remove(mask_path)

    # Store unmasked coordinates.
    unmasked_y, unmasked_x = np.where(unmasked)
    unmasked_y = unmasked_y.astype(np.int32, copy=False)
    unmasked_x = unmasked_x.astype(np.int32, copy=False)

    # Number of sources that can be placed in each image.
    n_sources = math.ceil(np.sum(unmasked)/density) 
    if (n_sources < 1):
        raise ValueError('No unmasked pixels available for source injection.')
    
    # The number of mosaics needed for minimum sources.
    n_img_max = math.ceil(min_sources/n_sources)  
    # The total number of sources to be placed.
    total_sources = n_sources*n_img_max

    print(f'Placing {n_sources} synthetic sources in {n_img_max} mosaics, ' 
          f'totalling {total_sources}.')

    # Before running SE fix some parameters.
    run_parameters = {}
    run_parameters['CHECKIMAGE_TYPE'] = 'NONE'
    run_parameters['EMPIRICAL'] = False
    run_parameters['TO_FLUX'] = 1
    run_parameters['VERBOSE_TYPE'] = 'QUIET'

    # Load and normalise the psf.
    psf = fits.getdata(psf_path)
    psf /= np.sum(psf)

    # Spawn one independent random stream per task.
    n_total_tasks = n_bins * n_img_max
    parent_seed = np.random.SeedSequence(random_seed)
    child_seeds = parent_seed.spawn(n_total_tasks)

    # Construct the tasks.
    tasks = []
    for bin_idx, (bin_low_, bin_high_) in enumerate(zip(bin_low, bin_high)):
        for mosaic_idx in range(n_img_max):

            seed_idx = bin_idx * n_img_max + mosaic_idx
            task_seed = int(child_seeds[seed_idx].generate_state(1, dtype=np.uint64)[0])

            tasks.append({'science_path': science_path, 'weight_path': weight_path, 'psf': psf,
                          'unmasked_y': unmasked_y, 'unmasked_x': unmasked_x, 
                          'n_sources': n_sources, 'bin_low': bin_low_, 'bin_high': bin_high_,
                          'offset': offset, 'flux_limits': flux_limits, 'min_sn': min_sn, 
                          'outdir': outdir, 'temp_name': temp_name, 'bin_idx': bin_idx,
                          'mosaic_idx': mosaic_idx, 'seed': task_seed, 'config': config_path, 
                          'sex_path': sex_path, 'parameters': dict(run_parameters)})

    # Execute and aggregate results back to each bin.
    recovered = _parallel_execute(_completeness_worker, tasks, n_workers)

    n_recovered_per_bin = np.zeros(n_bins, dtype=int)
    for result in recovered:
        bin_idx, n_recovered = result
        n_recovered_per_bin[bin_idx] += int(n_recovered)

    # Compute completeness and confidence intervals.
    complete = []
    error = []
    for n_recovered in n_recovered_per_bin:
        complete.append(n_recovered/total_sources)
        error.append(poisson_confidence_interval([n_recovered])/total_sources)

    return complete, error