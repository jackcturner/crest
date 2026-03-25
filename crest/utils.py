import os
import signal
import sys
import atexit

import numpy as np
import threading
from multiprocessing.dummy import Pool
from astropy.convolution import convolve_fft
from scipy.ndimage import median_filter
from scipy.ndimage import binary_dilation
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_com
import scipy.stats as st

def _parallel_execute(func, tasks, workers):
    """
    Execute tasks in a process pool.

    Arguments
    ---------
    func (callable)
        Function to execute for each task. 
        Should take a single task dictionary as an argument.
    tasks (list of dict)
        List of task dictionaries to process.
    workers (int)
        Number of worker processes to use.

    Returns
    -------
    results (list)
        List of results returned by func for each task.
    """

    def _run_task(t):
        return func(t)
    
    results = []
    with Pool(processes=workers) as pool:
        async_results = [pool.apply_async(_run_task, (t,)) for t in tasks]
        for ar in async_results:
            res = ar.get()
            if res is not None:
                results.append(res)
    return results

def _tile_worker(args):
    """
    Unified worker for processing imaging tiles. Mode will be determined 
    by the presence of specific keys.

    Arguments
    ---------
    args (dict)
        Dictionary containing the following keys:
        - block (np.ndarray): Tile cutout with extended halo.
        - slices (tuple): Coordinates defining the tile and halo slices.
                          (y0, y1, x0, x1, e0, e1, f0, f1)
                
        If running in ring-median mode:
        - ring_footprint (np.ndarray): Array defining the filtering 
                                       footprint.
        - filled_block (np.ndarray): 2D array of the tile with halo after 
                                     filling NaNs with zeros.

        If running in dilation mode:
        - dilate_footprint (np.ndarray): Array defining the dilation 
                                         footprint.
        
        If running in convolution mode:
        - kernel (np.ndarray): 2D array defining the convolution kernel.
        - mask (np.ndarray/None): Boolean array indicating pixels to 
                                  ignore during convolution.
    
    Returns    
    -------
    y0, y1, x0, x1, interior (tuple)
        The processed tile and its position without the halo.
    """

    block = args['block']
    (y0, y1, x0, x1, e0, e1, f0, f1) = args['slices']

    # Ring-median mode.
    if 'ring_footprint' in args:
        filled_block = args['filled_block']
        footprint = args['ring_footprint']
        filtered_block = median_filter(filled_block, footprint=footprint)
        out_block = block - filtered_block

    # Dilation mode.
    elif 'dilate_footprint' in args:
        footprint = args['dilate_footprint']
        out_block = binary_dilation(block, structure=footprint)

    # Otherwise convolution mode.
    else:
        kernel = args['kernel']
        mask = args.get('mask', None)
        out_block = convolve_fft(block, kernel, allow_huge=True, preserve_nan=True, mask=mask)

    # Extract the interior region to stitch back together.
    iy0 = y0 - e0
    iy1 = iy0 + (y1 - y0)
    ix0 = x0 - f0
    ix1 = ix0 + (x1 - x0)
    interior = out_block[iy0:iy1, ix0:ix1]

    return (y0, y1, x0, x1, interior)

def _construct_tiles(shape, num_tiles, halo):
    """
    Generate tile slices given a desired total number of tiles.

    Arguments
    ---------
    shape (tuple)
        Shape of the image to be tiled (ny, nx).
    num_tiles (int)
        Desired total number of tiles.
    halo (int)
        Number of pixels to extend each tile in each direction.
    
    Returns
    -------
    slices (list of tuples)
        List of tuples defining the tile (y0, y1, x0, x1) and halo 
        extended slices (e0, e1, f0, f1).
    """

    # Compute a near-square grid.
    ny, nx = shape
    n_x = int(np.ceil(np.sqrt(num_tiles)))
    n_y = int(np.ceil(num_tiles / max(1, n_x)))

    # Create integer edges that span the image.
    # Ensure last edge == ny/nx.
    y_edges = np.round(np.linspace(0, ny, n_y + 1)).astype(int)
    x_edges = np.round(np.linspace(0, nx, n_x + 1)).astype(int)

    # Loop over tiles and compute slices.
    # e and f are the extended slices that include the halo.
    slices = []
    for iy in range(n_y):
        y0 = int(y_edges[iy]); y1 = int(y_edges[iy + 1])
        for ix in range(n_x):
            x0 = int(x_edges[ix]); x1 = int(x_edges[ix + 1])
            e0 = max(0, y0 - halo)
            e1 = min(ny, y1 + halo)
            f0 = max(0, x0 - halo)
            f1 = min(nx, x1 + halo)
            slices.append((y0, y1, x0, x1, e0, e1, f0, f1))

    return slices

class TempFileManager:
    """
    Keep track of temporary files and remove them on code exit.
    """

    def __init__(self):
        """ 
        __init__ method for TempFileManager.
        """

        # Store temporary file paths here.
        self.temp_files = set()

        # Remove files on any kind of exit.
        atexit.register(self.cleanup)

        # Only register signal handlers when running in the main thread.
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGTERM, self.cleanup_on_signal)
                signal.signal(signal.SIGINT, self.cleanup_on_signal)
            except Exception:
                pass

    def register(self, path):
        """
        Register a temporary file for cleanup.

        Arguments
        ---------
        path (str)
            File path of the temporary file to be deleted later.
        """

        self.temp_files.add(path)

    def cleanup(self):
        """
        Delete all registered temporary files.
        """

        for file in self.temp_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f'Warning: Failed to delete {file} ({e})')

    def cleanup_on_signal(self, signum=None, frame=None):
        """
        Remove files on signal.
        """        

        # Perform cleanup of temp files first.
        self.cleanup()

        # Re-raise the signal.
        try:
            if signum == signal.SIGINT:
                signal.default_int_handler(signum, frame)
            else:
                sys.exit(0)
        except KeyboardInterrupt:
            raise

    def delete(self, path):
        """
        Delete a specific temporary file immediately and remove from registry.

        Arguments
        ---------
        path (str)
            File path of the temporary file to be deleted.
        """
        if os.path.exists(path):
            try:
                os.remove(path)
                self.temp_files.discard(path)
            except FileNotFoundError:
                self.temp_files.discard(path)
            except Exception as e:
                print(f'Warning: Failed to delete {path} ({e})')
        else:
            print(f'Warning: File {path} does not exist and cannot be deleted.')

def measure_curve_of_growth(image, radii, position=None):
    """
    Measure the Curve Of Growth of an image based on provided radii.
    
    Arguments
    ---------
    image (numpy.ndarray)
        The 2D image from which to measure the COG.
    radii (List[float])
        The radii in pixels at which to measure the enclosed flux.
    position (None, list[float]) 
        The x,y position of the source centre. If None, measure from 
        moments.

    Returns
    -------
    radii (List[float])
        The radii at which the enclosed energy was measured.
    cog (numpy.ndarray)
        The value of the COG at each radius.
    """

    # Calculate the centroid of the source.
    if type(position) == type(None):
        position = centroid_com(image)

    # Calculate cumulative aperture fluxes
    apertures = [CircularAperture(position, r = r) for r in radii]

    phot_table = aperture_photometry(image, apertures)
    cog = np.array([phot_table['aperture_sum_'+str(i)][0] for i in range(len(radii))])

    return radii, cog

def poisson_confidence_interval(counts, p=0.68):
    """ 
    Return the upper and lower Poisson confidence limits on a count.
    
    Arguments
    ---------
    counts (numpy.ndarray)
        1D array of counts.
    p (float)
        The confidence limit to return.
        
    Returns
    -------
    intervals (numpy.ndarray)
        2D array of upper and lower confidence limits.
    """
    
    lower = []
    upper = []

    for n in counts:
    
        if n>0:   
            interval=(st.chi2.ppf((1.-p)/2.,2*n)/2.,st.chi2.ppf(p+(1.-p)/2.,2*(n+1))/2.)       
        
        else:
            
            #this bit works out the case for n=0
            
            ul=(1.-p)/2.
            
            prev=1.0
            for a in np.arange(0.,5.0,0.001):
            
                cdf=st.poisson.cdf(n,a)
            
                if cdf<ul and prev>ul:
                    i=a
            
                prev=cdf
            
            interval=(0.,i)
        
        lower.append(interval[0])
        upper.append(interval[1])
    
    intervals = np.column_stack([lower, upper])
    
    return intervals