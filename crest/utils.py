import os
import signal
import sys
import atexit

import numpy as np
from multiprocessing.dummy import Pool
from astropy.convolution import convolve_fft
from scipy.ndimage import median_filter
from scipy.ndimage import binary_dilation

def _parallel_execute(func, tasks, workers):
    """Execute tasks in a process pool; on spawn-related RuntimeError fall back to a thread pool.

    This mirrors XGA's pragmatic approach of using threads for some image-processing tasks to
    avoid spawn/pickling/import-time issues while keeping the process-based executor for
    best throughput when available.
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
    Unified worker for tile tasks.

    Expected task dicts:
      - ring-median: {'block','filled_block','slices','ring_footprint'}
      - dilation: {'block','slices','dilate_footprint'}
      - convolution: {'block','slices','kernel'}

    Returns (y0,y1,x0,x1, interior)
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

    This computes a near-square grid (n_tiles_y, n_tiles_x) and yields
    the tuples (y0,y1,x0,x1,e0,e1,f0,f1) for each tile including halo.
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
        signal.signal(signal.SIGTERM, self.cleanup_on_signal)
        signal.signal(signal.SIGINT, self.cleanup_on_signal)

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
                except Exception as e:
                    print(f'Warning: Failed to delete {file} ({e})')

    def cleanup_on_signal(self, signum=None, frame=None):
        """
        Remove files on signal.
        """        
        # Perform cleanup of temp files first.
        self.cleanup()

        # For SIGINT, re-raise the default KeyboardInterrupt so that
        # the interpreter stops as expected. For other signals (e.g.
        # SIGTERM) exit cleanly.
        try:
            if signum == signal.SIGINT:
                signal.default_int_handler(signum, frame)
            else:
                sys.exit(0)
        except KeyboardInterrupt:
            # Allow KeyboardInterrupt to propagate.
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
            except Exception as e:
                print(f'Warning: Failed to delete {path} ({e})')
        else:
            print(f'Warning: File {path} does not exist and cannot be deleted.')