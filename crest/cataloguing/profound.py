import os
import subprocess
from collections import deque
from pathlib import Path
import yaml

class ProFound():
    """
    Wrapper around the ProFound.R ProFound class (via run_ProFound.R) 
    to allow running through Python.
    """

    def __init__(self, config_file, verbose=True):
        """
        __init__ method for ProFound.

        Arguments
        ---------
        config_file (str)
            Path to YAML configuration file.
        verbose (bool)
            If True, print progress messages.
        """

        # Resolve path to R wrappers.
        self.crest_path = str(Path(__file__).resolve().parent)

        script_path = Path(self.crest_path) / 'run_profound.R'
        if not script_path.exists():
            raise FileNotFoundError(f'Cannot find run_profound.R at {script_path}')

        # Store the configuration file path
        self.configfile = config_file

        # and the content.
        with open(self.configfile, 'r') as file:
            self.config = next(yaml.safe_load_all(file))

        self.verbose = verbose

    def _vprint(self, *args, **kwargs):
        """
        Print only when verbose output is enabled.
        """

        if self.verbose:
            print(*args, **kwargs)

    def _run_command(self, basecmd, keep_lines=2000):
        if self.verbose:
            # Stream combined output to avoid deadlocks and keep a full log.
            self._vprint("Running:", " ".join(basecmd))
            p = subprocess.Popen(
                basecmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            output_lines = deque(maxlen=keep_lines)
            for line in p.stdout:
                output_lines.append(line)
                print(line, end="")

            p.wait()
            out = "".join(output_lines)
            err = ""
        else:
            p = subprocess.Popen(basecmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = p.communicate()

        return p.returncode, out, err

    def measure_depth(self, science_path, psf_path, mask_path=None, error_path=None, 
                      parameters=None, radius=3.33, max_apers=50, max_iters=50000):
        """
        Measure the 5-sigma point source depth of an image using 
        ProFound. This method simply passes the parameters to 
        run_profound.R.
        
        Arguments
        ---------
        science_path (str)
            Filename of science fits image.
        psf_path (str)
            Filename of the PSF fits image used to scale the aperture 
            depths to total.
        mask_path (None, str)
            Path to the fits image mask. If None, generate and use
            a ProFound segmentation map.
        error_path (None, str)
            Path to the fits RMS map. 
            If None, only infinite non-source pixels will be masked.
        parameters (None, dict)
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

        if parameters is None:
            parameters = {}

        # Contruct the base command for running ProFound in depth mode.
        script_path = os.path.join(self.crest_path, 'run_profound.R')
        basecmd = [f'Rscript', script_path, 'type=depth', f'config_path={self.configfile}',
                   f'img1={science_path}', f'psf={psf_path}', f'radius={radius}',
                   f'max_apers={max_apers}', f'max_iters={max_iters}']
        
        # Add source mask.
        if isinstance(mask_path, type(None)):
            basecmd.append('mask=None')
        else:
            basecmd.append(f'mask={mask_path}')
        
        # Add error map.
        if isinstance(error_path, type(None)):
            basecmd.append('error=None')
        else:
            basecmd.append(f'error={error_path}')
        
        # Add the overwritten parameters.
        for key, value in parameters.items():
            basecmd.append(f'{key}={value}')

        # Now run on the command line.
        returncode, out, err = self._run_command(basecmd)

        if err:
            for line in err.splitlines():
                print(line)

        if returncode != 0:
            raise RuntimeError(
                'ProFound encountered an error.\n'
                f'Command: {" ".join(basecmd)}\n'
                f'Stdout:\n{out}\n'
                f'Stderr:\n{err}'
            )

        if 'Depth:' not in out:
            raise RuntimeError(
                'ProFound completed without returning a depth value.\n'
                'Expected token "Depth:" was not found in stdout.\n'
                f'Stdout:\n{out}\n'
                f'Stderr:\n{err}'
            )

        depth = out.split('Depth:')[1].strip()
        try:
            return float(depth)
        except ValueError as exc:
            raise RuntimeError(
                f'ProFound returned an unparsable depth value: {depth}'
            ) from exc

    def extract(self, science_path, parameters=None, outputs=None, cat_name=None, outdir='./'):
        """
        Perform source extraction and photometry using Profound. 
        This method simply passes the parameters to wrap_profound.R.

        Arguments
        ---------
        science_path (str, List[str])
            If str, the filename of the image to extract.
            If a List[str] filename of detection and measurement images.
        parameters (dict)
            Key-value pairs overwritting parameters in the config file 
            just for this run.
        outputs (list)
            List of output parameters to save.
        cat_name (None, str)
            The base name for the photometry catalogue. If None, use the 
            base name of the measurement file.
        outdir (str)
            Directory in which to store outputs. 

        Returns
        -------
        out_name (str)
            Path to the generated catalogue.
        """

        if parameters is None:
            parameters = {}

        # Construct the base command for running ProFound extraction.
        script_path = os.path.join(self.crest_path, 'run_profound.R')
        basecmd = [f'Rscript', script_path, 'type=extract', f'config_path={self.configfile}']

        # Add the science images.
        if type(science_path) == list:
            if len(science_path) == 2:
                basecmd += [f'img1={science_path[0]}', f'img2={science_path[1]}']
                name = os.path.basename(science_path[1]).replace(".fits", "_profound")
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                    'form [detection, measurement].') 
            
        elif isinstance(science_path, str):
            basecmd.append(f'img1={science_path}')
            name = os.path.basename(science_path).replace(".fits", "_profound")

        else:
            raise ValueError('Image inputs are not the correct format. Use strings for single image'
                           ' mode and lists of the form [detection, measurement] for double. '
                           'Use None for no weighting.')

        if isinstance(cat_name, type(None)):
            cat_name = name
        out_name = f'{outdir}/{cat_name}'

        # Get a comma seperated list of outputs.
        if outputs == None:
            basecmd.append('outputs=None')
        else:
            out_str = ''
            for output in outputs:
                out_str += f'{output},'
            basecmd.append(f'outputs={out_str[:-1]}')
        
        # Add the overwritten parameters.
        for key, value in parameters.items():
            basecmd.append(f'{key}={value}')
        
        # Add the catalogue name and output directory.
        basecmd.append(f'cat_name={cat_name}')
        basecmd.append(f'outdir={outdir}')

        # Now run on the command line.
        returncode, out, err = self._run_command(basecmd)

        if err:
            for line in err.splitlines():
                print(line)

        if returncode != 0:
            raise RuntimeError(
                'ProFound encountered an error.\n'
                f'Command: {" ".join(basecmd)}\n'
                f'Stdout:\n{out}\n'
                f'Stderr:\n{err}'
            )

        if 'out_name:' not in out:
            raise RuntimeError(
                'ProFound completed without returning an output catalogue name.\n'
                'Expected token "out_name:" was not found in stdout.\n'
                f'Stdout:\n{out}\n'
                f'Stderr:\n{err}'
            )

        out_name = out.split('out_name:')[1].strip()
        if out_name == '':
            raise RuntimeError(
                'ProFound returned an empty output catalogue name.\n'
                f'Stdout:\n{out}\n'
                f'Stderr:\n{err}'
            )

        return out_name