import os
import subprocess
from pathlib import Path
import yaml


class ProFound():
    """
    Wrapper around the ProFound.R ProFound class (via run_ProFound.R) 
    to allow running through Python.
    """

    def __init__(self, config_file):
        """
        __init__ method for ProFound.

        Arguments
        ---------
        config_file (str)
            Path to ".yml" configuration file.
        """

        # Store the configuration file path
        self.configfile = config_file

        # and the content.
        with open(self.configfile, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.config = content[0]

        # Resolve path to bundled CREST R wrappers from the installed package.
        self.crest_path = str(Path(__file__).resolve().parent)

        script_path = Path(self.crest_path) / 'run_profound.R'
        if not script_path.exists():
            raise FileNotFoundError(f'Cannot find run_profound.R at {script_path}')

    def measure_depth(self, science, psf, mask=None, error=None, parameters=None, radius=3.33,
                      max_apers=50, max_iters=50000):
        """
        Measure the 5-sigma point source depth of an image using 
        ProFound. This method simply passes the parameters to 
        wrap_profound.R.
        
        Arguments
        ---------
        science (str)
            Filename of science fits image.
        psf (str)
            Filename of the PSF fits image used to scale the aperture 
            depths to total.
        mask (None, str)
            Filename of the fits image mask. If None, generate and use
            a ProFound segmentation map.
        error (None, str)
            Filename of fits RMS map. If None, no weighting will be 
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

        if parameters is None:
            parameters = {}

        # Contruct the base command for running ProFound in depth mode.
        script_path = os.path.join(self.crest_path, 'run_profound.R')
        basecmd = [f'Rscript', script_path, 'type=depth', f'config_path={self.configfile}',
               f'img1={science}', f'psf={psf}',
                   f'radius={radius}', f'max_apers={max_apers}', f'max_iters={max_iters}']
        
        # Add source mask.
        if isinstance(mask, type(None)):
            basecmd.append('mask=None')
        else:
            basecmd.append(f'mask={mask}')
        
        # Add error map.
        if isinstance(error, type(None)):
            basecmd.append('error=None')
        else:
            basecmd.append(f'error={error}')
        
        # Add the overwritten parameters.
        for key, value in parameters.items():
            basecmd.append(f'{key}={value}')

        # Now run on the command line.     
        p = subprocess.Popen(basecmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     text=True, cwd=self.crest_path)
        for line in p.stderr:
            print(line)
        out, err = p.communicate()       

        # Get the depth and return it.
        if p.returncode == 0:
            if 'Depth:' in out:
                depth = out.split('Depth:')[1].strip()
                return float(depth)
        else:
            raise RuntimeError('ProFound encountered an error. Check the '
                               'output for further information.')

    def extract(self, science, parameters=None, outputs=None, cat_name=None, outdir='./'):
        """
        Perform source extraction and photometry using Profound. 
        This method simply passes the parameters to wrap_profound.R.

        Arguments
        ---------
        science (str, List[str])
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
        if type(science) == list:
            if len(science) == 2:
                basecmd += [f'img1={science[0]}', f'img2={science[1]}']
                name = os.path.basename(science[1]).replace(".fits","_profound")
            else:
                raise ValueError('Double image mode requires a list of weight paths of the '
                    'form [detection, measurement].') 
            
        elif isinstance(science, str):
            basecmd.append(f'img1={science}')
            name = os.path.basename(science).replace(".fits","_profound")

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
        
        # Add the catalogue name.
        basecmd.append(f'cat_name={cat_name}')

        # Finally the output directory.
        basecmd.append(f'outdir={outdir}')

        # Now run on the command line.     
        p = subprocess.Popen(basecmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     text=True, cwd=self.crest_path)
        for line in p.stderr:
            print(line)
        out, err = p.communicate()  

        # Get the catalogue name and return it.
        if p.returncode == 0:
            if 'out_name:' in out:
                out_name = out.split('out_name:')[1].strip()
                return out_name
        else:
            raise RuntimeError('ProFound encountered an error. Check the '
                    'output for further information.')  