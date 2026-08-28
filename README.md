# CREST: Consistent Reduction, Extraction and SED fitting Tools

CREST is a Python library that collates the most popular tools from the literature to generate multi-band galaxy catalogues from calibrated imaging. It provides a consistent Python interface to these tools, facilitating flexible pipelines and making it easy to explore the associated systematics. CREST currently includes image processing and source extraction, but will soon be extended to SED fitting.

## Installation

Start by cloning the repository to a convinient location,

```
git clone https://github.com/jackcturner/crest.git
```

and navigating into the directory. You can then install the package, preferably in a virtual environment, with

```
pip install .
```

or

```
pip install -e .
```

for development purposes. This will install the Python dependencies, incluing SEP and Photutils which can be used out of the box. 

If you intend to apply extinction corrections, you will also need to install the Python interface to the [NED extinction calculator](https://github.com/mmechtley/ned_extinction_calc).

If you intend to use [Source Extractor](https://github.com/astromatic/sextractor), you will need to install it separately. You can then add the executable to PATH, or pass its location at runtime. This is currently also required for completeness estimation.

If you intend to use [ProFound](https://github.com/asgr/ProFound), you will first need an [R](https://www.r-project.org) installation and Rscript on your PATH. You can then install the [ProFound](https://github.com/asgr/ProFound) module and the other dependencies. CREST was developed using [R](https://www.r-project.org) v4.3.2 and package versions

- [ProFound](https://github.com/asgr/ProFound) v1.23.0
- [Rfits](https://github.com/asgr/Rfits) v1.10.9
- [rwcs](https://github.com/asgr/Rwcs) v1.8.4
- [EBImage](https://bioconductor.org/packages/release/bioc/html/EBImage.html) v4.44.0
- [hash](https://cran.r-project.org/web/packages/hash/index.html) v2.2.6.3
- [glue](https://cran.r-project.org/web/packages/glue/index.html) v1.8.0
- [stringr](https://cran.r-project.org/web/packages/stringr/index.html) v1.5.1
- [yaml](https://cran.r-project.org/web/packages/yaml/index.html) v2.3.10
- [rhdf5](https://www.bioconductor.org/packages/devel/bioc/html/rhdf5.html) v2.46.1

## Tools & Usage

As well as some unique functionality, CREST primarily relies on current tools. CREST encloses these within Python wrappers, which are controlled by YAML configuration files. This not only facilitates flexible pipelines, but makes it easy to keep track of the various parameters and settings used at each step.

A subset of the available functionality includes

### Imaging

`Background` - Tiered source masking and background subtraction based on [Bagley et al. (2023)](https://arxiv.org/abs/2211.02495). Now with multi-threading to speed up convolution steps and detection threshold scaling in low-weight regions.

`PSF` - Empirical PSF generation based primarily on [Weaver et al. (2023)](https://arxiv.org/abs/2301.02671). Now with more flexible star candidate selection, more robust stacking and multi-threading to speed up convolution with kernels.

`measure_completeness` - Completeness estimation through injection and recovery of synthetic sources with Source Extractor, based on [Stone et al. (2024)](https://arxiv.org/abs/2405.18470).

### Cataloguing

Source extraction wrappers are written to mimic the Source Extractor workflow and now include empirical depth and uncertainty estimation based on [Finkelstein et al. (2023)](https://arxiv.org/abs/2211.05792).

`SourceExtractor` - [Bertin & Arnouts (1996)](https://ui.adsabs.harvard.edu/abs/1996A%26AS..117..393B/abstract).

`SEP` - [Barbary (2016)](https://joss.theoj.org/papers/10.21105/joss.00058)

`Photutils` - [Bradley et al. (2024)](https://zenodo.org/records/13989456)

`ProFound` - [Robotham et al. (2018)](https://arxiv.org/abs/1802.00937).

### Fitting

Python wrapped SED fitting tools will be added soon.

## Citation & Acknowledgement

If you use CREST in your research, please cite the [FLAGS-I](https://ui.adsabs.harvard.edu/abs/2026arXiv260807668T/abstract) paper where it is introduced. Please also ensure that you cite any of the papers relevant to the tools you use.