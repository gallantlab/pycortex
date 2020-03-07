pycortex
========
[![Build Status](https://travis-ci.org/gallantlab/pycortex.svg?branch=master)](https://travis-ci.org/gallantlab/pycortex)

[![quickflat demo](https://raw.github.com/jamesgao/pycortex/master/docs/wn_med.png)](https://gallantlab.github.io/pycortex)

Pycortex is a software library that allows you to visualize fMRI or other volumetric neuroimaging data on cortical surfaces.

Installation
------------
To install the stable release version of pycortex, do the following:

```bash
# First, install some required dependencies (if not already installed)
pip install -U setuptools wheel numpy cython
# Install the latest release of pycortex from pip
pip install -U pycortex
```

If you wish to install the development version of pycortex, you can install it directly from Github.

To do so, replace the second install line above with the following:

```bash
# Install development version of pycortex from github
pip install -U git+git://github.com/gallantlab/pycortex.git
```

Documentation
-------------
Pycortex documentation is available at [https://gallantlab.github.io/pycortex](https://gallantlab.github.io/pycortex).

You can find many examples of pycortex features in the [pycortex example gallery](https://gallantlab.github.io/pycortex/auto_examples/index.html).

To build the documentation locally:
```bash
# Install required dependencies for the documentation
pip install sphinx_gallery numpydoc
# Move into the docs folder (assuming already in pycortex directory)
cd docs
# Build a local version of the documentation site
make html
```

After you run the above, you can open `docs/_build/html/index.html` in a web browser to view the locally built documentation.

Demo
----
Pycortex is best used with [IPython]().

If you do not already have IPython, you can install it by running:
```bash
pip install ipython
```

To run the pycortex demo, using IPython, run:
```ipython
$ ipython
In [1]: import cortex
In [2]: cortex.webshow(cortex.Volume.random("S1", "fullhead"))
```

Citation
--------
If you use pycortex in published work, please cite the [pycortex paper](http://dx.doi.org/10.3389/fninf.2015.00023):

_Gao JS, Huth AG, Lescroart MD and Gallant JL (2015) Pycortex: an interactive surface visualizer for fMRI. Front. Neuroinform. 9:23. doi: 10.3389/fninf.2015.00023_

Getting help
------------
Please post on [NeuroStars](https://neurostars.org/) with the tag `pycortex` to
ask questions about how to use Pycortex.
