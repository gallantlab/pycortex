pycortex
========
[![Build Status](https://travis-ci.org/gallantlab/pycortex.svg?branch=master)](https://travis-ci.org/gallantlab/pycortex)

[![quickflat demo](https://raw.github.com/jamesgao/pycortex/master/docs/wn_med.png)](https://gallantlab.github.io/)

Pycortex is a software library that allows you to visualize fMRI or other volumetric neuroimaging data on cortical surfaces.

Quickstart
----------
```
python3 -m venv env  # use `virtualenv env` for python 2
source env/bin/activate
pip install -U setuptools wheel numpy cython
pip install -U git+git://github.com/gallantlab/pycortex.git
```
This command creates a new [virtualenv](https://docs.python.org/3/library/venv.html) for pycortex to resolve dependencies. Run `source env/bin/activate` whenever you need pycortex.

Documentation
-------------
NEW: Massively updated documentation for pycortex is available at https://gallantlab.github.io/. You can find many examples of pycortex features in the [pycortex example gallery](https://gallantlab.github.io/auto_examples/index.html).

To build the documentation locally:
```bash
pip install sphinx_gallery
pip install numpydoc
cd docs
make html
# open `docs/_build/html/index.html` in web browser
```

Demo
----
Pycortex is best used with [IPython](http://www.ipython.org/). Install it in your virtualenv using 
```
source env/bin/activate
pip install ipython
```
To run the pycortex demo,
```
$ ipython
In [1]: import cortex
In [2]: cortex.webshow(cortex.Volume.random("S1", "fullhead"))
```

Citation
--------
If you use pycortex in published work, please cite the [pycortex paper](http://dx.doi.org/10.3389/fninf.2015.00023):

_Gao JS, Huth AG, Lescroart MD and Gallant JL (2015) Pycortex: an interactive surface visualizer for fMRI. Front. Neuroinform. 9:23. doi: 10.3389/fninf.2015.00023_
