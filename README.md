pycortex
========
![quickflat demo](https://raw.github.com/jamesgao/pycortex/master/docs/wn_med.png)

Pycortex is a software library that allows you to visualize fMRI or other volumetric neuroimaging data on cortical surfaces.

Quickstart
----------
IMPORTANT: The current pip version of pycortex is out of date and broken. Please do not use it, but instead directly use the latest version of pycortex from github:

```
$ git clone http://github.com/gallantlab/pycortex
$ cd pycortex
$ python setup.py develop
```

Also, a new version of pycortex is nearly ready for release. It is currently on the branch `glrework-merged`.

This last command installs pycortex into the site-packages folder in your local python installation. This means that you will not need to change your PYTHONPATH variable (don't worry if you have no idea what that means). If you are working in a python terminal, do not try to import pycortex from inside the source code directory, or the import will fail. 

Demo
----
Pycortex is best used with [IPython](http://www.ipython.org/). To run this demo, please download this [example dataset](http://gallantlab.org/pycortex/S1_retinotopy.hdf).

```
$ ipython
In [1]: import cortex
In [2]: ds = cortex.load("S1_retinotopy.hdf")
In [3]: cortex.webshow(ds)
```

Documentation
-------------
NEW: Massively updated documentation for pycortex is available at https://gallantlab.github.io/pycortex. You can find many examples of pycortex features in the [pycortex example gallery](https://gallantlab.github.io/pycortex/auto_examples/index.html).

Citation
--------
If you use pycortex in published work, please cite the [pycortex paper](http://dx.doi.org/10.3389/fninf.2015.00023):

_Gao JS, Huth AG, Lescroart MD and Gallant JL (2015) Pycortex: an interactive surface visualizer for fMRI. Front. Neuroinform. 9:23. doi: 10.3389/fninf.2015.00023_
