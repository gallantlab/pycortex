pycortex
========
![quickflat demo](https://raw.github.com/jamesgao/pycortex/master/docs/wn_med.png)

Pycortex is a software that allows you to visualize fMRI or other volumetric mapping data on cortical surfaces.

Quickstart
----------
UPDATE 2015.11.24: Unfortunately, the pip install of pycortex is out of date and broken. The pip installation will thus not work. We apologize for any frustration or inconvenience this may have caused. We will eventually update the version stored on pip, but for now please see below for instructions on how to install pycortex directly from the git repository. 

The easiest way to configure your local python environment to suport pycortex is to use the [Anaconda python distribution](https://store.continuum.io/cshop/anaconda/). Download and install anaconda, then run the following command to install one non-standard library (nibabel) for reading and writing fMRI data:

```
$ sudo pip install nibabel 
```

This should work on Mac or Linux PCs. If you using Ubuntu, you can skip Anaconda and use the following command instead, which will install all python prerequisites for pycortex.

```
$ sudo apt-get install python-pip python-dev python-numpy python-scipy python-matplotlib python-h5py python-nibabel python-lxml python-shapely python-html5lib mayavi2 inkscape blender
```

To install from the github repository, call the following commands. For both commands, replace `<your_directory>` with the folder where you would like to store the pycortex source code.

```
$ git clone http://github.com/gallantlab/pycortex <your_directory> 
$ cd <your_directory>
$ sudo python setup.py install
```

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
Please find more complete documentation for pycortex at http://gallantlab.org/pycortex/docs/. The documentation for pycortex is currently incomplete, but will be improved in the coming days, weeks, or months.

Citation
--------
If you use pycortex in published work, please cite the [pycortex paper](http://dx.doi.org/10.3389/fninf.2015.00023):

_Gao JS, Huth AG, Lescroart MD and Gallant JL (2015) Pycortex: an interactive surface visualizer for fMRI. Front. Neuroinform. 9:23. doi: 10.3389/fninf.2015.00023_
