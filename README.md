pycortex
========
![quickflat demo](https://raw.github.com/jamesgao/pycortex/master/docs/wn_med.png)

Pycortex is a software that allows you to visualize fMRI or other volumetric mapping data on cortical surfaces.

Quickstart
----------
The easiest way to get pycortex is to use the [Anaconda python distribution](https://store.continuum.io/cshop/anaconda/). Download and install anaconda, then run the following commands:

```
$ sudo pip install nibabel
$ sudo pip install pycortex
```

If you are using [Ubuntu](http://ubuntu.com), using [Neurodebian](http://neuro.debian.net/) is highly recommended. The following command will install all python prerequisites:

```
$ sudo apt-get install python-pip python-dev python-numpy python-scipy python-matplotlib python-h5py python-nibabel python-lxml python-shapely python-html5lib mayavi2 inkscape blender
$ sudo pip install pycortex
```

Demo
----
Pycortex is best used with [IPython](http://www.ipython.org/). To run this demo, please download this [example dataset](http://gallantlab.org/pycortex/S1_retinotopy.hdf).

```
$ ipython
In [1]: import cortex
In [2]: ds = cortex.openFile("S1_retinotopy.hdf")
In [3]: cortex.webshow(ds)
```

Documentation
-------------
Please find more complete documentation for pycortex at http://gallantlab.org/pycortex/docs/. The documentation for pycortex is currently incomplete, but will be improved in the coming days.
