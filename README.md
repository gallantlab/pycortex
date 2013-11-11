pycortex
========
![quickflat demo](https://raw.github.com/jamesgao/pycortex/master/wn_med.png?token=22802__eyJzY29wZSI6IlJhd0Jsb2I6amFtZXNnYW8vcHljb3J0ZXgvbWFzdGVyL3duX21lZC5wbmciLCJleHBpcmVzIjoxMzg0NzYxMzI2fQ%3D%3D--14b71ce5a0b1fc170039ffcdc2f56e5a1fc431a9)

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
$ sudo apt-get install python-dev python-numpy python-scipy python-matplotlib python-h5py python-nibabel python-lxml python-shapely
$ sudo pip install pycortex
```

Demo
----
Pycortex is best used with [IPython](http://www.ipython.org/). To run this demo, please download this [example dataset](http://gallantlab.org/pycortex/retinotopy_demo.hdf).

```
$ ipython
In [1]: import pycortex
In [2]: ds = cortex.openFile("retinotopy_demo.hdf")
In [3]: cortex.webshow(ds)
```

Documentation
-------------
Please find more complete documentation for pycortex at http://gallantlab.org/pycortex/docs/. The documentation for pycortex is currently incomplete, but will be improved in the coming days.