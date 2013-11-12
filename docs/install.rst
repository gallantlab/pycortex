Installation
============

Pycortex is available on the Python Packaging Index. The easiest way to install it is with the Anaconda_. To use pycortex with Anaconda_, first install anaconda as instructed, then type the following commands into a terminal::

    sudo pip install nibabel
    sudo pip install pycortex

If you are running Ubuntu, the built-in python packages should be sufficient. Use the following commands::

    sudo apt-get install python-dev python-numpy python-scipy python-matplotlib python-h5py python-nibabel python-lxml python-shapely python-html5lib
    sudo pip install pycortex

.. _Anaconda: https://store.continuum.io/cshop/anaconda/



Configuration
-------------
The pycortex configuration file is located in :file:`~/.config/pycortex/options.cfg`. The location of the *filestore* (i.e. database) needs to be written into the file under the ``[basic]`` header::

   [basic]
   filestore=/abs/path/to/filestore

By default, the filestore is automatically installed in 