Installation
============
Pycortex is available on the Python Packaging Index. The easiest way to install it is with Anaconda_. To use pycortex with Anaconda_, first install anaconda as instructed, then type the following commands into a terminal::

    sudo pip install nibabel
    sudo pip install pycortex

If you are running Ubuntu, the built-in python packages should be sufficient. Use the following commands::

    sudo apt-get install python-dev python-numpy python-scipy python-matplotlib python-h5py python-nibabel python-lxml python-shapely python-html5lib
    sudo pip install pycortex

If you wish to run the latest bleeding-edge version of pycortex, use the following instructions (this is also a good thing to try if for whatever reason the pip install does not work)::

    # This will create a source code directory for pycortex. 
    git clone https://github.com/gallantlab/pycortex
    cd pycortex
    # This will install the pycortex code into your local python installation
    sudo python setup.py install

.. _Anaconda: https://store.continuum.io/cshop/anaconda/

Demo
----
To test if your install went well, first download the `example dataset <http://gallantlab.org/pycortex/S1_retinotopy.hdf>`_. Then run the following commands at a terminal::
    
    $ ipython
    In [1]: import cortex
    In [2]: ds = cortex.load("S1_retinotopy.hdf")
    In [3]: cortex.webshow(ds)

If everything went well, this should pop up a web browser window with the same view as http://gallantlab.org/pycortex/retinotopy_demo/.

Configuration
-------------
Pycortex will automatically create a database filestore when it is first installed. In Linux, this filestore is located at :file:`/usr/local/share/pycortex/`. On first import, it will also create a configuration file in your user directory which allows you to specify additional options, including alternate filestore locations.

In Linux, this user configuration file is located in :file:`~/.config/pycortex/options.cfg`. The location of the *filestore* (i.e. database) needs to be written into the file under the ``[basic]`` header::

   [basic]
   filestore=/abs/path/to/filestore

.. todo:: Additional option documentation
