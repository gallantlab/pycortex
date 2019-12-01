Installation
============
The best way to install pycortex currently is by getting the latest source code from github::

    git clone https://github.com/gallantlab/pycortex.git
    cd pycortex

    python setup.py develop


Dependencies
------------
The easiest way to get most of the dependencies is using Anaconda_. To use pycortex with Anaconda_, first install anaconda as instructed, then::

    pip install numpy Cython scipy h5py nibabel matplotlib Pillow numexpr tornado lxml networkx

You will also need to install Inkscape_ using whatever method is appropriate for your system. On Mac OS X you will also need to enable access to Inkscape on the command line, see these instructions_.

If you are running Ubuntu without Anaconda, use the following commands::

    sudo apt-get install python-dev python-numpy python-scipy python-matplotlib python-h5py python-nibabel python-lxml python-shapely python-html5lib inkscape

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _Inkscape: https://inkscape.org/en/
.. _instructions: http://wiki.inkscape.org/wiki/index.php/Mac_OS_X#Inkscape_command_line

Demo
----
To test if your install went well, first download the `example dataset <http://gallantlab.org/pycortex/S1_retinotopy.hdf>`_. Then run the following commands at a terminal::
    
    $ ipython
    In [1]: import cortex
    In [2]: ds = cortex.load("S1_retinotopy.hdf")
    In [3]: cortex.webshow(ds)

If everything went well, this should pop up a web browser window with the same view as http://gallantlab.org/pycortex/retinotopy_demo/.

Basic Configuration
-------------------
Pycortex will automatically create a database filestore when it is first installed. In Linux, this filestore is located at :file:`/usr/local/share/pycortex/`. On first import, it will also create a configuration file in your user directory which allows you to specify additional options, including alternate filestore locations. In Linux, this user configuration file is located in :file:`~/.config/pycortex/options.cfg`.

You can check the location of the filestore after installing by running::

    import cortex
    cortex.database.default_filestore

And you can check the location of the config file by running::

    import cortex
    cortex.options.usercfg

If you want to move the filestore, you need to update the config file::

   [basic]
   filestore=/abs/path/to/filestore

