Installation
============

To install the stable release version of pycortex, do the following::

    # First, install some required dependencies
    pip install -U setuptools wheel numpy cython
    # Install the latest release of pycortex from pip
    pip install -U pycortex


If you wish to install the development version of pycortex, you can install it directly from Github.

To do so, replace the second install line above with the following::

    # Install development version of pycortex from github
    pip install -U git+https://github.com/gallantlab/pycortex.git

Optional Dependencies
---------------------
For some functionality, you will also need to install Inkscape_, using whatever method is appropriate for your system.

On Mac OS X you will also need to enable access to Inkscape on the command line, see these instructions_.

.. _Inkscape: https://inkscape.org/en/
.. _instructions: http://wiki.inkscape.org/wiki/index.php/Mac_OS_X#Inkscape_command_line

Demo
----
To test if your install went well, you can run the pycortex demo.

Pycortex is best used with IPython_

If you do not already have IPython, you can install it by running::

    pip install ipython

To run the pycortex demo, using IPython, run::

    $ ipython
    In [1]: import cortex
    In [2]: cortex.webshow(cortex.Volume.random("S1", "fullhead"))

If everything went well, this should pop up a web browser window with a demo subject.

.. _IPython: http://www.ipython.org/

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
