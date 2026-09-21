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

Before the browser opens, pycortex prints a security warning and asks you to
confirm. The viewer is served by a small web server that listens on *every*
network interface with no authentication, so while it is running anyone who can
reach your machine over the network can read the data you are plotting, read
files underneath your working directory, and supply the contents of the
screenshots and SVGs the viewer saves. Answer ``y`` to start the viewer once,
``n`` to abort, or ``i`` to start it and never be asked again — ``i`` writes
``skip_security_warning = true`` to the ``[webshow]`` section of your user
configuration file (see `Basic Configuration`_ below), and deleting that line
restores the prompt. Setting the ``PYCORTEX_SKIP_SECURITY_WARNING`` environment
variable suppresses the prompt for a single session, which is the right choice
for scripted or headless use.

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

Pycortex also caches some derived, subject-specific files (such as flatmap
and MNI transform caches) that don't need to live in the filestore itself.
By default, these caches are stored inside the filestore, at
:file:`{filestore}/{subject}/cache`. To store them somewhere else instead
(for example on faster local storage), set the ``cache`` option under
``[basic]`` in the config file::

   [basic]
   cache=/abs/path/to/cache

When ``cache`` is set, per-subject cache files are stored at
:file:`{cache}/{subject}/cache`.
