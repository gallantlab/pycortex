Installation
============

pycortex relies on a large number of open source projects:

    numpy
    scipy
    matplotlib
    shapely
    traits
    mayavi
    lxml
    html5lib
    tornado

    jquery
    jquery-ui
    Three.js
    jquery.minicolors
    ddSlick

The pycortex configuration file is located in :file:`~/.config/pycortex/options.cfg`. The location of the *filestore* (i.e. database) needs to be written into the file under the ``[basic]`` header::

   [basic]
   filestore=/abs/path/to/filestore

