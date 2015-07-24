.. pycortex documentation master file, created by
   sphinx-quickstart on Thu Sep 13 19:15:55 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pycortex!
====================

.. image:: wn_med.png
    :width: 100%

pycortex is a package that allows interactive visualization of cortical fMRI data.

This documentation is still under development, and may contain inaccurate information. If you find anything that appears to be documented incorrectly or appears to be a bug, please let us know on github at https://github.com/jamesgao/pycortex/issues.

For example::
    
    import cortex
    im = cortex.quickshow((np.random.randn(31, 100, 100), "S1", "fullhead"))

Contents:

.. toctree::
   :maxdepth: 3
   
   install
   segmentation
   database
   align
   dataset
   rois
   transforms

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

