Data input format
=================
Pycortex implements a set of helper functions that help normalize and standard data formats. It also wraps pytables to provide a standard way to store your data in HDF format.

Quickstart
----------
Datasets can be defined implicitly using a tuple syntax::

    import cortex
    cortex.webshow((np.random.randn(32, 100, 100), "AH", "AH_huth"))

To create a dataset manually::

    ds = cortex.Dataset(name=(np.random.randn(32, 100, 100), "AH", "AH_huth"))

This implicitly creates a VolumeData or VertexData object, depending on the shape of the data. For the full syntax::

    volume = cortex.VolumeData(np.random.randn(32, 100, 100), "AH", "AH_huth", cmap="RdBu")
    ds = cortex.Dataset(rand_data=volume)
    ds.save("test_data.hdf", pack=False)

Loading data::

    ds = cortex.openFile("test_data.hdf")


VolumeData
----------
Volumetric data is encapsulated by the VolumeData class. VolumeData expects at least three arguments to initialize properly:

    * data : Data to be made into a VolumeData object. Can be one of the following:
    * subject : string name of the subject
    * xfmname : string name of the transform

Additional optional arguments are mostly for display properties. Some useful ones:
    
    * mask : an important keyword argument, string name of the mask. If not provided, VolumeData will guess.
    * cmap : string name of the colormap
    * vmin / vmax : colormap limits
