Data input format
=================

.. currentmodule:: cortex.dataset

Pycortex implements a set of helper functions that help normalize and standard data formats. It also wraps pytables to provide a standard way to store your data in HDF format.

Quickstart
----------
:class:`Dataset` objects can be defined implicitly using a tuple syntax::

    import cortex
    cortex.webshow((np.random.randn(32, 100, 100), "AH", "AH_huth"))

To create a dataset manually::

    ds = cortex.Dataset(name=(np.random.randn(32, 100, 100), "AH", "AH_huth"))

This implicitly creates a :class:`VolumeData` or :class:`VertexData` object, depending on the shape of the data. For the full syntax::

    volume = cortex.VolumeData(np.random.randn(32, 100, 100), "AH", "AH_huth", cmap="RdBu")
    ds = cortex.Dataset(rand_data=volume)
    ds.save("test_data.hdf", pack=False)

Loading data::

    ds = cortex.openFile("test_data.hdf")


VolumeData
----------
Volumetric data is encapsulated by the :class:`VolumeData` class. :class:`VolumeData` expects at least three arguments to initialize properly:

    * **data**: Data to be made into a VolumeData object. Can be one of the following:
        * Numpy array with shape
            * (t, z, y, x, c): Time series volume data in raw colors
            * (z, y, x, c): Single volume data in raw colors
            * (t, z, y, x): Time series volume data
            * (x, y, z): Single volume data
            * (t, m, c): Time series masked data in raw colors
            * (t, m): Time series masked data
            * (m, c): Single masked data in raw colors
            * (m,): Single masked data
        * Tuple with (filename, data) for matfiles or hdf files
        * Filename of a NifTi file
    * **subject**: string name of the subject
    * **xfmname**: string name of the transform

Additional optional arguments are mostly for display properties. Some useful ones:
    
    * **mask**: an important keyword argument, string name of the mask. If not provided, VolumeData will guess.
    * **cmap**: string name of the colormap
    * **vmin / vmax**: colormap limits

To create :class:`VolumeData` objects::

    import cortex
    ones = cortex.VolumeData(np.ones((32, 100, 100)), "AH", "AH_huth", cmap="RdBu_r")

:class:`VolumeData` objects also allow standard numpy operations to be performed::

    fives = ones + 4
    np.allclose(fives.volume, 5)

:class:`VolumeData` also include a few important properties and methods:

.. autosummary::
    :toctree:generated/

    VolumeData.map
    VolumeData.copy
    VolumeData.volume
    VolumeData.priority


Raw Data
~~~~~~~~
Any data that has uint8 type is automatically considered to be "raw" color data. This allows the support and plotting of RGB values as computed seperately, outside of the regular colormaps. If the data is input with type uint8, only a few data shapes are legal: (t, z, y, x, c), (z, y, x, c), (t, m, c), and (m, c). 

To input masked data, typically you can just 

Masked Data
~~~~~~~~~~~
If you are inputting masked data for a VolumeData entry, the mask MUST be in the pycortex database. This ensures that transforms don't get edited when you have masks associated. The database module checks for masks before it attempts to write to a transform to ensure nothing bad happens to the masks.

If a VolumeData object receives data with dimensionality 3, 

VertexData
----------
VertexData objects hold data that has been masked into vertex space. Typically, you will not directly create your own VertexData objects. 

Dataset
-------
Dataset objects hold a list of VolumeData and VertexData in a dictionary. Its main function is saving out data into a standardized HDF format. Datasets also allow subject data packing, if you desire to send data to someone without the subject database.

To save data::

    import cortex
    ds = cortex.Dataset(rand=(np.random.randn(32, 100, 100), "AH", "AH_huth"))
    ds.save("/tmp/dataset.hdf", pack=True)

Note that this example uses normalization to implicitly create a VolumeData object. 

HDF5 format::

    /subjects/
        s1/
            rois.svg
            transforms/
                xfm1/
                    xfm[4,4]
                    masks/
                        thin[z,y,x]
                        thick[z,y,x]
                xfm2/
                    xfm[4,4]
                    masks/
                        thin[z,y,x]
            surfaces
                fiducial
                    lh
                        pts[n,3]
                        polys[m,3]
                    rh
                        pts[n,3]
                        polys[m,3]
    /datasets/
        ds1
        ds2