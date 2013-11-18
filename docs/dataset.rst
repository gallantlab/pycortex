Data input format
=================

.. currentmodule:: cortex.dataset

Pycortex implements a set of helper functions that help normalize and standard data formats. It also wraps h5py to provide a standard way to store your data in HDF format.

Quickstart
----------
:class:`Dataset` objects can be defined implicitly using a tuple syntax::

    import cortex
    cortex.webshow((np.random.randn(32, 100, 100), "S1", "fullhead"))

To create a dataset manually::

    ds = cortex.Dataset(name=(np.random.randn(32, 100, 100), "S1", "fullhead"))

To generate a data view and save it::

    dv = cortex.DataView((np.random.randn(32, 100, 100), "S1", "fullhead"), cmap="RdBu_r", vmin=-3, vmax=3)
    ds = cortex.Dataset(name=dv)
    ds.save("test_data.hdf")

Loading data::

    ds = cortex.openFile("test_data.hdf")

If you already have a Nifti file with the alignment written into the 

Overview
--------
Pycortex's main data structures consists of the :class:`Dataset`, the :class:`DataView`, and the :class:`BrainData` objects. These three objects serve different roles:

    * :class:`BrainData` objects store brain data together with the subject information. Most pycortex data structures will automatically cast a tuple into the appropriate :class:`VolumeData` and :class:`VertexData` objects.
    * :class:`DataView` objects store :class:`BrainData` objects along with view attributes such as the colormap, minimum, and maximum. The stored :class:`BrainData` object is only a reference.
    * :class:`Dataset` objects store a collection of :class:`DataView` objects with associated names. It provides additional functionality to store and load the DataViews as HDF files.

These data structures can typically be cast from simpler structures such as tuples or dictionaries.

BrainData
---------
Brain data is encapsulated by the :class:`VolumeData` and :class:`VertexData` classes. :class:`VolumeData` expects three arguments to initialize properly:

    * **data**: Data to be interpreted as volumetric data. Can be one of the following:
        * Numpy array with shape
            * (t, z, y, x, c): Time series volume data in raw colors
            * (z, y, x, c): Single volume data in raw colors
            * (t, z, y, x): Time series volume data
            * (x, y, z): Single volume data
            * (t, m, c): Time series masked data in raw colors
            * (t, m): Time series masked data
            * (m, c): Single masked data in raw colors
            * (m,): Single masked data
        * Filename of a NifTi file
    * **subject**: string name of the subject
    * **xfmname**: string name of the transform

:class:`VertexData` objects require just two elements:

    * **data**: Data to be interpreted as vertex data. Can be one of the following:
        * Numpy array with shape
            * raw linear movie: (t, v, c)
            * reg linear movie: (t, v)
            * raw linear image: (v, c)
            * reg linear image: (v,)
    * **subject**: string name of the subject

Generally, you will not need to instantiate BrainData objects yourself. They are automatically instantiated when Dataset objects or DataView objects are created. VertexData objects allow numpy operations::
    
    ones = cortex.DataView((np.ones(32, 100, 100), "S1", "fullhead"))
    fives = cortex.DataView(ones.data + 4)
    np.allclose(fives.data.volume, 5)

:class:`VolumeData` also include a few important properties and methods:

.. autosummary::
    :toctree:generated/

    VolumeData.map
    VolumeData.copy
    VolumeData.volume

Raw Data
~~~~~~~~
Any data that has uint8 type is automatically considered to be "raw" color data. This allows the support and plotting of RGB values as computed seperately, outside of colormaps. If the data is input with type uint8, only a few data shapes are legal: (t, z, y, x, c), (z, y, x, c), (t, m, c), and (m, c). 

Masked Data
~~~~~~~~~~~
Masked data is data in which the following operation has occurred::
    
    data = np.random.randn(32, 100, 100)
    mask = np.random.rand(32, 100, 100) > .5
    masked = data[mask]

This is exceptionally useful for doing voxel selection. Masks should be stored in the pycortex database in order to avoid overwritten transforms. However, :class:`VolumeData` objects do allow you to store custom masks. The syntax for using custom masks is as follows::

    vol = cortex.VolumeData(data, subject, xfmname, mask=mask)

The **mask** parameter may be either an ndarray with the same shape as the transform reference or the name of a transform in the database. If 

DataView
--------


Dataset
-------
Dataset objects hold a dictionary of DataView objects. Its main function is saving out data into a standardized HDF format. Datasets also allow subject data packing, if you desire to send data to someone without the subject database.

To save data::

    import cortex
    ds = cortex.Dataset(rand=(np.random.randn(32, 100, 100), "AH", "AH_huth"))
    ds.save("/tmp/dataset.hdf", pack=True)

HDF5 format::

    /subjects/
        s1/
            rois/
                name[n]
                name[n]
                name[n]
            transforms/
                xfm1/
                    xfm[4,4]
                    masks/
                        thin[z,y,x]
                        thick[z,y,x]
                        __sha1hash[z,y,x]
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
    /data/
        hash
            -> subject
            -> xfmname
            -> maskname
    /views
        name1[dataref, desc, cmap, vmin, vmax, state]
        name2[dataref, desc, cmap, vmin, vmax, state]
