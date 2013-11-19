Data input format
=================

.. currentmodule:: cortex.dataset

Pycortex implements a set of helper functions that help normalize and standard data formats. It also wraps h5py to provide a standard way to store your data in HDF format.

Quickstart
----------
:class:`Dataset` objects can be defined implicitly using a tuple syntax::

    import cortex
    cortex.webshow((np.random.randn(31, 100, 100), "S1", "fullhead"))

To create a dataset manually::

    ds = cortex.Dataset(name=(np.random.randn(31, 100, 100), "S1", "fullhead"))

To generate a data view and save it::

    dv = cortex.DataView((np.random.randn(31, 100, 100), "S1", "fullhead"), cmap="RdBu_r", vmin=-3, vmax=3)
    ds = cortex.Dataset(name=dv)
    ds.save("test_data.hdf")

Loading data::

    ds = cortex.openFile("test_data.hdf")

If you already have a Nifti file with the alignment written into the header::

    dv = cortex.DataView(("path/to/nifti_file.nii.gz", "S1", "identity"))
    cortex.quickshow(dv)

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

Generally, you will not need to instantiate BrainData objects yourself. They are automatically instantiated when Dataset objects or DataView objects are created. BrainData objects allow numpy operations::
    
    ones = cortex.DataView((np.ones(31, 100, 100), "S1", "fullhead"))
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

The **mask** parameter may be either an ndarray with the same shape as the transform reference, or the name of a mask in the database. If a "linear" shaped data is passed to VolumeData and a mask name is not specified, pycortex will attempt to locate the correct mask from the database. An exception will occur if a matching one is not found.

DataView
--------
DataView objects allow you to define a specific set of view attributes to attach to BrainData objects. This separation of data from its view allows for data deduplication if you want different views of the same data. For example::

    vol = cortex.VolumeData(np.random.randn(31, 100, 100), "S1", "fullhead")
    dv1 = cortex.DataView(vol, cmap="hot", vmin=0, vmax=1)
    dv2 = cortex.DataView(vol, cmap="RdBu_r", vmin=-1, vmax=1)

If you have not saved out a dataset file, then both ``dv1`` and ``dv2`` only contain references to the same ``vol`` VolumeData object. During the dataset saving process, hashes of the volumes are taken, allowing data deduplication. Datasets loaded from files reference the same h5py datasets to avoid memory duplication.

Dataview objects can be instantiated with many attributes. The default ones are as follows:

    * **data**: either a BrainData object or a multiview list
    * **description**: a text string describing the view on this data
    * **cmap**: a string that's passed to matplotlib that determines the colormap. This parameter is ignored if the BrainData object is raw.
    * **vmin**: the colormap minimum
    * **vmax**: the colormap maximum
    * **priority**: Priority of this data view when included with others in a :class:`Dataset`.

Additional attributes are saved in the **attrs** dictionary inside the DataView. These are automatically stored inside the Dataset HDF file.

Multiviews
~~~~~~~~~~
DataView objects allow the definition of something called a "multiview". The majority of the multiview functionality is not yet implemented yet. However, one important functionality is already supported -- two-dimensional colormaps.

Multiviews are intended to support the ability to view multiple brains at the same time. For example, the `movie demo <http://gallantlab.org/pycortex/movie_demo/>`_ is a rudimentary multiview, where two subjects are displayed at the same time. A multiview is defined by passing a list of BrainData objects as the first parameter.

Two-dimensional colormaps are supported using a multiview definition. If any multiview entry contains a 2-tuple instead of a BrainData object, the pair of BrainData objects are the two dimension for a 2D colormap. For example::
    
    dim1 = cortex.VolumeData(np.random.randn(31, 100, 100), "S1", "fullhead")
    dim2 = cortex.VolumeData(np.random.randn(31, 100, 100), "S1", "fullhead")
    dv = cortex.DataView([(dim1, dim2)], cmap="RdBu_covar", vmin=[(-1,-3)], vmax=[(1,3)])

Notice that the vmin/vmax values should match the structure of input.

Dataset
-------
Dataset objects hold a dictionary of DataView objects. Its main function is saving out data into a standardized HDF format. Datasets also allow subject data packing, if you desire to send data to someone without the subject database.

To save data::

    import cortex
    ds = cortex.Dataset(rand=(np.random.randn(31, 100, 100), "S1", "fullhead"))
    ds.save("/tmp/dataset.hdf", pack=True)

The saved out HDF5 format has the following structure::

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
