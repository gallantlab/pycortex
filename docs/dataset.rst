Pycortex Data Structures
========================

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

    dv = cortex.Volume(np.random.randn(31, 100, 100), "S1", "fullhead", cmap="RdBu_r", vmin=-3, vmax=3)
    ds = cortex.Dataset(name=dv)
    ds.save("test_data.hdf")

Loading data::

    ds = cortex.load("test_data.hdf")

If you already have a Nifti file aligned to a transform in the database::

    dv = cortex.Volume("path/to/nifti_file.nii.gz", "S1", "fullhead")
    cortex.quickshow(dv)

Overview
--------
Pycortex's main data structures consists of the :class:`Dataset`, and the Dataview classes :class:`Volume`, :class:`Vertex`, :class:`VolumeRGB`, :class:`VertexRGB`, :class:`Volume2d`, :class:`Vertex2D`.

    * :class:`Dataset` objects store a collection of :class:`Dataview` objects with associated names. It provides additional functionality to store and load the DataViews as HDF files.

    * :class:`Volume` is a :class:`Dataview` object that holds either a single or a time series of volumetric data (IE the data is in the original volume space).
    * :class:`Vertex` is a :class:`Dataview` object that holds either a single or a time series of vertex data (IE the data has been projected onto the surface vertices).
    * :class:`VolumeRGB` is a :class:`Dataview` object that contains 3 or 4 :class:`Volume` objects corresponding to the red, green, blue, and optionally alpha channels of a raw dataview.
    * :class:`Volume2D` is a :class:`Dataview` object that holds a pair of volumetric data, to be displayed using a 2D colormap.

Dataviews
---------
Dataview is a parent class that is instantiated in the form of one of the following subclasses: :class:`Volume`, :class:`Vertex`, :class:`VolumeRGB`, :class:`VertexRGB`, :class:`Volume2D`, :class:`Vertex2D`. All dataviews have the following additional kwargs:

    * **cmap**: colormap name. This should be a colormap name in matplotlib. See `matplotlib colormaps <http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps>`_  for acceptable names.
    * **vmin** / **vmax**: colormap minimum and maximums, same as in imshow.
    * **description**: A text description of the data view. This is shown as a subheading in the webgl view.
    * **priority**: Priority of this data view when included with others in a :class:`Dataset`.
    * Other kwargs: Additional string-based attributes are saved here. For example, additional quickflat options can be specified here, to be automatically applied during the :method:`cortex.quickflat.make_figure` command. webgl views currently do not support any other view options.

In all following examples, **kwargs** contains these options.

Volume and Vertex
~~~~~~~~~~~~~~~~~
These are your basic data classes. They have the following call signatures::

    vol = cortex.Volume(nparray, subject, xfmname, mask=None, **kwargs)
    vert = cortex.Vertex(nparray, subject, **kwargs)

In addition, there are two classmethods that allow you to define standard empty or random data views::

    empty = cortex.Volume.empty(subject, xfmname, **kwargs)
    rand = cortex.Volume.random(subject, xfmname, **kwargs)
    verts = cortex.Vertex.random(subject, **kwargs)

The :class:`Volume` and :class:`Vertex` objects support standard numpy operations. For example::

    vol = cortex.Volume.empty(subject, xfmname) + 1

:class:`Volume` also supports masked data. Masked data is data in which the following operation has occurred::

    data = np.random.randn(32, 100, 100)
    mask = np.random.rand(32, 100, 100) > .5

This is good for limiting the number of voxels required to model. Masks are delicate; you have to have the exact mask, otherwise the linear ordering of the voxels required to reconstruct the 3D volume is lost. It is hightly recommended that you store masks in the pycortex database for this reason. See :ref:`database-masks` for more information about storing masks.

Generally, the :class:`Volume` class will automatically understand masked data, assuming the mask exists in the database. If you have masked data where the mask does not exist in the database, you must provide it in the mask keyword argument.

Volume and Vertex objects can also be implicitly defined using a tuple syntax, as highlighted in the quickstart above.

RGB data
~~~~~~~~
If you have data which does not require colormapping, it is possible to directly plot them using an RGB dataview. The calling signature looks like this::

    rgb = cortex.VolumeRGB(red, green, blue, subject=None, xfmname=None, alpha=None, **kwargs)
    verts = cortex.VertexRGB(red, green, blue, subject=None, alpha=None, **kwargs)

You can provide either numpy arrays or :class:`Volume` objects for the red, green, blue, and alpha channels. If you provide :class:`Volume` objects, the subject and xfmname keywords must remain None. If you provide numpy arrays for red/green/blue, subject must not be None.

If you provided either numpy arrays or :class:`Volume` objects without vmin/vmax, the data will be normalized and cast to uint8 in order to display them. This means that the data will automatically be scaled between 0-255 and quantized. If your data is already normalized between 0 and 1, this will not occur.

2D dataviews
~~~~~~~~~~~~
In order to specify 2D data views in webgl, this helper class lets you specify a pair of :class:`Volume` objects to be plotted using a 2D colormap. Currently, quickflat does not yet support 2D colormaps. To declare a 2D dataview::

    dim1 = cortex.Volume.random(subject, xfmname)
    dim2 = cortex.Volume.random(subject, xfmname)
    twod = cortex.Volume2D(dim1, dim2, subject=None, xfmname=None, vmin2=None, vmax2=None, **kwargs)

Dataset
-------
Dataset objects hold a dictionary of DataView objects. Its main function is saving out data into a standardized HDF format. Datasets also allow subject data packing, if you desire to send data to someone without the subject database.

In addition, Datasets provide data deduplication. This is especially important for the webgl views, in order to reduce the amount of data transfer between server and client. Deduplication is accomplished using a SHA1 hash of the raw numpy data. This is not a foolproof method, but given the constraints of visualizing volumetric data, additional Numpy attributes do not need to be included in the hash.

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
            -> maskname
    /views
        name1[dataref, desc, cmap, vmin, vmax, state]
        name2[dataref, desc, cmap, vmin, vmax, state]

