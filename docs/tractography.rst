Tractography
============

Pycortex can draw streamlines -- the output of diffusion tractography, for
instance pyAFQ or DIPY bundles -- inside the WebGL viewer, on top of the
cortical surface. Streamlines are held in a :class:`cortex.Tractogram`, a
:class:`Dataview` like :class:`Volume` or :class:`Vertex`, so they travel
through the same :class:`Dataset` and viewer machinery as everything else.

This is a viewer-only feature. Streamlines are 3-D curves through the white
matter and have no flatmap representation, so :mod:`cortex.quickflat` has no
code path for them at all: :func:`cortex.quickshow` takes a single dataview
rather than a :class:`Dataset`, and handing it a :class:`Tractogram` raises
rather than drawing a flatmap without the streamlines.

.. contents::
   :local:


Installing the reader
---------------------

Reading TRX files needs `trx-python <https://github.com/tee-ar-ex/trx-python>`_,
which pycortex does not require by default::

    pip install "pycortex[tractography]"

The dependency is imported lazily, so pycortex works normally without it; only
:meth:`Tractogram.from_trx` needs it. ``trx-python`` requires Python 3.11 or
newer. Building a tractogram from arrays needs nothing extra.


Loading streamlines
-------------------

From a TRX file::

    import cortex

    tract = cortex.Tractogram.from_trx("bundles.trx", "S1")

``from_trx`` also accepts an already-open ``TrxFile``, which is what you want
when you built the tractogram in memory or need to pass reader options. Since
``trx-python``'s loader also reads ``.trk`` and ``.tck``, those work through the
same call::

    from trx.io import load

    tract = cortex.Tractogram.from_trx(load("bundle.trk", reference="t1.nii.gz"), "S1")

From plain arrays -- a list of ``(L_i, 3)`` point arrays, one per streamline::

    tract = cortex.Tractogram.from_streamlines(list_of_arrays, "S1")

Or from the flat representation the class stores internally, an ``(N, 3)``
array of points plus an ``(M+1,)`` table of offsets delimiting the ``M``
streamlines (the TRX convention, with a trailing sentinel equal to ``N``)::

    tract = cortex.Tractogram(points, offsets, "S1")

Per-point and per-streamline scalars, and named subsets of streamlines, are
carried alongside::

    tract = cortex.Tractogram.from_streamlines(
        list_of_arrays, "S1",
        dpv={"fa": fa_per_point},          # (N,) or (N, k), one row per point
        dps={"length": length_per_line},   # (M,) or (M, k), one row per streamline
        groups={"CST_L": np.array([0, 1, 2]), "AF_L": np.array([3, 4])},
    )

``from_trx`` fills all three from the TRX file's ``dpv/``, ``dps/`` and
``groups/`` entries, so a pyAFQ TRX arrives with its bundles already named.


Alignment
---------

Streamline positions must be in the same millimeter space as the subject's
fiducial surfaces, which pycortex stores in FreeSurfer scanner RAS. When the
diffusion data was registered to the same T1 that produced the surfaces -- the
usual case for a single-subject pipeline -- tractography output is already in
that space and needs no transform.

Otherwise pass a 4x4 affine, applied to the points as homogeneous coordinates
right after loading::

    tract = cortex.Tractogram.from_trx("mni_bundles.trx", "fsaverage", xfm=mni_to_surface)

Nothing about the alignment is checked or inferred: streamlines in the wrong
space render in the wrong place rather than raising.


Coloring
--------

``color`` picks how streamlines are colored, and is resolved in Python (by
:meth:`Tractogram.vertex_colors`) into one RGB triple per point, so the viewer
receives finished colors:

``"orientation"`` (the default)
    Color by the local tangent direction: ``|dx|, |dy|, |dz|`` mapped to red,
    green and blue. This is the standard directionally-encoded color scheme, in
    which left-right tracts read red, anterior-posterior green, and
    inferior-superior blue.

an ``(r, g, b)`` tuple
    One constant color for every streamline. Useful when several tractograms
    are shown together and each should be identifiable.

``"dpv:<name>"`` or ``"dps:<name>"``
    Color by one of the named per-point or per-streamline scalars, mapped
    through ``cmap``, ``vmin`` and ``vmax`` exactly as for any other dataview.
    Per-streamline values are broadcast to every point of their streamline.
    This is how a tractometry result -- one value per node along a bundle --
    gets onto the streamlines::

        tract = cortex.Tractogram.from_trx(
            "bundles.trx", "S1", color="dpv:fa", cmap="viridis", vmin=0, vmax=1,
        )


Displaying
----------

A tractogram cannot be shown on its own; the viewer is built around a cortical
surface, so pass it in a :class:`Dataset` together with at least one
:class:`Volume` or :class:`Vertex`::

    ds = cortex.Dataset(overlay=cortex.Vertex(values, "S1"), bundles=tract)
    cortex.webgl.show(ds)

Static viewers work the same way; the streamline buffers are written next to
the other payloads, under ``tracts/``::

    cortex.webgl.make_static("/path/to/viewer", ds)

A running viewer can also be handed new tractograms, like any other data::

    handle = cortex.webgl.show(ds)
    handle.addData(more_bundles=other_tract)

.. note::
   Single-file exports produced by ``htmlembed`` do not yet inline the
   streamline buffers, so a static viewer with tractograms must be served as a
   directory.


In the viewer
-------------

Tractograms get their own panel under the dataset box, since what they show is
data rather than surface geometry. Each tractogram is one entry in that panel,
with a visibility checkbox, its name, and a toggle that expands the entry to
reveal an opacity slider with a value box beside it for typing an exact
number, and, when the tractogram has groups, a *bundles*
section: an ``all`` / ``none`` pair of links and one checkbox per group,
labeled with its streamline count. Bundles are listed alphabetically rather
than in the order the file gives them, with digit runs compared as numbers so
that ``CST_2`` precedes ``CST_10``. A streamline is drawn while it belongs to
at least one checked group; streamlines in no group at all are covered by a
trailing ``(ungrouped)`` checkbox, which always sorts last.

Streamlines are hidden as soon as the surface starts to inflate or flatten,
since their coordinates only mean anything against the folded surface.

The same controls are reachable from Python through the viewer handle::

    handle.tracts.bundles.setOpacity(0.5)
    handle.tracts.bundles.setGroupVisible("CST_L", False)
    handle.tracts.bundles.hideAllGroups()
    handle.tracts.bundles.showAllGroups()

Seeing the streamlines *inside* the brain needs a translucent cortex, which is
the ``surface_opacity`` slider in the surface controls (see
:doc:`userguide/webgl`). With an opaque surface only the parts of a streamline
that emerge from the cortex are visible.

Lowering a tractogram's own opacity fades it toward whatever is behind it, but
does not let you see one streamline through another: streamlines occlude each
other by depth at every opacity, so which bundle looks nearest never changes as
the slider moves. Showing a crossing bundle that is hidden behind another means
unchecking the one in front, not fading it. At an opacity of zero the
tractogram disappears completely, rather than leaving streamline-shaped
cut-outs in a translucent surface.


Large tractograms
-----------------

Every point is uploaded to the browser as three floats plus three color bytes,
and each streamline segment costs two indices, so a whole-brain tractogram of a
few million points is tens of megabytes and will make the viewer sluggish long
before it runs out of memory. :meth:`Tractogram.subsample` decimates one for
interactive use::

    tract.subsample(max_streamlines=5000)   # random subset, seeded and reproducible
    tract.subsample(step=10)                # every 10th streamline

:meth:`Tractogram.select` and :meth:`Tractogram.get_group` return new
tractograms restricted to a chosen set of streamlines or to a single named
group, with the group memberships remapped to the surviving streamlines.


Current limits
--------------

* Streamlines render one pixel wide. WebGL clamps ``gl_LineWidth`` to 1 on
  nearly every desktop driver, so ``linewidth`` has no effect there; drawing
  thicker tracts needs tubes or instanced quads, which the bundled Three.js
  (r69) makes expensive.
* A tractogram is limited to about 4.3 billion points by the 32-bit index used
  on the wire; the practical limit is much lower, see above.
* :class:`Tractogram` is not written to HDF5. Saving a :class:`Dataset` that
  contains one raises :exc:`NotImplementedError`; keep the TRX file as the
  source of truth and rebuild the tractogram when loading.
* No flatmap or quickflat support, and no projection of streamline endpoints
  onto the surface.
