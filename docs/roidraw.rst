In-browser ROI drawing
======================

`pycortex-roidraw <https://github.com/gallantlab/pycortex-roidraw>`_ is a drop-in add-on that
lets you draw, edit, and export ROIs directly in a pycortex WebGL viewer — no Inkscape and no
re-generation of the viewer required. You lasso a region on the flattened cortical surface; the
stroke is fitted to a smooth, re-editable bezier curve, drawn as a white outline and label baked
into the surface, and exported to a portable JSON file.

It ships as a single self-contained bundle (``roidraw.bundle.js``, CSS included) that can be added
to **any** pycortex viewer — a static ``make_static`` export or a freshly generated dynamic viewer.

.. note::

   pycortex-roidraw lives in its own repository and is distributed separately from pycortex. The
   ROIs it exports are a portable JSON **vertex set** (with the editable bezier), independent of
   the Inkscape-based :doc:`surface-defined ROI system <rois>` — they are not read by
   ``get_roi_masks`` and friends. Use it for quick interactive annotation in the viewer; use the
   Inkscape workflow when you need ROI masks in the pycortex Python API.

Adding it to a viewer
---------------------

1. Download ``roidraw.bundle.js`` from the
   `latest release <https://github.com/gallantlab/pycortex-roidraw/releases/latest>`_
   (or build it from source — see the project README).
2. Copy it next to the viewer's HTML.
3. Add two tags before the closing ``</body>`` (pycortex ``make_static`` fragments have no
   ``</body>`` — append at the end instead)::

       <script src="roidraw.bundle.js"></script>
       <script>window.ROIDraw.autoAttach();</script>

``autoAttach()`` waits for the viewer to finish loading, then attaches. That is the entire
integration. The project also provides a ``bake.py`` helper that injects the bundle and the two
script tags into an existing static viewer non-destructively.

Drawing and editing ROIs
------------------------

A **Display / Draw** toggle is added at the top of the viewer. Switch to **Draw** and the brain
flattens and an ROI panel appears.

============== ==========================================================================
Gesture        Action
============== ==========================================================================
drag           Lasso a region, name it, and fit it to a smooth bezier drawn on the surface
scroll wheel   Zoom (to draw fine detail)
shift + drag   Pan the surface
shift + click  Inspect the voxel under the cursor
esc            Cancel the current lasso (or finish editing)
============== ==========================================================================

Click **✎ edit** next to an ROI in the panel to reveal its bezier anchors and tangent handles:
drag anchors and handles to reshape the curve, double-click the curve to insert an anchor,
double-click an anchor to toggle smooth/corner, and press ``Delete`` to remove one. Vertex
membership is re-derived from the bezier on every change, so the exported vertex set always
matches the curve you see.

Export format
-------------

The panel's **Export JSON** button writes a ``rois.json`` file holding, per ROI, the
per-hemisphere subject vertex indices, an ordered boundary ring, a label vertex, and the editable
bezier (control points in view-independent flat-UV coordinates). It re-imports — here or in any
viewer on the same surface — to the exact same outline, ready to re-edit.

Full documentation
------------------

See the `pycortex-roidraw README <https://github.com/gallantlab/pycortex-roidraw>`_ for the
complete gesture reference, the JSON schema, the helper scripts, and notes on porting the tool to
other viewer engines.
