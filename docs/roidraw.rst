In-browser ROI and sulcus drawing
=================================

`pycortex-roidraw <https://github.com/gallantlab/pycortex-roidraw>`_ is a drop-in add-on that
lets you draw, edit, and export ROIs and sulci directly in a pycortex WebGL viewer — no Inkscape
and no re-generation of the viewer required. You draw on the flattened cortical surface; the
stroke is fitted to a smooth, re-editable bezier curve, drawn as a colored outline and label baked
into the surface.

ROIs are **closed** curves that carry per-hemisphere vertex membership and export to a portable
JSON vertex set. Sulci are **open** curves that carry no vertex data and export as an
``overlays.svg`` fragment — the same representation pycortex already uses for sulci.

It ships as a single self-contained bundle (``roidraw.bundle.js``, CSS included) that can be added
to **any** pycortex viewer — a static ``make_static`` export or a freshly generated dynamic viewer.

.. note::

   pycortex-roidraw lives in its own repository and is distributed separately from pycortex. The
   ROIs it exports are a portable JSON **vertex set** (with the editable bezier), independent of
   the Inkscape-based :doc:`surface-defined ROI system <rois>` — they are not read by
   ``get_roi_masks`` and friends. Use it for quick interactive annotation in the viewer; use the
   Inkscape workflow when you need ROI masks in the pycortex Python API.

   Drawn **sulci** are different: they are exported as ``overlays.svg`` markup, so they *are* read
   by pycortex's own machinery (``quickflat``'s sulci overlay, the WebGL viewer, Inkscape) once the
   fragment is merged into a subject's overlay file.

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

Drawing and editing
-------------------

A **Display / Draw** toggle is added at the top of the viewer. Switch to **Draw**: the brain
flattens and a draw panel appears. A ``ROI | Sulcus`` selector at the top of the panel chooses what
a plain drag draws.

================= =========================================================================
Gesture           Action
================= =========================================================================
drag (ROI)        Lasso a region, name it, and fit it to a smooth closed bezier
drag (Sulcus)     Trace along the sulcus, name it, and fit it to a smooth open bezier
scroll wheel      Zoom (to draw fine detail)
shift + drag      Pan the surface
shift + click     Inspect the voxel under the cursor
esc               Cancel the current stroke (or finish editing)
================= =========================================================================

Click **✎ edit** next to any shape in the panel to reveal its bezier anchors and tangent handles:
drag anchors and handles to reshape the curve, double-click the curve to insert an anchor,
double-click an anchor to toggle smooth/corner, and press ``Delete`` to remove one. An open curve's
endpoints show a single handle each and are always corners. For an ROI, vertex membership is
re-derived from the bezier on every change, so the exported vertex set always matches the curve you
see; a sulcus has no membership, and its label instead follows the curve.

Export formats
--------------

**Export ROIs (JSON)** writes a ``rois.json`` file holding, per ROI, the per-hemisphere subject
vertex indices, an ordered boundary ring, a label vertex, and the editable bezier (control points
in view-independent flat-UV coordinates). It re-imports — here or in any viewer on the same
surface — to the exact same outline, ready to re-edit.

**Export sulci (SVG)** writes a ``sulci.svg`` fragment in pycortex's own overlay format: open,
unfilled ``<path>`` elements in a ``sulci`` layer, one inkscape-labeled group per named sulcus.
Trace a sulcus on each hemisphere and give both strokes the same name, and they merge into a single
group with one ``<path>`` per hemisphere — exactly how a hand-authored sulcus such as ``CaS`` is
stored. Merge the fragment into the subject's ``overlays.svg`` and ``quickflat``, the WebGL viewer,
and Inkscape all read it.

Sulci carry no vertex data, matching pycortex: there is no ``get_sulci_verts``, and sulci are
display geometry. Sulcus export is one-way — they are not re-imported from SVG.

Full documentation
------------------

See the `pycortex-roidraw README <https://github.com/gallantlab/pycortex-roidraw>`_ for the
complete gesture reference, the JSON schema, the helper scripts, and notes on porting the tool to
other viewer engines.
