In-browser ROI and sulcus drawing
=================================

`pycortex-roidraw <https://github.com/gallantlab/pycortex-roidraw>`_ is a drop-in add-on that
lets you draw, edit, and export ROIs and sulci directly in a pycortex WebGL viewer — no Inkscape
and no re-generation of the viewer required. You draw on the flattened cortical surface; the
stroke is fitted to a smooth, re-editable bezier curve, drawn as a colored outline and label baked
into the surface.

ROIs are **closed** curves that carry per-hemisphere vertex membership and export to a portable
JSON vertex set. Sulci are **open** curves that carry no vertex data and export as a standalone SVG
whose ``sulci`` layer is the same representation pycortex already uses for sulci.

It ships as a single self-contained bundle (``roidraw.bundle.js``, CSS included) that can be added
to **any** pycortex viewer — a static ``make_static`` export or a freshly generated dynamic viewer.

.. note::

   pycortex-roidraw lives in its own repository and is distributed separately from pycortex. The
   ROIs it exports are a portable JSON **vertex set** (with the editable bezier), independent of
   the Inkscape-based :doc:`surface-defined ROI system <rois>` — they are not read by
   ``get_roi_masks`` and friends. Use it for quick interactive annotation in the viewer; use the
   Inkscape workflow when you need ROI masks in the pycortex Python API.

   Drawn **sulci** are different: they are exported as ``overlays.svg`` markup, so they *are* read
   by pycortex's own machinery (``quickflat``'s sulci overlay, the WebGL viewer, Inkscape) once
   their shape groups are copied into a subject's overlay file — see :ref:`installing-drawn-sulci`.

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

**Export sulci (SVG)** writes a standalone ``sulci.svg`` document whose ``sulci`` layer is in
pycortex's own overlay format: open, unfilled ``<path>`` elements, one inkscape-labeled group per
named sulcus. Trace a sulcus on each hemisphere and give both strokes the same name, and they merge
into a single group with one ``<path>`` per hemisphere — exactly how a hand-authored sulcus such as
``CaS`` is stored.

Sulci carry no vertex data, matching pycortex: there is no ``get_sulci_verts``, and sulci are
display geometry. The ``sulci_labels`` layer is deliberately left empty — pycortex derives each
sulcus's label position from its path geometry when the overlay is loaded, so no label needs to be
written. Sulcus export is one-way: sulci are not re-imported from SVG.

.. _installing-drawn-sulci:

Installing drawn sulci into a subject
-------------------------------------

.. warning::

   Copy the ``<g inkscape:label="…">`` groups out of the exported file's ``sulci_shapes`` group and
   into the **existing** ``sulci_shapes`` group of the subject's ``overlays.svg``.

   Do **not** append the whole ``<g id="sulci">`` layer. ``SVGOverlay`` keys its layers by
   ``inkscape:label``, so a second layer labelled ``sulci`` replaces the subject's own in
   ``SVGOverlay.layers`` — every sulcus already in that file becomes invisible to pycortex.

So, given an exported ``sulci.svg`` containing a group for the central sulcus:

.. code-block:: xml

   <!-- sulci.svg, as downloaded -->
   <svg xmlns="http://www.w3.org/2000/svg"
        xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ...>
     <g inkscape:groupmode="layer" id="sulci" inkscape:label="sulci" style="display:inline">
       <g inkscape:groupmode="layer" id="sulci_shapes" inkscape:label="shapes">

         <g inkscape:groupmode="layer" inkscape:label="CS">     <!-- copy THIS group ... -->
           <path style="fill:none;stroke:white;..." d="M412.55,301.90C..." />
           <path style="fill:none;stroke:white;..." d="M598.31,297.44C..." />
         </g>

       </g>
       <g inkscape:groupmode="layer" id="sulci_labels" inkscape:label="labels" />
     </g>
   </svg>

paste that one ``<g inkscape:label="CS">`` group inside the subject's own
``<g id="sulci_shapes">``, alongside the sulci already there. Then::

    import cortex
    svg = cortex.db.get_overlay('S1')
    svg.sulci['CS']                                   # the drawn sulcus, parsed
    cortex.quickflat.make_figure(volume, with_sulci=True)

The paths' coordinates are already in the overlay's own coordinate system, so nothing needs to be
rescaled. Inkscape can also open the exported file directly.

Full documentation
------------------

See the `pycortex-roidraw README <https://github.com/gallantlab/pycortex-roidraw>`_ for the
complete gesture reference, the JSON schema, the helper scripts, and notes on porting the tool to
other viewer engines.
