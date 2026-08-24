
WebGL Viewer Overview
=====================

WebGL viewers enable interactive 3D displays of brain data in the web browser using a simple set of commands (see demo examples). The use of a web browser as the front end for pycortex also allows an unprecedented level of interactivity. For example, the anatomical surface can be flattened interactively simply by dragging a slider. This interactive design helps the user to develop a clear sense of the correspondence between flattened and folded surfaces. 

- adjust camera angle and zoom in real time
- toggle display of ROIs and various labels
- switch between multiple datasets
- flatten and inflate cortical surface
- create scripted animations
- select points by clicking on them
- switch between colormaps

Pycortex can also display temporally varying time-series data on the cortical surface in real time. This allows simultaneous visualization of the experimental paradigm and the functional data in real time 

It is simple to post pycortex visualizations to a web page for public viewing. These static visualizations are generated using a simple command that generates a single web page with most resources embedded directly. The surface structure, data, and the webpage can then be posted to any public facing web site. For example, the online Neurovault data repository (http://neurovault.org) now makes use of pycortex, and any fMRI data uploaded to Neurovault can be visualized automatically in pycortex. These visualizations are visible at a static web address that can be referenced in papers and shared with anyone with a web browser.


.. seealso::

   You can draw, edit, and export ROIs directly in the WebGL viewer with the
   :doc:`pycortex-roidraw </roidraw>` add-on.

Using the WebGL Viewer
----------------------

There are two ways to create a WebGL viewer. A **dynamic viewer** is temporary viewer that is hosted by the python process that generated it. A **static viewer** is a viewer that is saved permanently to disk and will persist beyond the lifetime of the python process. Using a static viewer requires hosting the created directory with a webserver such as nginx.


Keyboard Shortcuts
^^^^^^^^^^^^^^^^^^

There are many keyboard shortcuts that allow for more fluid interaction with the WebGL viewer. A complete list of keyboard shortcuts can also be displayed in the viewer by pressing the **h** key.

=============   ====================================
Key             Action
=============   ====================================
f               flatten brain
i 	            inflate brain
k 	            inflate brain to surface cuts
r               fold brain into original coordinates (without moving camera)
t               reset entire view (fold brain, reset camera position, rotation, and zoom)
shift + wheel   change inflation level
p               show pial surface
u               show fiducial surface
y               show white matter surface
l               toggle labels
h               toggle keyboard shortcut overview
+/-	            switch between datasets
e               toggle X slice
d               toggle Y slice
c               toggle Z slice
q/w             switch between X slices
a/s             switch between Y slices
z/x             switch between Z slices
o               toggle data opacity
m               toggle multiple layers
alt + wheel     change cortical depth
shift + l       toggle left hemisphere
shift + r       toggle right hemisphere
shift + s       save current view as png
=============   ====================================


Mouse and Trackpad Controls
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The camera is driven directly from the 3D view.

============================   ====================================
Action                         Effect
============================   ====================================
left-click and drag            rotate the brain (pan, when flattened)
shift or middle-click + drag   pan the view
ctrl or right-click + drag     zoom in and out
scroll wheel                   zoom in and out
pinch                          zoom in and out
============================   ====================================

Pinch-to-zoom works both on laptop trackpads and on touchscreens, and zooms the
brain rather than the surrounding web page.

View Controls
^^^^^^^^^^^^^

The **Open Controls** button in the upper right corner opens a control panel for many different display options.

Camera Controls
***************

======== =====================================================
name     description
======== =====================================================
azimuth  camera rotation within the xy plane
altitude angle above the xy plane
radius   zoom level
fold     fold brain into original coordinates
reset    reset entire view (position, rotation, zoom, fold)
inflate  inflate cortical surface
flatten  flatten cortical surface
======== =====================================================


Surface Controls
****************

=================== ==================================
name                description
=================== ==================================
unfold              level of unfolding
pivot               angle between hemispheres
shift               distance between hemispheres
depth               cortical depth
bumpy_flatmap       give the flatmap relief, see below
bumpy_flatmap_scale how much to exaggerate that relief
left                toggle left hemisphere
right               toggle right hemisphere
=================== ==================================


Lighting Controls
*****************

These live in the ``lighting`` sub-menu of the surface controls.

==================== ==========================================================
name                 description
==================== ==========================================================
topleft_lighting     crossfade from the default headlight (0) to a light from
                     the upper left (1)
uniform_illumination how flat and shadowless the illumination is, 0 to 1
specularity          specular reflection level
==================== ==========================================================

Unfolding towards a flatmap drives illumination to fully uniform and
specularity to zero, since a flat sheet has no shape for directional lighting to
reveal. Turning on ``bumpy_flatmap`` gives the flatmap real relief again:
unfolding then drives top-left lighting to 1 instead, which is the direction
that reads as shaded relief. Like ``pivot``, the sliders move to show the values
in effect and can still be dragged afterwards; the next change to ``unfold`` or
``bumpy_flatmap`` drives them back from the configured defaults.

The relief is the shape a slab of cortex would actually take if it were peeled
off the white matter and laid flat: thicker over gyri, which flattening
compresses, and thinner over sulci, which it stretches. It is computed by
relaxing the cortical slab as an elastic solid, which lets the pial surface
slide sideways as it settles rather than sitting in a vertical column above the
white matter -- see :class:`cortex.polyutils.FlatSlab`. That calculation takes
under a minute, so it is done once when the flatmap is imported and cached in
the subject's database entry. It is deliberately not generated on demand: a
subject imported before this existed gets a flatmap with no relief, and the
viewer says so, rather than stopping while it is computed. Run
``cortex.db.get_surfinfo(subject, type='bumpy_flatmap')`` once for such a
subject and it will be picked up from then on.

``bumpy_flatmap_scale`` exaggerates the relief, and is a slider in the surface
controls as well as a setting. At 1.0 the bumps are at their true scale, which
for a 2-5 mm slab is subtle next to a whole flatmap; larger values make the
folding easier to read. The exaggeration is vertical only, as on a topographic
map: the height above the sheet grows and the sideways sliding of the pial
surface does not, so turning the slider up does not move the relief around
relative to the data drawn underneath it. The shading follows, since a height
field scaled by *s* has a normal whose in-plane components scale by *s* and
whose out-of-plane component does not. The slider starts wherever the
configuration file put it, and runs from 0 to five times true scale -- or to
twice the configured value if that is already higher. It does nothing while
``bumpy_flatmap`` is off, since there is no relief to scale. Being a display
setting, it does not invalidate the cached geometry, so it can be dragged
around freely.

The relief appears over the second half of the unfold, between the inflated
surface and the flatmap, and the anatomical and inflated surfaces are untouched
by it. The offsets are in the flatmap's own coordinates, which have no meaning
on a folded surface.


Overlay Controls
****************

======= ===================
name 	description
======= ===================
visible toggle roi outlines
labels  toggle roi labels
======= ===================


WebGL Viewer Technical Details
------------------------------

Pycortex uses custom shaders that implement pixel-based mapping. During 3D graphics rendering, the color of each pixel is determined by some predefined code at the fragment shading step. Under a traditional fixed-function pipeline, fragment shading is performed by a rasterizer that implements vertex-based mapping (Woo et al., 1999). In contrast, the fragment shader in pycortex projects each pixel into the functional space in 3D, and then samples the underlying volume data by reading from a texture. Nearest-neighbor or trilinear sampling is automatically performed by OpenGL when the data is read from the texture. This generates a fully interactive and accurate real-time visualization.

The webgl module contains code that parses and generates the HTML and javascript code required to display surface data in a web browser. It provides two possible use cases: a dynamic view that can be controlled by a back end python web server, and a static view that generates static HTML files for upload into an existing web server. The OpenCTM library (Geelnard, 2009) is used to compress the surface mesh into a form that can be utilized by the web browser. If a dynamic view is requested, the webgl module sets up a local web server with all the required surface and data files accessible to the web browser. If a static view is requested, all HTML and javascript code is embedded into a single HTML document and saved to a set of files. Data (in the form of compressed mosaic images) and surface structures are stored separately. These standalone visualizations can then be copied to a web server to be shared with colleagues, included as links in published articles, or shared online with a broad audience.

The data display can be modified interactively in numerous ways. The dynamic view has two sliding windows that contain display options. The **unfold** slider in the control panel linearly interpolates the shape of the cortical mesh between the original (folded) anatomical, inflated, and flattened surfaces. This allows the unfolding process to be visualized continuously, and it clarifies the correspondence between 3D anatomical features and the cortical flatmap. The sliding window located at the top contains options that change how the data is displayed. Different colormaps can be selected and the colormap ranges can be altered dynamically. 2D colormaps are also supported, allowing two datasets to be contrasted simultaneously. Multiple datasets can be loaded and compared directly by simply toggling between them. Sliders are provided to change the transparency of the dropout, overlay, data, and curvature layers.

Pycortex also includes a javascript plugin architecture that allows new interactive visualizations to be developed easily. For example, the static viewer released with Huth et al. (2012) http://gallantlab.org/brainviewer/huthetal2012/ contains a plugin that allows the user to visualize how 1765 distinct semantic features are mapped across the cortical surface (Figure 7). Clicking a point on the brain picks the closest voxel and the viewer displays the semantic category tuning for the associated voxel.

Finally, pycortex provides a bi-directional communication framework between python and javascript, so that actions in javascript can be scripted and manipulated in python. This powerful interaction dynamic allows exploratory data analysis in a way never before possible for fMRI.

For further details see *Gao JS, Huth AG, Lescroart MD and Gallant JL (2015) Pycortex: an interactive surface visualizer for fMRI. Front. Neuroinform. 9:23. doi: 10.3389/fninf.2015.00023*
