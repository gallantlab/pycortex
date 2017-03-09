
WebGL Viewer Overview
=====================

WebGL viewers enable interactive 3D displays of brain data in the web browser using a simple set of commands (see demo examples). The use of a web browser as the front end for pycortex also allows an unprecedented level of interactivity. For example, the anatomical surface can be flattened interactively simply by dragging a slider. This interactive design helps the user to develop a clear sense of the correspondence between flattened and folded surfaces. 

- adjust camera angle and zoom in real time
- toggle display of ROIs
- switch between multiple datasets
- flatten and inflate cortical surface
- create scripted animations
- select points by clicking on them
- switch colormaps

Pycortex can also display temporally varying time-series data on the cortical surface in real time. This allows simultaneous visualization of the experimental paradigm and the functional data in real time 


It is simple to post pycortex visualizations to a web page for public viewing. These static visualizations are generated using a simple command that generates a single web page with most resources embedded directly. The surface structure, data, and the webpage can then be posted to any public facing web site. For example, the online Neurovault data repository (http://neurovault.org) now makes use of pycortex, and any fMRI data uploaded to Neurovault can be visualized automatically in pycortex. These visualizations are visible at a static web address that can be referenced in papers and shared with anyone with a web browser.



Keyboard Shortcuts
------------------

There are many keyboard shortcuts that allow for more fluid interaction witht he WebGL viewer. A complete list of keyboard shortcuts and also be displayed in the viewer by pressing the `h` key.


=== ====================================
Key Action
=== ====================================
f   flatten brain
i 	inflate brain
r   fold brain into original coordinates
+/-	switch between datasets
=== ====================================


WebGL viewer functionality
--------------------------

The data display can be modified interactively in numerous ways. The dynamic view has two sliding windows that contain display options. The large slider at the bottom linearly interpolates the shape of the cortical mesh between the original (folded) anatomical, inflated, and flattened surfaces. This allows the unfolding process to be visualized continuously, and it clarifies the correspondence between 3D anatomical features and the cortical flatmap. The sliding window located at the top contains options that change how the data is displayed. Different colormaps can be selected and the colormap ranges can be altered dynamically. 2D colormaps are also supported, allowing two datasets to be contrasted simultaneously. Multiple datasets can be loaded and compared directly by simply toggling between them. Sliders are provided to change the transparency of the dropout, overlay, data, and curvature layers.

As explained earlier, pycortex uses custom shaders that implement pixel-based mapping. During 3D graphics rendering, the color of each pixel is determined by some predefined code at the fragment shading step. Under a traditional fixed-function pipeline, fragment shading is performed by a rasterizer that implements vertex-based mapping (Woo et al., 1999). In contrast, the fragment shader in pycortex projects each pixel into the functional space in 3D, and then samples the underlying volume data by reading from a texture. Nearest-neighbor or trilinear sampling is automatically performed by OpenGL when the data is read from the texture. This generates a fully interactive and accurate real-time visualization.

The webgl module contains code that parses and generates the HTML and javascript code required to display surface data in a web browser. It provides two possible use cases: a dynamic view that can be controlled by a back end python web server, and a static view that generates static HTML files for upload into an existing web server. The OpenCTM library (Geelnard, 2009) is used to compress the surface mesh into a form that can be utilized by the web browser. If a dynamic view is requested, the webgl module sets up a local web server with all the required surface and data files accessible to the web browser. If a static view is requested, all HTML and javascript code is embedded into a single HTML document and saved to a set of files. Data (in the form of compressed mosaic images) and surface structures are stored separately. These standalone visualizations can then be copied to a web server to be shared with colleagues, included as links in published articles, or shared online with a broad audience.

Pycortex also includes a javascript plugin architecture that allows new interactive visualizations to be developed easily. For example, the static viewer released with Huth et al. (2012) http://gallantlab.org/brainviewer/huthetal2012/ contains a plugin that allows the user to visualize how 1765 distinct semantic features are mapped across the cortical surface (Figure 7). Clicking a point on the brain picks the closest voxel and the viewer displays the semantic category tuning for the associated voxel.

Finally, pycortex provides a bi-directional communication framework between python and javascript, so that actions in javascript can be scripted and manipulated in python. This powerful interaction dynamic allows exploratory data analysis in a way never before possible for fMRI.



