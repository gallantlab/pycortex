"""
================================
Static viewer with html template
================================
A static viewer is a brain viewer that exists permanently on a filesystem
The viewer is stored in a directory that stores html, javascript, data, etc
The viewer directory must be hosted by a server such as nginx.

Pycortex has the functionality of provide a custom html template to specify 
the attributes of the viewer. The file 'my_template.html' included in this
directory has css code which changes the color scheme of the controls 
window. The background is now white, with black text, and the sliders 
are red instead of blue. 

The templates files included with pycortex () have been modified by adding 
the following css code:

    <style type = text/css>
    .dg li:not(.folder) {
        background: white;
        color: black;
        text-shadow: none;
    }
    .dg li:not(.folder):hover {
        background: #d2cfcf;
    }
    .dg .cr.number input[type=text] {
        color: black;
        background: #c3c3c3;
    }
    .dg .cr.function:hover, .dg .cr.boolean:hover {
        background: #d2cfcf;
    }
    .dg .cr.number input[type=text] :hover {
        color: black;
        background: #c3c3c3;
    }
    .dg .c .slider-fg{
        background-color:red
    }
    .dg .c .slider:hover .slider-fg {
        background: red;
    }

    </style>


"""

import cortex

import numpy as np
np.random.seed(1234)

# gather data Volume
volume = cortex.Volume.random(subject='S1', xfmname='fullhead')

# select path for static viewer on disk
viewer_path = '/path/to/store/viewer'

# create viewer using the 'my_template.html' template:
cortex.webgl.make_static(outpath=viewer_path, data=volume,
                        template = 'my_template.html')

# a webserver such as nginx can then be used to host the static viewer
# with the new template