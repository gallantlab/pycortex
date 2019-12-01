"""
=================================
Dynamic viewer with html template
=================================
A webgl viewer displays a 3D view of brain data in a web browser

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

# create viewer
cortex.webgl.show(data=volume, template='my_template.html', recache=True)

# a port number will then be output, for example "Started server on port 39140"
# the viewer can then be accessed in a web browser, in this case at "localhost:39140"
# the viewer will display the modifications specified in the template