'''
================
Plot Volume Data
================

This plots example data onto an example subject, S1
'''

import cortex
import numpy as np

test_data = np.random.randn(31,100,100)

dv = cortex.Volume(test_data, "S1", "fullhead")
cortex.quickshow(dv)

