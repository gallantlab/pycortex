import cortex
import os
import numpy as np

PATH = os.path.expanduser("~/www/webgl/cortex_test_2")
SUBJECT = "AH"
XFMNAME = "AH_huth"
DATA = np.random.randn(31,100,100)

cortex.webgl.view.make_static(PATH, DATA, SUBJECT, XFMNAME)

