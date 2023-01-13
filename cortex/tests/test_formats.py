import os
import tempfile

import cortex
from cortex.formats import read_gii, write_gii

from numpy.testing import assert_array_equal

def test_write_read_gii():
    wm, polys = cortex.db.get_surf("S1", "wm", "lh")
    with tempfile.TemporaryDirectory() as tmpdir:
        fnout = os.path.join(tmpdir, "out.gii")
        write_gii(fnout, wm, polys)
        wm2, polys2 = read_gii(fnout)
        assert_array_equal(wm, wm2)
        assert_array_equal(polys, polys2)

