
import os
import subprocess
import tempfile

import numpy as np

from ..options import config
from .. import formats


class ExactGeodesicException(Exception):
    """Raised when exact_geodesic_distance() is unavailable or used improperly

    - to create a fallback to geodesic_distance()
    """
    pass


class ExactGeodesicMixin(object):
    """Mixin for computing exact geodesic distance along surface"""

    def exact_geodesic_distance(self, vertex):
        """Compute exact geodesic distance along surface

        - uses VTP geodesic algorithm

        Parameters
        ----------
        - vertex : int or list of int
            index of vertex or vertices to compute geodesic distance from
        """
        if isinstance(vertex, list):
            return np.vstack(self.exact_geodesic_distance(v) for v in vertex).min(0)
        else:
            return self.call_vtp_geodesic(vertex)

    def call_vtp_geodesic(self, vertex):
        """Compute geodesic distance using VTP method

        VTP Code
        --------
        - uses external authors' implementation of [Qin el al 2016]
        - https://github.com/YipengQin/VTP_source_code
        - vtp code must be compiled separately to produce VTP executable
        - once compiled, place path to VTP executable in pycortex config
        - i.e. in config put:
            [geodesic]
            vtp_path = /path/to/compiled/VTP

        Parameters
        ----------
        - vertex : int
            index of vertex to compute geodesic distance from
        """

        if config.has_option('geodesic', 'vtp_path'):
            vtp_path = config.get('geodesic', 'vtp_path')
        else:
            raise ExactGeodesicException('must set config["geodesic"]["vtp_path"]')

        if not os.path.exists(vtp_path):
            raise ExactGeodesicException('vtp_path does not exist: ' + str(vtp_path))

        # initialize temporary files
        f_obj, tmp_obj_path = tempfile.mkstemp()
        f_output, tmp_output_path = tempfile.mkstemp()

        # create object file
        formats.write_obj(tmp_obj_path, self.pts, self.polys)

        # run algorithm
        cmd = [vtp_path, '-m', tmp_obj_path, '-s', str(vertex), '-o', tmp_output_path]
        subprocess.call(cmd)

        # read output
        with open(tmp_output_path) as f:
            output = f.read()
            distances = np.array(output.split('\n')[:-2], dtype=float)

        if distances.shape[0] == 0:
            raise ExactGeodesicException('VTP error')

        os.close(f_obj)
        os.close(f_output)

        return distances
