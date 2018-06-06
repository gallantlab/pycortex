
import os
import subprocess
import tempfile

import numpy as np

from ..options import config


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

        - uses authors' implementation of [Qin el al 2016]

        Parameters
        ----------
        - vertex : int
            index of vertex to compute geodesic distance from
        """

        vtp_path = config.get('geodesic', 'vtp_path')

        # initialize temporary files
        f_obj, tmp_obj_path = tempfile.mkstemp()
        f_output, tmp_output_path = tempfile.mkstemp()

        # create object file
        self.export_to_obj_file(tmp_obj_path)

        # run algorithm
        cmd = [vtp_path, '-m', tmp_obj_path, '-s', str(vertex), '-o', tmp_output_path]
        subprocess.call(cmd)

        # read output
        with open(tmp_output_path) as f:
            output = f.read()
            distances = np.array(output.split('\n')[:-2], dtype=float)

        if distances.shape[0] == 0:
            raise Exception('VTP error')

        os.close(f_obj)
        os.close(f_output)

        return distances

    def export_to_obj_file(self, path_or_handle):
        """export Surface to Wavefront obj for use by external programs

        https://en.wikipedia.org/wiki/Wavefront_.obj_file

        Parameters
        ----------
        - path_or_handle : str or file handle
            - location to write .obj file to
        """

        if isinstance(path_or_handle, str):
            with open(path_or_handle, 'w') as f:
                return self.export_to_obj_file(f)

        else:
            path_or_handle.write('mtllib none.mtl\n')
            path_or_handle.write('o hemi\n')

            for vertex in self.pts:
                path_or_handle.write('v ' + ' '.join(vertex.astype(str)) + '\n')

            path_or_handle.write('usemtl None\n')
            path_or_handle.write('s off\n')

            for polygon in self.polys:
                polygon = polygon + 1
                path_or_handle.write('f ' + ' '.join(polygon.astype(str)) + '\n')
