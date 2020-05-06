"""Module containing utils for testing"""
import subprocess as sp
from shutil import which


def has_installed(name):
    return which(name) is not None


def inkscape_version():
    if not has_installed('inkscape'):
        return None
    cmd = 'inkscape --version'
    output = sp.check_output(cmd.split(), stderr=sp.PIPE)
    # b'Inkscape 1.0 (4035a4f, 2020-05-01)\n'
    version = output.split()[1]
    if isinstance(version, bytes):
        version = version.decode('utf-8')
    return version


INKSCAPE_VERSION = inkscape_version()


