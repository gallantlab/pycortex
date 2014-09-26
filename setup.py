#!/usr/bin/env python

import os
import sys
from numpy.distutils.misc_util import get_numpy_include_dirs

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            'test',
            )).intersection(sys.argv)) > 0:
    # This formulation is taken from nibabel.
    # "setup_egg imports setuptools setup, thus monkeypatching distutils."
    # Turns out, this patching needs to happen before disutils.core.Extension
    # is imported in order to use cythonize()...
    from setuptools import setup
else:
    # Use standard
    from distutils.core import setup

from distutils.command.install import install
from distutils.core import Extension

from Cython.Build import cythonize


def set_default_filestore(prefix, optfile):
    config = configparser.ConfigParser()
    config.read(optfile)
    config.set("basic", "filestore", os.path.join(prefix, "db"))
    config.set("webgl", "colormaps", os.path.join(prefix, "colormaps"))
    with open(optfile, 'w') as fp:
        config.write(fp)

class my_install(install):
    def run(self):
        install.run(self)
        optfile = [f for f in self.get_outputs() if 'defaults.cfg' in f]
        prefix = os.path.join(self.install_data, "share", "pycortex")
        set_default_filestore(prefix, optfile[0])
        self.copy_tree('filestore', prefix)
        for root, folders, files in os.walk(prefix):
            for folder in folders:
                os.chmod(os.path.join(root, folder), 511)
            for fname in files:
                os.chmod(os.path.join(root, fname), 438)

ctm = Extension('cortex.openctm', 
            ['cortex/openctm.pyx',
             'OpenCTM-1.0.3/lib/openctm.c',
             'OpenCTM-1.0.3/lib/stream.c',
             'OpenCTM-1.0.3/lib/compressRAW.c',
             'OpenCTM-1.0.3/lib/compressMG1.c',
             'OpenCTM-1.0.3/lib/compressMG2.c',
             'OpenCTM-1.0.3/lib/liblzma/Alloc.c',
             'OpenCTM-1.0.3/lib/liblzma/LzFind.c',
             'OpenCTM-1.0.3/lib/liblzma/LzmaDec.c',
             'OpenCTM-1.0.3/lib/liblzma/LzmaEnc.c',
             'OpenCTM-1.0.3/lib/liblzma/LzmaLib.c',
            ], libraries=['m'], include_dirs=
            ['OpenCTM-1.0.3/lib/', 
             'OpenCTM-1.0.3/lib/liblzma/'
            ] + get_numpy_include_dirs(),
            define_macros=[
                ('LZMA_PREFIX_CTM', None),
                ('OPENCTM_BUILD', None),
                #('__DEBUG_', None),
            ]
        )
formats = Extension('cortex.formats', ['cortex/formats.pyx'],
                    include_dirs=get_numpy_include_dirs())

setup(name='pycortex',
      version='0.1.1',
      description='Python Cortical mapping software for fMRI data',
      author='James Gao',
      author_email='james@jamesgao.com',
      packages=['cortex', 'cortex.webgl', 'cortex.mapper', 'cortex.dataset', 'cortex.blender', 'cortex.tests'],
      ext_modules=cythonize([ctm, formats]),
      package_data={
            'cortex':[ 
                'svgbase.xml',
                'defaults.cfg',
                'bbr.sch'
            ],
            'cortex.webgl': [
                '*.html', 
                'favicon.ico', 
                'resources/js/*.js',
                'resources/js/ctm/*.js',
                'resources/css/*.css',
                'resources/css/ui-lightness/*.css',
                'resources/css/ui-lightness/images/*',
                'resources/images/*'
            ]
      },
      requires=['mayavi', 'lxml', 'numpy', 'scipy (>=0.9.0)', 'tornado (>3.1)', 'shapely', 'html5lib', 'h5py (>=2.3)', 'numexpr'],
      cmdclass=dict(install=my_install),
      include_package_data=True,
      test_suite='nose.collector'
)

