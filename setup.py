#!/usr/bin/env python

import os
from glob import glob
from numpy.distutils.misc_util import get_numpy_include_dirs

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from setuptools import setup, Extension
from setuptools.command.install import install

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
        prefix = os.path.join(self.install_base, "share", "pycortex")
        set_default_filestore(prefix, optfile[0])
        self.copy_tree('filestore', prefix)
        for root, folders, files in os.walk(prefix):
            for folder in folders:
                os.chmod(os.path.join(root, folder), 511)
            for fname in files:
                os.chmod(os.path.join(root, fname), 438)


# Files listed in MANIFEST.in are not copied in wheels, use data_files instead.
# data_files = [
#     ('share/pycortex/colormaps', 'filestore/colormaps/RdBu.png'),
#     ...
# ]
data_files = [
    # [10:] to remove "filestore/"
    ('share/pycortex/' + os.path.dirname(file)[10:], [file])
    for file in glob('filestore/**/*', recursive=True)
    if os.path.isfile(file)
]


# Modified from DataLad codebase to load version from pycortex/version.py
def get_version():
    """Load version from version.py without entailing any imports
    Parameters
    ----------
    name: str
      Name of the folder (package) where from to read version.py
    """
    # This might entail lots of imports which might not yet be available
    # so let's do ad-hoc parsing of the version.py
    with open(os.path.abspath('cortex/version.py')) as f:
        version_lines = list(filter(lambda x: x.startswith('__version__'), f))
    assert (len(version_lines) == 1)
    return version_lines[0].split('=')[1].strip(" '\"\t\n")


ctm = Extension('cortex.openctm', [
            'cortex/openctm.pyx',
            'OpenCTM-1.0.3/lib/openctm.c',
            'OpenCTM-1.0.3/lib/stream.c',
            'OpenCTM-1.0.3/lib/compressRAW.c',
            'OpenCTM-1.0.3/lib/compressMG1.c',
            'OpenCTM-1.0.3/lib/compressMG2.c',
            'OpenCTM-1.0.3/lib/liblzma/Alloc.c',
            'OpenCTM-1.0.3/lib/liblzma/LzFind.c',
            'OpenCTM-1.0.3/lib/liblzma/LzmaDec.c',
            'OpenCTM-1.0.3/lib/liblzma/LzmaEnc.c',
            'OpenCTM-1.0.3/lib/liblzma/LzmaLib.c',], 
            libraries=['m'], include_dirs=[
            'OpenCTM-1.0.3/lib/', 
            'OpenCTM-1.0.3/lib/liblzma/'] + get_numpy_include_dirs(),
            define_macros=[
                ('LZMA_PREFIX_CTM', None),
                ('OPENCTM_BUILD', None),
                #('__DEBUG_', None),
            ]
        )
formats = Extension('cortex.formats', ['cortex/formats.pyx'],
                    include_dirs=get_numpy_include_dirs())

DISTNAME = 'pycortex'
# VERSION needs to be modified under cortex/version.py
VERSION = get_version()
DESCRIPTION = 'Python Cortical mapping software for fMRI data'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = 'James Gao'
AUTHOR_EMAIL = 'james@jamesgao.com'
LICENSE = '2-clause BSD license'
URL = 'http://gallantlab.github.io/pycortex'
DOWNLOAD_URL = URL
with open('requirements.txt') as f:
    INSTALL_REQUIRES = f.read().split()


setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=[
          'cortex',
          'cortex.webgl',
          'cortex.mapper',
          'cortex.dataset',
          'cortex.blender',
          'cortex.tests',
          'cortex.quickflat',
          'cortex.polyutils',
          'cortex.export'
      ],
      data_files=data_files,
      ext_modules=cythonize([ctm, formats]),
      package_data={
            'cortex': [
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
                'resources/css/images/*',
                'resources/css/ui-lightness/*.css',
                'resources/css/ui-lightness/images/*',
                'resources/images/*'
            ]
            },
      setup_requires=['Cython', 'numpy'],
      install_requires=INSTALL_REQUIRES,
      cmdclass=dict(install=my_install),
      include_package_data=True,
      classifiers=[
          'Development Status :: 6 - Mature',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering :: Visualization'
      ]
)
