#!/usr/bin/env python

from distutils.core import setup, Extension

setup(name='pycortex',
      version='0.1.0',
      description='Python Cortical mapping software for fMRI data',
      author='James Gao',
      author_email='james@jamesgao.com',
      packages=['cortex', 'cortex.webgl'],
      ext_modules=[
        Extension('cortex._vtkctm', 
            ['cortex/vtkctm.c',
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
            ], define_macros=[
                  ('LZMA_PREFIX_CTM', None),
                  ('OPENCTM_BUILD', None),
            ]
        )
      ],
      package_data={
            'cortex':[ 'svgbase.xml' ],
            'cortex.webgl':
                  ['*.html', 
                   'favicon.ico', 
                   'resources/js/*.js',
                   'resources/js/ctm/*.js',
                   'resources/css/*.css',
                   'resources/css/ui-lightness/*.css',
                   'resources/css/ui-lightness/images/*',
                   'resources/colormaps/*.png',
                   'resources/images/*',
                  ]
            },
      requires=['mayavi', 'lxml', 'numpy', 'scipy (>=0.9.0)', 'tornado (>2.1)', 'shapely', 'html5lib'],
)
