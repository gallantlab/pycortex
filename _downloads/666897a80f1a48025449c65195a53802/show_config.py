"""
=====================================================
Finding out where the config and filestore are
=====================================================

Easily locating your config file and filestore locations.
This comes in useful when things don't work because the config file is not set correctly.
"""
from __future__ import print_function
import cortex
from cortex.options import config

##########################################################
# Finding where your config file is.
print(cortex.options.usercfg)

##########################################################
# Finding where the current filestore is.
# Useful for when your subjects don't show up in cortex.db, and all you have is S1.
print(config.get('basic', 'filestore'))

##########################################################
# Finding where pycortex is looking for colormaps.
# Useful for when you get color map not found messages.
print(config.get('webgl', 'colormaps'))

##########################################################
# To look at your config file, it is recommended that you open it with a text editor.
# However, you *can* still look at options from within pycortex.

# sections gets the upper-level sections in the config file
sections = config.sections()
print(sections)

# items gets the option items within a section as a list of key-value pairs.
basic_config = config.items('paths_default')
print(basic_config)