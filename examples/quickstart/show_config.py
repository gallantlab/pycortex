"""
=====================================================
Finding out where the config and filestore are
=====================================================

Easily locating your config file and filestore locations.
This comes in useful when things don't work because the config file is not set correctly.
"""
from __future__ import print_function
import cortex

##########################################################
# Finding where your config file is.
print(cortex.options.usercfg)

##########################################################
# Finding where the current filestore is.
# Useful for when your subjects don't show up in cortex.db
print(cortex.options.get('basic', 'filestore'))

############################################