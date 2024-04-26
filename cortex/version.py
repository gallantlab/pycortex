"""Defines version to be imported in the module and obtained from setup.py
"""
# This file has been copied and modified from the DataLad codebase
# https://github.com/datalad/datalad/blob/master/datalad/version.py
# www.datalad.org
# Many thanks to the DataLad developers

import sys
from os.path import lexists, dirname, join as opj, curdir

# Hard coded version, to be done by release process,
# it is also "parsed" (not imported) by setup.py, that is why assigned as
# __hardcoded_version__ later and not vice versa
#
# NOTE this should have the format of
# NEW_RELEASE.dev0
# since the __version__ will be later automatically parsed by adding the git
# commit hash
# so valid versions will be 1.2.1.dev0, 1.4.dev0, etc
__version__ = '1.2.8'
__hardcoded_version__ = __version__
__full_version__ = __version__

# NOTE: might cause problems with "python setup.py develop" deployments
#  so I have even changed buildbot to use  pip install -e .
moddir = dirname(__file__)
projdir = curdir if moddir == 'cortex' else dirname(moddir)
if lexists(opj(projdir, '.git')):
    # If under git -- attempt to deduce a better "dynamic" version following git
    try:
        from subprocess import Popen, PIPE
        # Note: Popen does not support `with` way correctly in 2.7
        #
        git = Popen(
            ['git', 'describe', '--abbrev=4', '--dirty', '--match', r'[0-9]*\.*'],
            stdout=PIPE, stderr=PIPE,
            cwd=projdir
        )
        if git.wait() != 0:
            raise OSError("Could not run git describe")
        line = git.stdout.readlines()[0]
        _ = git.stderr.readlines()
        # Just take describe and replace initial '-' with .dev to be more "pythonish"
        # Encoding simply because distutils' LooseVersion compares only StringType
        # and thus misses in __cmp__ necessary wrapping for unicode strings
        #
        # remove the version from describe and stick the current version
        # hardcoded in __version__
        dev_suffix = line.strip().decode('ascii').split('-')
        # in case we are at release, we shouldn't have any dev
        if len(dev_suffix) > 1:
            dev_suffix = dev_suffix[1:]
            dev_suffix = ".dev{}".format('-'.join(dev_suffix))
        else:
            dev_suffix = ""
        # remove dev suffix from hardcoded version
        if 'dev' in __version__:
            __version__ = '.'.join(__version__.split('.')[:-1])
        # stick the automatically generated dev version
        __full_version__ = __version__ + dev_suffix
        # To follow PEP440 we can't have all the git fanciness
        __version__ = __full_version__.split('-')[0]
    except (SyntaxError, AttributeError, IndexError):
        raise
    except:
        # just stick to the hard-coded
        pass
