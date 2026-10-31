"""Defines version to be imported in the module and obtained from setup.py
"""
# Version is now automatically managed by setuptools-scm
# which generates cortex/_version.py during build

try:
    from cortex._version import version as __version__
    __full_version__ = __version__
except ImportError:
    # Build ran without setuptools-scm (e.g. --no-build-isolation without it
    # installed), so _version.py was never generated. Use an obviously-wrong
    # placeholder rather than a real-looking version that goes stale.
    __version__ = '0+unknown'
    __full_version__ = __version__

__hardcoded_version__ = __version__
