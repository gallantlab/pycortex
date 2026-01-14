"""Defines version to be imported in the module and obtained from setup.py
"""
# Version is now automatically managed by setuptools-scm
# which generates cortex/_version.py during build

try:
    from cortex._version import version as __version__
    __full_version__ = __version__
except ImportError:
    # Fallback for development environments without setuptools-scm
    __version__ = '1.3.0.dev0'
    __full_version__ = __version__

__hardcoded_version__ = __version__
