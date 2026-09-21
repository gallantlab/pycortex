"""Pin ``cortex.db`` to the filestore bundled with pycortex.

The filestore is a configured path (``basic.filestore`` in ``options.cfg``),
so on a machine with a real filestore the suite would otherwise run against
whatever subjects that machine happens to have. Every test here uses the demo
subject ``S1``, and the reference renders in ``reference_images/`` are pixel
comparisons against the bundled one; a lab filestore with its own ``S1`` would
fail them for reasons that have nothing to do with the code under test, and
would collect flatmap caches along the way.
"""
import os
import sys

import cortex
from cortex import database, options


def _bundled_filestore():
    """The demo filestore shipped alongside the installed ``cortex`` package."""
    pkgdir = os.path.dirname(os.path.abspath(cortex.__file__))
    candidates = [
        # Source checkout or editable install: filestore/ sits beside cortex/.
        os.path.join(pkgdir, os.pardir, "filestore", "db"),
        # Installed: setup.py copies filestore/ to <install_base>/share/pycortex.
        os.path.join(sys.prefix, "share", "pycortex", "db"),
    ]
    for path in candidates:
        path = os.path.realpath(path)
        if os.path.isdir(path):
            return path
    raise RuntimeError(
        "could not locate the filestore bundled with pycortex; looked in "
        + ", ".join(candidates)
    )


FILESTORE = _bundled_filestore()

options.config.set("basic", "filestore", FILESTORE)
database.default_filestore = FILESTORE
# The `filestore=default_filestore` defaults throughout database.py were bound
# at import, so the singleton has to be repointed by hand. Everything reached
# through it (SubjectDB and below) is passed `self.filestore` explicitly.
cortex.db.filestore = FILESTORE
cortex.db.reload_subjects()


def pytest_report_header(config):
    return f"pycortex filestore: {FILESTORE}"
