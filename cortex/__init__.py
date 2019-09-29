# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set fileencoding=utf-8 ft=python sts=4 ts=4 sw=4 et:
import warnings
import tempfile
import tarfile
import wget
from cortex.dataset import Dataset, Volume, Vertex, VolumeRGB, VertexRGB, Volume2D, Vertex2D
from cortex import align, volume, quickflat, webgl, segment, options
from cortex.database import db
from cortex.utils import *
from cortex.quickflat import make_figure as quickshow
from cortex.volume import mosaic, unmask
import cortex.export

try:
    from cortex import formats
except ImportError:
    raise ImportError("Either are running pycortex from the source directory, or the build is broken. "
                      "If your current working directory is 'cortex', where pycortex is installed, then change this. "
                      "If your current working directory is somewhere else, then you may have to rebuild pycortex.")

load = Dataset.from_file

try:
    from cortex import webgl
    from cortex.webgl import show as webshow
except ImportError:
    pass

try:
    from cortex import anat
except ImportError:
    pass

# Create deprecated interface for database
class dep(object):
    def __getattr__(self, name):
        warnings.warn("cortex.surfs is deprecated, use cortex.db instead", Warning)
        return getattr(db, name)
    def __dir__(self):
        warnings.warn("cortex.surfs is deprecated, use cortex.db instead", Warning)
        return db.__dir__()

surfs = dep()

def download_subject(subject_id='fsaverage', url=None, pycortex_store=None):
    """Download subjects to pycortex store
    
    Parameters
    ----------
    subject_id : string
        subject identifying string in pycortex. This assumes that 
        the file downloaded from some URL is of the form <subject_id>.tar.gz
    url: string or None
        URL from which to download. Not necessary to specify for subjects 
        known to pycortex (None is OK). Known subjects will have a default URL. 
        Currently,the only known subjects is 'fsaverage', but there are plans 
        to add more in the future.
    pycortex_store : string or None
        Directory to which to put the new subject folder. If None, defaults to
        the `filestore` variable specified in the pycortex config file. 
    
    """
    # Lazy imports
    import tarfile
    import wget
    import os
    # Map codes to URLs; more coming eventually
    id_to_url = dict(fsaverage='https://ndownloader.figshare.com/files/17827577?private_link=4871247dce31e188e758',
                     )
    if url is None:
        if not subject_id in id_to_url:
            raise ValueError('Unknown subject_id!')
        url = id_to_url[subject_id]
    print("Downloading from: {}".format(url))
    # Download to temp dir
    tmp_dir = tempfile.gettempdir()
    wget.download(url , tmp_dir)
    print('Downloaded subject {} to {}'.format(subject_id, tmp_dir))
    # Un-tar to pycortex store
    if pycortex_store is None:
        # Default location is config file pycortex store.
        pycortex_store = options.config.get('basic', 'filestore')
    pycortex_store = os.path.expanduser(pycortex_store)
    with tarfile.open(os.path.join(tmp_dir, subject_id + '.tar.gz'), "r:gz") as tar:
        print("Extracting subject {} to {}".format(subject_id, pycortex_store))
        tar.extractall(path=pycortex_store)

import sys
if sys.version_info.major == 2:
    stdout = sys.stdout
    reload(sys)
    sys.setdefaultencoding('utf8')
    sys.stdout = stdout

__version__ = '1.1.dev0'
