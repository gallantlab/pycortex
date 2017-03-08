import json
import numpy as np

from .views import Dataview, Volume, Vertex
from .braindata import VolumeData, VertexData, _hash
from ..database import db

from .. import options
default_cmap = options.config.get("basic", "default_cmap")

class DataviewRGB(Dataview):
    """Abstract base class for RGB data views.
    """
    def __init__(self, subject=None, alpha=None, description="", state=None, **kwargs):
        self.alpha = alpha
        self.subject = self.red.subject
        self.movie = self.red.movie
        self.description = description
        self.state = state
        self.attrs = kwargs
        if 'priority' not in self.attrs:
            self.attrs['priority'] = 1

        # If movie, make sure each channel has the same number of time points
        if self.red.movie:
            if not self.red.data.shape[0] == self.green.data.shape[0] == self.blue.data.shape[0]:
                raise ValueError("For movie data, all three channels have to be the same length")

    def uniques(self, collapse=False):
        if collapse:
            yield self
        else:
            yield self.red
            yield self.green
            yield self.blue
            if self.alpha is not None:
                yield self.alpha

    def _write_hdf(self, h5, name="data", xfmname=None):
        self._cls._write_hdf(self.red, h5)
        self._cls._write_hdf(self.green, h5)
        self._cls._write_hdf(self.blue, h5)

        alpha = None
        if self.alpha is not None:
            self._cls._write_hdf(self.alpha, h5)
            alpha = self.alpha.name

        data = [self.red.name, self.green.name, self.blue.name, alpha]
        viewnode = Dataview._write_hdf(self, h5, name=name, 
            data=[data], xfmname=xfmname)

        return viewnode

    def to_json(self, simple=False):
        sdict = super(DataviewRGB, self).to_json(simple=simple)

        if simple:
            sdict['name'] = self.name
            sdict['subject'] = self.subject
            sdict['min'] = 0
            sdict['max'] = 255
        else:
            sdict['data'] = [self.name]
            sdict['cmap'] = [default_cmap]
            sdict['vmin'] = [0]
            sdict['vmax'] = [255]
        return sdict

class VolumeRGB(DataviewRGB):
    """
    Contains RGB (or RGBA) colors for each voxel in a volumetric dataset. 
    Includes information about the subject and transform for the data.

    Each color channel is represented as a separate Volume object (these can 
    either be supplied explicitly as Volume objects or implicitly as numpy 
    arrays). The vmin for each Volume will be mapped to the minimum value for
    that color channel, and the vmax will be mapped to the maximum value.

    Parameters
    ----------
    red : ndarray or Volume
        Array or Volume that represents the red component of the color for each 
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume.
    green : ndarray or Volume
        Array or Volume that represents the green component of the color for each 
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume.
    blue : ndarray or Volume
        Array or Volume that represents the blue component of the color for each 
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        red must be a Volume from which the subject can be extracted.
    xfmname : str, optional
        Transform name. Must exist in the pycortex database. If not given,
        red must be a Volume from which the subject can be extracted.
    alpha : ndarray or Volume, optional
        Array or Volume that represents the alpha component of the color for each 
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume. If
        None, all voxels will be assumed to have alpha=1.0.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    state : optional
        TODO: WHAT THE FUCK IS THIS
    **kwargs
        All additional arguments in kwargs are passed to the VolumeData and 
        Dataview.

    """
    _cls = VolumeData
    def __init__(self, red, green, blue, subject=None, xfmname=None, alpha=None, description="", 
        state=None, **kwargs):
        if isinstance(red, VolumeData):
            if not isinstance(green, VolumeData) or red.subject != green.subject:
                raise TypeError("Invalid data for green channel")
            if not isinstance(blue, VolumeData) or red.subject != blue.subject:
                raise TypeError("Invalid data for blue channel")
            self.red = red
            self.green = green
            self.blue = blue
        else:
            if subject is None or xfmname is None:
                raise TypeError("Subject and xfmname are required")
            self.red = Volume(red, subject, xfmname)
            self.green = Volume(green, subject, xfmname)
            self.blue = Volume(blue, subject, xfmname)


        if alpha is None:
            alpha = np.ones(self.red.volume.shape)

        if not isinstance(alpha, Volume):
            alpha = Volume(alpha, self.red.subject, self.red.xfmname)

        self.alpha = alpha

        if self.red.xfmname == self.green.xfmname == self.blue.xfmname == self.alpha.xfmname:
            self.xfmname = self.red.xfmname
        else:
            raise ValueError('Cannot handle different transforms per volume')

        super(VolumeRGB, self).__init__(subject, alpha, description=description, state=state, **kwargs)


    def to_json(self, simple=False):
        sdict = super(VolumeRGB, self).to_json(simple=simple)
        if simple:
            sdict['shape'] = self.red.shape
        else:
            sdict['xfm'] = [list(np.array(db.get_xfm(self.subject, self.xfmname, 'coord').xfm).ravel())]

        return sdict

    @property
    def volume(self):
        """5-dimensional volume (t, z, y, x, rgba) with data that has been mapped
        into 8-bit unsigned integers that correspond to colors.
        """
        volume = []
        for dv in (self.red, self.green, self.blue, self.alpha):
            vol = dv.volume.copy()
            if vol.dtype != np.uint8:
                if dv.vmin is None:
                    if vol.min() < 0:
                        vol -= vol.min()
                else:
                    vol -= dv.vmin

                if dv.vmax is None:
                    if vol.max() > 1:
                        vol /= vol.max()
                else:
                    vol /= dv.vmax - dv.vmin

                vol = (np.clip(vol, 0, 1) * 255).astype(np.uint8)
            volume.append(vol)

        return np.array(volume).transpose([1, 2, 3, 4, 0])

    def __repr__(self):
        return "<RGB volumetric data for (%s, %s)>"%(self.red.subject, self.red.xfmname)

    def __hash__(self):
        return hash(_hash(self.volume))

    @property
    def name(self):
        return "__%s"%_hash(self.volume)[:16]

    def _write_hdf(self, h5, name="data"):
        return super(VolumeRGB, self)._write_hdf(h5, name=name, xfmname=[self.xfmname])

class VertexRGB(DataviewRGB):
    """
    Contains RGB (or RGBA) colors for each vertex in a surface dataset. 
    Includes information about the subject.

    Each color channel is represented as a separate Vertex object (these can 
    either be supplied explicitly as Vertex objects or implicitly as numpy 
    arrays). The vmin for each Vertex will be mapped to the minimum value for
    that color channel, and the vmax will be mapped to the maximum value.

    Parameters
    ----------
    red : ndarray or Vertex
        Array or Vertex that represents the red component of the color for each 
        voxel. Can be a 1D or 3D array (see Vertex for details), or a Vertex.
    green : ndarray or Vertex
        Array or Vertex that represents the green component of the color for each 
        voxel. Can be a 1D or 3D array (see Vertex for details), or a Vertex.
    blue : ndarray or Vertex
        Array or Vertex that represents the blue component of the color for each 
        voxel. Can be a 1D or 3D array (see Vertex for details), or a Vertex.
    subject : str, optional
        Subject identifier. Must exist in the pycortex database. If not given,
        red must be a Vertex from which the subject can be extracted.
    alpha : ndarray or Vertex, optional
        Array or Vertex that represents the alpha component of the color for each 
        voxel. Can be a 1D or 3D array (see Vertex for details), or a Vertex. If
        None, all vertices will be assumed to have alpha=1.0.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    state : optional
        TODO: WHAT THE FUCK IS THIS
    **kwargs
        All additional arguments in kwargs are passed to the VertexData and 
        Dataview.

    """
    _cls = VertexData
    def __init__(self, red, green, blue, subject=None, alpha=None, description="", 
        state=None, **kwargs):

        if isinstance(red, VertexData):
            if not isinstance(green, VertexData) or red.subject != green.subject:
                raise TypeError("Invalid data for green channel")
            if not isinstance(blue, VertexData) or red.subject != blue.subject:
                raise TypeError("Invalid data for blue channel")
            self.red = red
            self.green = green
            self.blue = blue
        else:
            if subject is None:
                raise TypeError("Subject name is required")
            self.red = Vertex(red, subject)
            self.green = Vertex(green, subject)
            self.blue = Vertex(blue, subject)

        super(VertexRGB, self).__init__(subject, alpha, description=description, 
            state=state, **kwargs)

    @property
    def vertices(self):
        """3-dimensional volume (t, v, rgba) with data that has been mapped
        into 8-bit unsigned integers that correspond to colors.
        """
        alpha = self.alpha
        if alpha is None:
            alpha = np.ones_like(self.red.data)
        if not isinstance(alpha, Vertex):
            alpha = Vertex(alpha, self.subject)

        verts = []
        for dv in (self.red, self.green, self.blue, alpha):
            vert = dv.vertices.copy()
            if vert.dtype != np.uint8:
                if dv.vmin is None:
                    if vert.min() < 0:
                        vert -= vert.min()
                else:
                    vert -= dv.vmin

                if dv.vmax is None:
                    if vert.max() > 1:
                        vert /= vert.max()
                else:
                    vert /= dv.vmax - dv.vmin

                vert = (np.clip(vert, 0, 1) * 255).astype(np.uint8)
            verts.append(vert)

        return np.array(verts).transpose([1, 2, 0])

    def to_json(self, simple=False):
        sdict = super(VertexRGB, self).to_json(simple=simple)
        
        if simple:
            sdict.update(dict(split=self.red.llen, frames=self.vertices.shape[0]))

        return sdict

    @property
    def left(self):
        return self.vertices[:,:self.red.llen]

    @property
    def right(self):
        return self.vertices[:,self.red.llen:]

    def __repr__(self):
        return "<RGB vertex data for (%s)>"%(self.subject)

    def __hash__(self):
        return hash(_hash(self.vertices))

    @property
    def name(self):
        return "__%s"%_hash(self.vertices)[:16]
