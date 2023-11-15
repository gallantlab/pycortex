import numpy as np
import colorsys
import warnings

from .views import Dataview, Volume, Vertex
from .braindata import VolumeData, VertexData, _hash
from ..database import db

from .. import options
default_cmap = options.config.get("basic", "default_cmap")


class Colors(object):
    """
    Set of known colors
    """
    RoseRed = (237, 35, 96)
    LimeGreen = (141, 198, 63)
    SkyBlue = (0, 176, 218)
    DodgerBlue = (30, 144, 255)
    Red = (255, 000, 000)
    Green = (000, 255, 000)
    Blue = (000, 000, 255)


def RGB2HSV(color):
    """
    Converts RGB to HS
    Parameters
    ----------
    color : tuple<uint8, uint8, uint8>
        RGB color value

    Returns
    -------
    tuple<int, float, float>
        HSV values. Hue in degrees, saturation and value on [0, 1]

    """
    hue, saturation, value = colorsys.rgb_to_hsv(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    hue *= 360
    return (int(hue), saturation, value)


def HSV2RGB(color):
    """
    Converts HSV to RGB

    Parameters
    ----------
    color : tuple<int, float, float>
        HSV values. Hue in degrees, saturation and value on [0, 1]

    Returns
    -------
    tuple<uint8, uint8, uint8>
        RGB color value
    """
    r, g, b = colorsys.hsv_to_rgb(color[0] / 360.0, color[1], color[2])
    return (int(r * 255), int(g * 255), int(b * 255))


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

    def get_cmapdict(self):
        return dict()


class VolumeRGB(DataviewRGB):
    """
    Contains RGB (or RGBA) colors for each voxel in a volumetric dataset.
    Includes information about the subject and transform for the data.

    Three data channels are mapped into a 3D color set. By default the data
    channels are mapped on to red, green, and blue. They can also be mapped to
    be different colors as specified, and then linearly combined.

    Each data channel is represented as a separate Volume object (these can
    either be supplied explicitly as Volume objects or implicitly as numpy
    arrays). The vmin for each Volume will be mapped to the minimum value for
    that data channel, and the vmax will be mapped to the maximum value.
    If `shared_range` is True, the vim and vmax will instead computed by
    combining all three data channels.

    Parameters
    ----------
    channel1 : ndarray or Volume
        Array or Volume for the first data channel for each
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume.
    channel2 : ndarray or Volume
        Array or Volume for the second data channel for each
        voxel. Can be a 1D or 3D array (see Volume for details), or a Volume.
    channel3 : ndarray or Volume
        Array or Volume for the third data channel for or each
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
        TODO: describe what this is
    channel1color : tuple<uint8, uint8, uint8>
        RGB color to use for the first data channel
    channel2color : tuple<uint8, uint8, uint8>
        RGB color to use for the second data channel
    channel3color : tuple<uint8, uint8, uint8>
        RGB color to use for the third data channel
    max_color_value : float [0, 1], optional
        Maximum HSV value for voxel colors. If not given, will be the value of
        the average of the three channel colors.
    max_color_saturation: float [0, 1]
        Maximum HSV saturation for voxel colors.
    shared_range : bool
        Use the same vmin and vmax for all three color channels?
    shared_vmin : float, optional
        Predetermined shared vmin. Does nothing if shared_range == False. If not given,
        will be the 1st percentile of all values across all three channels.
    shared_vmax : float, optional
        Predetermined shared vmax. Does nothing if shared_range == False. If not given,
        will be the 99th percentile of all values across all three channels
    **kwargs
        All additional arguments in kwargs are passed to the VolumeData and
        Dataview.

    """
    _cls = VolumeData

    def __init__(self, channel1, channel2, channel3, subject=None, xfmname=None, alpha=None, description="",
                 state=None, channel1color=Colors.Red, channel2color=Colors.Green, channel3color=Colors.Blue,
                 max_color_value=None, max_color_saturation=1.0, shared_range=False, shared_vmin=None,
                 shared_vmax=None, **kwargs):

        channel1color = tuple(channel1color)
        channel2color = tuple(channel2color)
        channel3color = tuple(channel3color)

        if isinstance(channel1, VolumeData):
            if not isinstance(channel2, VolumeData) or channel1.subject != channel2.subject:
                raise TypeError("Data channel 2 is not a VolumeData object or is from a different subject")
            if not isinstance(channel3, VolumeData) or channel1.subject != channel3.subject:
                raise TypeError("Data channel 2 is not a VolumeData object or is from a different subject")
            if (subject is not None) and (channel1.subject != subject):
                raise ValueError('Subject in VolumeData objects is different than specified subject')
            if (channel1color == Colors.Red) and (channel2color == Colors.Green) and (channel3color == Colors.Blue) \
                    and shared_range is False:
                # R/G/B basis can be directly passed through
                self.red = channel1
                self.green = channel2
                self.blue = channel3
                self.alpha = alpha
            else:  # need to remap colors
                red, green, blue, alpha = VolumeRGB.color_voxels(
                    channel1, channel2, channel3,
                    channel1color, channel2color, channel3color,
                    max_color_value, max_color_saturation,
                    shared_range, shared_vmin, shared_vmax, alpha=alpha
                )
                self.red = Volume(red, channel1.subject, channel1.xfmname)
                self.green = Volume(green, channel1.subject, channel1.xfmname)
                self.blue = Volume(blue, channel1.subject, channel1.xfmname)
                self.alpha = alpha
        else:
            if subject is None or xfmname is None:
                raise TypeError("Subject and xfmname are required")
            if (channel1color == Colors.Red) and (channel2color == Colors.Green) and (channel3color == Colors.Blue)\
                    and shared_range is False:
                # R/G/B basis can be directly passed through
                self.red = Volume(channel1, subject, xfmname)
                self.green = Volume(channel2, subject, xfmname)
                self.blue = Volume(channel3, subject, xfmname)
                self.alpha = alpha
            else:  # need to remap colors
                red, green, blue, alpha = VolumeRGB.color_voxels(
                    channel1, channel2, channel3,
                    channel1color, channel2color, channel3color,
                    max_color_value, max_color_saturation,
                    shared_range, shared_vmin, shared_vmax, alpha=alpha
                )
                self.red = Volume(red, subject, xfmname)
                self.green = Volume(green, subject, xfmname)
                self.blue = Volume(blue, subject, xfmname)
                self.alpha = alpha

        if self.red.xfmname == self.green.xfmname == self.blue.xfmname == self.alpha.xfmname:
            self.xfmname = self.red.xfmname
        else:
            raise ValueError('Cannot handle different transforms per volume')

        super(VolumeRGB, self).__init__(subject, alpha, description=description, state=state, **kwargs)

    @property
    def alpha(self):
        """Compute alpha transparency"""
        alpha = self._alpha
        if alpha is None:
            alpha = np.ones(self.red.volume.shape)
            alpha = Volume(alpha, self.red.subject, self.red.xfmname, vmin=0, vmax=1)
        if not isinstance(alpha, Volume):
            if alpha.dtype != np.uint8 and (alpha.min() < 0 or alpha.max() > 1):
                warnings.warn(
                    "Some alpha values are outside the range of [0, 1]. "
                    "Consider passing a Volume object as alpha with explicit vmin, vmax "
                    "keyword arguments.",
                    Warning
                )
            alpha = Volume(alpha, self.red.subject, self.red.xfmname, vmin=0, vmax=1)

        rgb = np.array([self.red.volume, self.green.volume, self.blue.volume])
        mask = np.isnan(rgb).any(axis=0)
        alpha.volume[mask] = alpha.vmin
        return alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

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
            if dv.volume.dtype != np.uint8:
                vol = dv.volume.astype("float32", copy=True)
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
            else:
                vol = dv.volume.copy()
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

    @property
    def raw(self):
        return self

    @staticmethod
    def color_voxels(channel1, channel2, channel3, channel1color, channel2color,
                     channel3Color, value_max, saturation_max, common_range,
                     common_min, common_max, alpha=None):
        """
        Colors voxels in 3 color dimensions but not necessarily canonical red, green, and blue
        Parameters
        ----------
        channel1 : ndarray or Volume
            voxel values for first channel
        channel2 : ndarray or Volume
            voxel values for second channel
        channel3 : ndarray or Volume
            voxel values for third channel
        channel1color : tuple<uint8, uint8, uint8>
            color in RGB for first channel
        channel2color : tuple<uint8, uint8, uint8>
            color in RGB for second channel
        channel3Color : tuple<uint8, uint8, uint8>
            color in RGB for third channel
        value_max : float, optional
            Maximum HSV value for voxel colors. If not given, will be the value of
            the average of the three channel colors.
        saturation_max : float [0, 1]
            Maximum HSV saturation for voxel colors.
        common_range : bool
            Use the same vmin and vmax for all three color channels?
        common_min : float, optional
            Predetermined shared vmin. Does nothing if shared_range == False. If not given,
            will be the 1st percentile of all values across all three channels.
        common_max : float, optional
            Predetermined shared vmax. Does nothing if shared_range == False. If not given,
            will be the 99th percentile of all values across all three channels
        alpha : ndarray or Volume, optional
            Alpha values for each voxel. If None, alpha is set to 1 for all voxels. 

        Returns
        -------
        red : ndarray of channel1.shape
            uint8 array of red values
        green : ndarray of channel1.shape
            uint8 array of green values
        blue : ndarray of channel1.shape
            uint8 array of blue values
        alpha : ndarray
            If alpha=None, uint8 array of alpha values with alpha=1 for every voxel. 
            Otherwise, the same alpha values that were passed in. Additionally, 
            voxels with NaNs will have an alpha value of 0.
        """
        # normalize each channel to [0, 1]
        data1 = channel1.data if isinstance(channel1, VolumeData) else channel1
        data1 = data1.astype(float)
        data2 = channel2.data if isinstance(channel2, VolumeData) else channel2
        data2 = data2.astype(float)
        data3 = channel3.data if isinstance(channel3, VolumeData) else channel3
        data3 = data3.astype(float)

        if (data1.shape != data2.shape) or (data2.shape != data3.shape):
            raise ValueError('Volumes are of different shapes')

        # Create an alpha mask now, before casting nans to 0
        # Voxels with at least one channel equal to NaN will be masked out.
        mask = np.isnan(np.array([data1, data2, data3])).any(axis=0)
        # Now convert to NaNs to num for all channels
        data1 = np.nan_to_num(data1)
        data2 = np.nan_to_num(data2)
        data3 = np.nan_to_num(data3)

        if common_range:
            if common_min is None:
                if common_max is None:
                    common_min = np.percentile(np.hstack((data1, data2, data3)), 1)
                else:
                    common_min = 0
            if common_max is None:
                common_max = np.percentile(np.hstack((data1, data2, data3)), 99)
            data1 -= common_min
            data2 -= common_min
            data3 -= common_min
            data1 /= (common_max - common_min)
            data2 /= (common_max - common_min)
            data3 /= (common_max - common_min)
        else:
            channelMin = np.percentile(data1, 1)
            channelMax = np.percentile(data1, 99)
            data1 -= channelMin
            data1 /= (channelMax - channelMin)
            channelMin = np.percentile(data2, 1)
            channelMax = np.percentile(data2, 99)
            data2 -= channelMin
            data2 /= (channelMax - channelMin)
            channelMin = np.percentile(data3, 1)
            channelMax = np.percentile(data3, 99)
            data3 -= channelMin
            data3 /= (channelMax - channelMin)
        data1 = np.clip(data1, 0, 1)
        data2 = np.clip(data2, 0, 1)
        data3 = np.clip(data3, 0, 1)

        channel1color = np.array(channel1color)
        channel2color = np.array(channel2color)
        channel3Color = np.array(channel3Color)

        averageColor = (channel1color + channel2color + channel3Color) / 3

        if value_max is None:
            _, _, value = RGB2HSV(averageColor)
            value_max = value

        red = np.zeros_like(data1, np.uint8)
        green = np.zeros_like(data1, np.uint8)
        blue = np.zeros_like(data1, np.uint8)
        for i in range(data1.size):
            this_color = data1.flat[i] * channel1color + data2.flat[i] * channel2color + data3.flat[i] * channel3Color
            this_color /= 3.0
            if (value_max != 1.0) or (saturation_max != 1.0):
                hue, saturation, value = RGB2HSV(this_color)
                saturation /= saturation_max
                value /= value_max
                if saturation > 1:
                    saturation = 1.0
                if value > 1:
                    value = 1.0
                this_color = HSV2RGB([hue, saturation, value])
            red.flat[i] = this_color[0]
            green.flat[i] = this_color[1]
            blue.flat[i] = this_color[2]

        # Now make an alpha volume
        if alpha is None:
            alpha = np.ones_like(red, np.uint8) * 255
        alpha[mask] = 0

        return red, green, blue, alpha


class VertexRGB(DataviewRGB):
    """
    Contains RGB (or RGBA) colors for each vertex in a surface dataset.
    Includes information about the subject.

    Each color channel is represented as a separate Vertex object (these can
    either be supplied explicitly as Vertex objects or implicitly as np
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
        TODO: describe what this is
    **kwargs
        All additional arguments in kwargs are passed to the VertexData and
        Dataview.

    """
    _cls = VertexData
    blend_curvature = _cls.blend_curvature  # hacky inheritance

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

        self.alpha = alpha

        super(VertexRGB, self).__init__(subject, alpha, description=description,
                                        state=state, **kwargs)

    @property
    def alpha(self):
        """Compute alpha transparency"""
        alpha = self._alpha
        if alpha is None:
            alpha = np.ones(self.red.vertices.shape[1])
            alpha = Vertex(alpha, self.red.subject, vmin=0, vmax=1)
        if not isinstance(alpha, Vertex):
            if alpha.dtype != np.uint8 and (alpha.min() < 0 or alpha.max() > 1):
                warnings.warn(
                    "Some alpha values are outside the range of [0, 1]. "
                    "Consider passing a Vertex object as alpha with explicit vmin, vmax "
                    "keyword arguments.",
                    Warning
                )
            alpha = Vertex(alpha, self.red.subject, vmin=0, vmax=1)

        rgb = np.array([self.red.data, self.green.data, self.blue.data])
        mask = np.isnan(rgb).any(axis=0)
        alpha.data[mask] = alpha.vmin
        return alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def vertices(self):
        """3-dimensional volume (t, v, rgba) with data that has been mapped
        into 8-bit unsigned integers that correspond to colors.
        """
        verts = []
        for dv in (self.red, self.green, self.blue, self.alpha):
            if dv.vertices.dtype != np.uint8:
                vert = dv.vertices.astype("float32", copy=True)
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
            else:
                vert = dv.vertices.copy()
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

    @property
    def raw(self):
        return self
