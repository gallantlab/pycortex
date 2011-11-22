import os
import tempfile
import multiprocessing as mp
import numpy as np
from scipy.interpolate import interp1d, Rbf

try:
    from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, Any, Color,Enum, Callable, on_trait_change
    from traitsui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor

    from tvtk.api import tvtk
    from tvtk.pyface.scene import Scene

    from mayavi import mlab
    from mayavi.core.ui import lut_manager
    from mayavi.core.api import PipelineBase, Source, Filter, Module
    from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
except ImportError:
    from enthought.traits.api import HasTraits, Instance, Array, Bool, Dict, Any, Range, Color,Enum, Callable, on_trait_change
    from enthought.traits.ui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor, Handler

    from enthought.tvtk.api import tvtk
    from enthought.tvtk.pyface.scene import Scene

    from enthought.mayavi import mlab
    from enthought.mayavi.core.ui import lut_manager
    from enthought.mayavi.core.api import PipelineBase, Source, Filter, Module
    from enthought.mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene

import db

svg_format = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="{width}"
   height="{height}"
   id="svg2"
   version="1.1"
   inkscape:version="0.48.1 r9760"
   sodipodi:docname="drawing.svg">
  <defs
     id="defs4" />
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.35"
     inkscape:cx="932.14286"
     inkscape:cy="448.57143"
     inkscape:document-units="px"
     inkscape:current-layer="layer3"
     showgrid="false"
     inkscape:window-width="1918"
     inkscape:window-height="1163"
     inkscape:window-x="0"
     inkscape:window-y="35"
     inkscape:window-maximized="0" />
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="data"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(0,0)"
     style="display:inline">
    <g
       inkscape:groupmode="layer"
       id="layer4"
       inkscape:label="_data"
       style="display:inline"
       sodipodi:insensitive="true">
      <image
         y="0"
         x="0"
         height="{height}"
         width="{width}"
         id="image3115"
         xlink:href="data:image/png;base64,{pngdata}"/>
    </g>
  </g>
  <g
     inkscape:groupmode="layer"
     id="layer3"
     inkscape:label="rois"
     transform="translate(0,0)"
     style="display:inline" />
</svg>"""

clear_white_black = np.tile(0, [85, 4]), np.tile(255, [86, 4]), np.tile(0, [85, 4]),
clear_white_black[-1][:,-1] = 255
clear_white_black = np.vstack(clear_white_black)

class Mixer(HasTraits):
    points = Any
    polys = Array(shape=(None, 3))
    data = Array

    mix = Range(0., 1., value=1)
    figure = Instance(MlabSceneModel, ())
    data_src = Instance(Source)
    surf = Instance(Module)

    colormap = Enum(*lut_manager.lut_mode_list())
    fliplut = Bool

    def _data_src_default(self):
        pts = self.points(1)
        src = mlab.pipeline.triangular_mesh_source(
            pts[:,0], pts[:,1], pts[:,2],
            self.polys, figure=self.figure.mayavi_scene)
        #Set the texture coordinates
        pts -= pts.min(0)
        pts /= pts.max(0)
        src.data.point_data.t_coords = pts[:,[0,2]]

        if self.data is not None:
            src.mlab_source.scalars = self.data
        return src

    def _surf_default(self):
        n = mlab.pipeline.poly_data_normals(self.data_src, figure=self.figure.mayavi_scene)
        surf = mlab.pipeline.surface(n, figure=self.figure.mayavi_scene)
        #surf.actor.enable_texture = True
        #surf.actor.texture_source_object = self.tex
        surf.actor.texture.interpolate = True
        surf.actor.texture.repeat = False
        surf.actor.texture.lookup_table = tvtk.LookupTable(
            table=clear_white_black, range=(-1,1))
        return surf

    @on_trait_change("figure.activated")
    def _start(self):
        self.figure.scene.background = (0,0,0)
        self.figure.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        self.figure.scene.render_window.stereo_type = "anaglyph"
        self.figure.camera.view_up = [0,0,1]
        self.reset_view()

        self.data_src
        self.surf
        self.colormap = "RdBu"
        self.fliplut = True
    
    #@on_trait_change('mix')
    def _mix_changed(self):
        self.data_src.data.points.from_array(self.points(self.mix))
        self.figure.renderer.reset_camera_clipping_range()
        self.figure.render()
        #def func():
        #    self.data_src.data.points = self.points(self.mix)
        #    GUI.invoke_later(self.data_src.data.update)
        #threading.Thread(target=func).start()
    
    @on_trait_change("colormap, fliplut")
    def _update_colors(self):
        self.surf.parent.scalar_lut_manager.lut_mode = self.colormap
        self.surf.parent.scalar_lut_manager.reverse_lut = self.fliplut
    
    def reset_view(self, center=True):
        '''Sets the view so that the flatmap is centered'''
        #set up the flatmap view
        self.mix = 1
        ptmax = self.data_src.data.points.to_array().max(0)
        ptmin = self.data_src.data.points.to_array().min(0)
        size = ptmax-ptmin
        focus = size / 2 + ptmin
        if center:
            focus[[0,2]] = 0
        
        x, y = self.figure.get_size()
        h = y / float(x) * size[0] / 2
        h /= np.tan(np.radians(self.figure.camera.view_angle / 2))
        campos = focus - [0, h, 0]
        self.figure.camera.position, self.figure.camera.focal_point = campos, focus
        self.figure.renderer.reset_camera_clipping_range()
        self.figure.render()
    
    def saveflat(self, filename, height=1024):
        #Save the current view to restore
        startmix = self.mix
        lastpos = self.figure.camera.position, self.figure.camera.focal_point

        #Turn on offscreen rendering
        mlab.options.offscreen = True
        x, y = self.figure.get_size()
        ptmax = self.data_src.data.points.to_array().max(0)
        ptmin = self.data_src.data.points.to_array().min(0)
        size = ptmax-ptmin
        aspect = size[0] / size[-1]
        width = height * aspect
        self.figure.set_size((width, height))
        self.figure.interactor.update_size(int(width), int(height))
        self.figure.off_screen_rendering = True
        self.reset_view(center=center)

        if os.path.splitext(filename)[-1].lower() == ".svg":
            
        else:
            self.figure.save_png(filename)
        

        #Restore the last view, turn off offscreen rendering
        self.figure.interactor.update_size(x, y)
        self.figure.set_size((x,y))
        self.mix = startmix
        self.figure.camera.position, self.figure.camera.focal_point = lastpos
        self.figure.renderer.reset_camera_clipping_range()
        self.figure.off_screen_rendering = False
        self.figure.render()

    view = View(
        HGroup(
            Group(
                Item("figure", editor=SceneEditor(scene_class=MayaviScene)),
                "mix",
                show_labels=False),
            Group(
                Item('colormap',
                     editor=ImageEnumEditor(values=lut_manager.lut_mode_list(),
                     cols=6, path=lut_manager.lut_image_dir)),
                "fliplut"),
        show_labels=False),
        resizable=True, title="Mixer")

def _get_surf_interp(subject, types=('inflated',), hemisphere="both"):
    types = ("fiducial",) + types + ("flat",)
    pts = []
    for t in types:
        pt, polys, norm = db.surfs.getVTK(subject, t, hemisphere=hemisphere)
        pts.append(pt)

    #flip the flats to be on the X-Z plane
    flatpts = np.zeros_like(pts[-1])
    flatpts[:,[0,2]] = pts[-1][:,:2]
    flatpts[:,1] = pts[-2].min(0)[1]
    pts[-1] = flatpts

    interp = interp1d(np.linspace(0,1,len(pts)), pts, axis=0)
    return interp, polys

def show(data, subject, xfm, types=('inflated',), hemisphere="both"):
    '''View epi data, transformed into the space given by xfm. 
    Types indicates which surfaces to add to the interpolater. Always includes fiducial and flat'''
    interp, polys = _get_surf_interp(subject, types, hemisphere)

    if hasattr(data, "get_affine"):
        #this is a nibabel file -- it has the nifti headers intact!
        if isinstance(xfm, str):
            xfm = db.surfs.getXfm(subject, xfm, xfmtype="magnet")
            assert xfm is not None, "Cannot find transform by this name!"
            xfm = np.dot(np.linalg.inv(data.get_affine()), xfm[0])
        data = data.get_data()
    else:
        xfm = db.surfs.getXfm(subject, xfm, xfmtype="coord")
        assert xfm is not None, "Cannot find coord transform, please provide a nifti!"
        xfm = xfm[0]
    assert xfm.shape == (4, 4), "Not a transform matrix!"

    pts = interp(0)
    wpts = np.append(pts, np.ones((len(pts),1)), axis=-1).T
    coords = np.dot(xfm, wpts)[:3].T.round().astype(int)
    scalars = np.array([data.T[tuple(p)] for p in coords])
    
    m = Mixer(points=interp, polys=polys, data=scalars)
    m.edit_traits()
    return m

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Display epi data on various surfaces, \
        allowing you to interpolate between the surfaces")
    parser.add_argument("epi", type=str)
    parser.add_argument("--transform", "-T", type=str)
    parser.add_argument("--surfaces", nargs="*")