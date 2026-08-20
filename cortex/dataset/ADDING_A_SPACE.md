# Adding a new kind of brain data to `cortex.dataset`

How to add a `BrainSpace` and the three view classes that go with it. The
class graph these plug into is described in [INHERITANCE.md](INHERITANCE.md); this
document is only the procedure and the constraints on it.

The short version: **one space, three view classes that are mostly signature.** The
three abstract columns supply colormapping, `.raw`, `uniques`, `to_json`, HDF, NaN
handling, alpha and the arithmetic operators, so none of that is yours to write.

## The four declarations

The three bases supply everything else.

```python
@register_space
class MySpace(BrainSpace):
    hdf_key = "myspace"          # a label; nothing reads it -- see below
    @property
    def xfmname(self): ...                # the transform to sample through, or None
    def coerce(self, data): ...           # validate; record per-array geometry
    def is_movie(self, data): ...         # does it have a leading time axis
    @property
    def spatial_shape(self): ...
    def wrap(self, data, **kw): ...       # build a MyView over `data`
    def wrap_rgb(self, r, g, b, a, **kw): ...   # build a MyViewRGB
    def to_json(self): ...
    def write_hdf_attrs(self, h5, node): ...
    @classmethod
    def from_hdf(cls, attrs, *, subject, xfmname, mask): ...
    @classmethod
    def views(cls):
        return SpaceViews(scalar=MyView, twod=MyView2D, rgb=MyViewRGB)
        # `twod` and `rgb` are read when rebuilding a view from HDF; `scalar` is
        # not, since `wrap()` already builds one. Declared for completeness.

# written out in full below, under "The three view classes"
class MyView(ScalarView, MySpatial): ...     # + space-specific accessors
class MyView2D(Dataview2D[MyView], MySpatial): ...   # + a ctor naming its params
class MyViewRGB(DataviewRGB[MyView], MySpatial): ...
```

`spec_keys` is the one other declaration a space usually wants: the names of its
constructor arguments besides `subject` -- `("xfmname",)` for `VolumeSpace`, empty
for `SurfaceSpace`. `BrainSpace.from_spec` reads it to require each of them before
constructing, which is how the composite views build a space when they are handed
raw arrays rather than channel objects.

Six more members are concrete on `BrainSpace`. Inherit them unless the note says
otherwise:

- `view_xfmname`, derived as `None if self.xfmname is None else [self.xfmname]`, so
  implementing `xfmname` gets slot 7 right for free. Override it only if a space
  needs a slot-7 value that is not just its transform name.
- `template_shape`, the shape a *fresh* array should have, which is what
  `ScalarView.empty` and `random` read. It defaults to `spatial_shape`, which is
  correct whenever the geometry is known as soon as the space is built. Override it
  only if it is not: `VolumeSpace` does, because `spatial_shape` reports the
  geometry of an array already bound by `coerce` and is `()` until then, so a fresh
  volume space has to look its shape up from the transform. Getting this wrong is
  quiet -- `empty` would build a zero-dimensional array rather than fail.
- `describe_layout(data)`, the keys telling the browser how to unpack the array it
  is sent, merged into `to_json(simple=True)`. Empty by default. `VolumeSpace`
  returns `shape` (the grid the mosaic tiles unpack into) and `SurfaceSpace`
  returns `split` (the hemisphere boundary) and `frames`. Distinct from `to_json`,
  which describes the *space* rather than a particular array bound to it -- which
  is why only this one takes the data. These keys are read by `dataset.js`, so
  they are a hard interface.
- `pack_for_webgl(data, raw=...)`, the array encoded for the browser. Defaults to
  raising, since only two encodings exist and a space with no browser
  representation is legitimate; a space wanting one returns `MosaicTexture` or
  `VertexAttributes`. See "What a new spatial kind must implement to be rendered"
  below.
- `to_dense(data, movie)`, the array over the whole geometry with a leading frame
  axis -- what the scalar column returns as `renderer_data`. The default adds the
  frame axis and nothing else. Override it only if the space has a sparse form, as
  `VolumeSpace` does to unmask.
- `align(first, second)`, two views' arrays in a layout where position *i* means
  the same place in both -- what a 2D view needs before it can colormap its two
  dimensions jointly. The stored arrays serve any space in which one array position
  is one location, which is why this is concrete. `VolumeSpace` overrides it
  because a flattened array's positions mean something only relative to its own
  mask: two arrays under the *same* mask already line up and are far smaller, while
  under different masks -- or with one flattened and one not -- only the unmasked
  volumes are comparable. A space that cannot align two views at all raises.

`hdf_key` is a label rather than a mechanism, despite the name. Detection on load
walks `registered_spaces()` in registration order and takes the first space whose
`from_hdf` returns non-`None`; nothing reads `hdf_key`, and neither built-in writes
a discriminator, because legacy files predate the idea and carry none. It is worth
setting anyway as the one place a space names itself — a space that *does* want a
key on disk has an obvious value to write in `write_hdf_attrs` and match in
`from_hdf`.

### What goes on the space

Anything that depends on the geometry. If you find yourself wanting to branch on
which space you are in, from inside a column or a consumer, the answer is a method
on the space instead.

Concretely, the space owns: the shape of a frame (`spatial_shape`) and of a fresh
array (`template_shape`); validation and any per-array facts `coerce` records;
densification (`to_dense`); how two arrays are made comparable (`align`); the JSON
keys describing an array's layout (`describe_layout`) and the space itself
(`to_json`); the wire encoding (`pack_for_webgl`); the HDF attrs and their reader
(`write_hdf_attrs`, `from_hdf`); and the three view classes (`views`).

It does not own anything that relates *two* spaces. `Volume.map`, `Vertex.map` and
`Vertex.volume` transform between spaces through `cortex.utils.get_mapper`, so they
live on the views. A mapper for your space belongs beside those.

### The three view classes, written out

`MySpace` is where the thought goes. The three view classes are mostly signature:
across the six built-in views they are **141 lines of signature, 383 of docstring
and 228 of body**, and a single constructor is routinely thirty-five lines for two
statements. Copy the three skeletons below and fill them in.

Each member is marked **required** (the bases will not let you instantiate without
it), **narrowing** (the implementation is inherited; you are only restating the
concrete type so callers do not need a cast), or **optional**.

#### The scalar view

```python
class MyView(ScalarView, MySpatial):
    """One-line summary, then a numpydoc ``Parameters`` block.

    Not optional. `docs/api_reference_flat.rst` autosummaries this class through
    `_templates/class.rst`, which renders the class docstring verbatim -- so this
    is where users read what the constructor takes. Budget 25-35 lines.
    """

    def __init__(
        self,
        data: Union[npt.NDArray, str, None],
        subject: Union[str, bytes],
        myarg: str,                      # one per name in MySpace.spec_keys,
                                         # positional: `Volume(arr, "S1", "fullhead")`
                                         # is the spelling every doc and test uses
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state: Any = None,
        priority: int = 1,
        attrs: Optional[Mapping[str, Any]] = None,   # metadata; the only route
                                         # for a key that is not a parameter.
                                         # Do NOT add **kwargs -- see
                                         # INHERITANCE.md, "Unknown keywords"
    ) -> None:
        # The whole job: build the space, hand it up. ScalarView.__init__ calls
        # space.coerce(data), which is where per-array geometry gets recorded.
        super().__init__(
            data,
            MySpace(subject, myarg),
            cmap=cmap, vmin=vmin, vmax=vmax,
            description=description, state=state,
            priority=priority, attrs=attrs,
        )
        # Nothing else. `ScalarView.__init__` defaults vmin/vmax to the 1st/99th
        # data percentiles, so this view leaves construction with numeric bounds
        # like every other -- see INHERITANCE.md, "One rule for vmin/vmax".

    _space: MySpace                      # a *bare* annotation, which narrows the
                                         # attribute without shadowing the property

    @property                            # narrowing -- but load-bearing, since
    def space(self) -> MySpace:          # everything below reads space-specific
        return self._space               # members off it

    @property                            # narrowing; ScalarView._build_raw does it
    def raw(self) -> MyViewRGB:
        return cast(MyViewRGB, self._build_raw())

    # `empty`/`random` are the one place `**kwargs` remains, forwarding to the
    # constructor above -- so a bad keyword here is caught on the call rather than
    # by mypy. See INHERITANCE.md, "Unknown keywords".
    @classmethod                         # optional, but two lines each
    def empty(cls, subject: str, myarg: str, value: float = 0, **kwargs: Any) -> Self:
        shape = MySpace(subject, myarg).template_shape
        return cls(cls._sample(shape, value), subject, myarg, **kwargs)

    @classmethod
    def random(cls, subject: str, myarg: str, **kwargs: Any) -> Self:
        shape = MySpace(subject, myarg).template_shape
        return cls(cls._sample(shape, None), subject, myarg, **kwargs)

    def __repr__(self) -> str:           # optional
        return "<my data for (%s)>" % self.subject

    # Then whatever vocabulary the space exposes, one line each. These are what
    # `Vertex.llen`/`rlen`/`nverts`/`hem` and `Volume.linear`/`mask`/`mask_name`
    # are, and they are the reason `space` is narrowed above:
    #
    #     @property
    #     def nthings(self) -> int:
    #         return self.space.nthings
```

These three skeletons, filled in against a synthetic ten-element space, support
everything the built-in views do: construction from an array and from a movie,
`renderer_data`, the arithmetic operators, `empty`/`random`, `.raw`, 2D
construction from either arrays or views, RGB construction including movies,
`to_json` in both modes, `uniques()` collapsed and expanded, `name`,
`as_renderable`, and a full HDF round trip in which all three columns reload as
their own classes with the space rebuilt from what `write_hdf_attrs` wrote. That
filled-in version is `test_the_documented_skeleton_for_a_new_space_actually_works`
in `cortex/tests/test_new_space.py`; run it, then copy it.

Everything else on the scalar column is inherited and should not be reimplemented:
`data`, `movie`, `shape`, `name`, `copy`, `to_json`, `uniques`, `save`, the eight
arithmetic operators, `_write_data_hdf` and `_write_hdf` -- 29 members, 86
statements.

#### The 2D view

```python
class MyView2D(Dataview2D[MyView], MySpatial):
    """Summary plus a numpydoc ``Parameters`` block, as above. Budget 30-40 lines."""

    def __init__(
        self,
        dim1: Union[npt.NDArray, MyView],
        dim2: Union[npt.NDArray, MyView],
        subject: Optional[str] = None,   # optional, because it can come from dim1
        myarg: Optional[str] = None,     # likewise
        description: str = "",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vmin2: Optional[float] = None,
        vmax2: Optional[float] = None,
        alpha: Optional[npt.NDArray] = None,   # overrides the colormap's alpha
        state: Any = None,
        priority: int = 1,
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        chan1, chan2 = _resolve_2d_channels(
            dim1,
            dim2,
            channel_cls=MyView,
            space_cls=MySpace,
            subject=subject,
            spec={"myarg": myarg},       # the same keys as MySpace.spec_keys;
                                         # _resolve_channels uses them both to
                                         # validate channel objects and to build
                                         # the space when handed raw arrays
            ranges=((vmin, vmax), (vmin2, vmax2)),
        )
        super().__init__(
            chan1, chan2,
            description=description,
            cmap=cmap, vmin=vmin, vmax=vmax, vmin2=vmin2, vmax2=vmax2,
            alpha=alpha, state=state, priority=priority, attrs=attrs,
        )

    @property                            # narrowing; Dataview2D.raw does it, via
    def raw(self) -> MyViewRGB:          # space.align + space.wrap_rgb
        return cast(MyViewRGB, super().raw)

    def __repr__(self) -> str:           # optional
        return "<2D my data for (%s)>" % self.dim1.subject
```

That is the whole class. No `renderer_data`, no `volume`/`vertices`, no `to_json`,
no `_write_hdf` -- a 2D view owns no array of its own, so all of that is on
`Dataview2D`. Override `_write_hdf` only if slot 7 needs something other than
`space.view_xfmname`, as `Volume2D` does to record a transform name per dimension.

#### The RGB view

```python
class MyViewRGB(DataviewRGB[MyView], MySpatial):
    """Summary plus a numpydoc ``Parameters`` block. Budget 60-70 lines -- the
    three channel colours, the HSV caps and the per-channel bounds all need
    documenting."""

    def __init__(
        self,
        channel1: Union[npt.NDArray, MyView],
        channel2: Union[npt.NDArray, MyView],
        channel3: Union[npt.NDArray, MyView],
        subject: Optional[str] = None,
        myarg: Optional[str] = None,
        alpha: Optional[Union[npt.NDArray, MyView]] = None,
        description: str = "",
        state: Any = None,
        channel1color: Color[int] = Colors.Red,
        channel2color: Color[int] = Colors.Green,
        channel3color: Color[int] = Colors.Blue,
        max_color_value: Optional[float] = None,
        max_color_saturation: float = 1.0,
        vmin: Optional[Union[float, tuple[float, float, float]]] = None,
        vmax: Optional[Union[float, tuple[float, float, float]]] = None,
        autorange: Literal["shared", "individual"] = "individual",
        priority: int = 1,
        attrs: Optional[Mapping[str, Any]] = None,   # forward it, or a reloaded
                                                     # view loses its metadata
    ) -> None:
        red, green, blue, resolved_alpha = _resolve_rgb_channels(
            (channel1, channel2, channel3),
            channel_cls=MyView,
            space_cls=MySpace,
            subject=subject,
            spec={"myarg": myarg},
            colors=(channel1color, channel2color, channel3color),
            max_color_value=max_color_value,
            max_color_saturation=max_color_saturation,
            vmin=vmin, vmax=vmax, autorange=autorange,
            alpha=alpha,
        )
        super().__init__(
            red, green, blue,
            alpha=resolved_alpha,
            subject=subject,
            description=description, state=state, priority=priority,
            attrs=attrs,
        )

    @property                            # narrowing, if any space-specific member
    def space(self) -> MySpace:          # is read below. Sound because
        return self.red.space            # DataviewRGB[MyView] fixes the channel
                                         # type; the generic base cannot say so

    def __repr__(self) -> str:           # optional
        return "<RGB my data for (%s)>" % self.subject
```

All three columns take `attrs` as a named parameter and none ends in `**kwargs`,
so an unknown keyword to any of them is a plain `TypeError` from Python -- which is
also what lets mypy flag it. A space's three view classes each have to declare and
forward `attrs`, or metadata is dropped when the view is rebuilt from HDF; the
skeleton test catches that omission.

Also the whole class. `name`, `__hash__`, `renderer_data`, `to_json`, `_write_hdf`,
`alpha` and its NaN masking, `_default_alpha`, `_channel_stack`, `_rgba_stack`,
`uniques`, `copy` and `color_voxels` are all on `DataviewRGB` -- 23 members, 155
statements, and eleven of those members were per-subclass until recently.

### The channel resolvers

Both composite columns accept either already-built channel views or raw arrays, and
must reject the mixture. Do not re-implement that: call `_resolve_2d_channels` from
your 2D constructor and `_resolve_rgb_channels` from your RGB one, passing
`channel_cls`, `space_cls`, `subject`, and a `spec` dict whose keys are your
`spec_keys`. They validate, build the space when handed arrays, and return the
channels.

```python
space, views = _resolve_channels(
    [dim1, dim2], channel_cls=MyView, space_cls=MySpace,
    subject=subject, spec={"myarg": myarg}, argnames=("dim1", "dim2"),
)
```

`argnames` only shapes the error messages, so that they name the argument your
signature uses.

One asymmetry to expect: the RGB resolver requires `ndarray` channels, while the 2D
one calls `np.asarray` on whatever it is given. Pass arrays to both and it will not
matter.


## Getting it rendered

Two renderers, asking for different amounts. `quickflat` needs only what you have
already written. `webgl` needs a wire encoding, and if neither built-in encoding
fits, changes in JavaScript as well.

### `quickflat` — nothing beyond the space

Implement `renderer_data` on your views and give the space an `xfmname` (`None` if
the data is not sampled through a transform), and `quickshow`,
`make_flatmap_image` and `make_png` work. The renderer asks the space what to
sample through and the view for the array; it never asks what kind it is holding.

Three flatmap *decorations* require a transform and will refuse a space that has
none, with a `TypeError` naming the class: `with_dropout`,
`with_connected_vertices` and `add_connected_vertices`. There is nothing a
transformless space could do with them.

### `webgl` — pick an encoding, or write one

`Package` reads `renderer_data` and hands it to
`space.pack_for_webgl(data, raw=...)`, which returns a `WebGLPayload`. `raw` is
true when the array is 4-channel uint8 from an RGB view.

Two encodings exist, because `dataset.js` reads two:

| | `MosaicTexture` | `VertexAttributes` |
| --- | --- | --- |
| use when | values sit on a 3-D grid sampled through a transform | values sit on the surface's vertices |
| array shape | `(frames, z, y, x[, 4])` | `(frames, nverts[, 4])` |
| served as | one PNG per frame, tiled | one `.npy` array |
| JSON keys | `raw`, `mosaic` | `raw` |
| vertex order | n/a | permuted into the CTM's order by `reorder` |
| premultiplied alpha | no | **yes** |

Return one of them and you are done — nothing downstream changes:

```python
def pack_for_webgl(self, data, *, raw):
    from ._webgl import VertexAttributes
    return VertexAttributes(data, raw=raw)
```

`BrainSpace.pack_for_webgl` raises by default, so a space with no browser
representation simply does not override it, and `quickflat` still works.

The premultiplied-alpha asymmetry in that table is not a choice: Three.js
premultiplies a texture on upload, and does nothing to a vertex attribute, so the
per-vertex encoding has to do it in Python and the texture one must not.

#### If neither encoding fits

You are now changing the viewer, and these are the places. Nothing in
`cortex/dataset` or `webgl/data.py` needs to change.

**1. Serialization — `resources/js/dataset.js`.**

- `module.fromJSON` chooses the payload class by testing
  `dataset.data[name].mosaic === undefined`. It is a two-way branch; add yours,
  keyed on a JSON key your `describe()` emits.
- Write a payload class beside `module.VolumeData` and `module.VertexData`. Follow
  whichever is closer: `VolumeData` loads PNGs into textures and exposes
  `init(uniforms, dim, xfm, filter)` and `setFilter`; `VertexData` loads a `.npy`
  through `NParray.fromURL`, splits it at `json.split` into per-hemisphere
  `Float32Array`s, and re-maps through the CTM's index shuffle.
- `module.DataView` sets `this.vertex = this.data[0].mosaic === undefined`. This is
  the *same* test as in `fromJSON`, made a second time, and it selects the shader.
  A third kind needs it generalized, not just extended.
- The keys your payload's `describe()` and your space's `describe_layout()` emit
  are read here. Emit a key only when it applies: the JS tests for
  `undefined`, so a null is not the same as absent.

**2. Shaders — `resources/js/shaderlib.js`.**

- `Shaders.surface_vertex` and `Shaders.surface_pixel` are the two sampling paths;
  `mriview_surface.js` picks between them on `dataview.vertex`. A third sampling
  scheme means a third function and a third branch there.
- `DataView.getShader` passes flags your shader can read: `rgb` (from
  `data[0].raw`), `twod` (two channels), `sampler`, `voxline`. Add a flag rather
  than a new code path if the difference is small.
- Per-vertex attributes are declared as `attributes['data'+i]`, typed `v4` when
  `rgb` and `f` otherwise. Geometry that is not per-vertex — point sprites,
  instanced spheres — needs its own attribute set and its own draw call, which is
  the largest piece of work here.

**3. UI — `resources/js/dataset.js` and `mriview.js`.**

- `DataView.ui` is a `jsplot.Menu` populated with the colormap, `vmin` and `vmax`
  controls, and only when `!this.data[0].raw` — an RGB view carries its own colours
  and gets none. If your kind needs its own controls (electrode size, say), add
  them here.
- `mriview.js` `addData`/`setData` and the `#datasets` list are what present the
  loaded views; a kind that is selectable the same way needs nothing there.

### Consumers that will refuse your space

These name a concrete class because they need a specific capability, so they reject
a new space rather than mis-draw it: `volume.show_slice` needs a `Volume`,
`blender.add_cutdata` needs a single colormap, and `Mapper.__call__` needs a
`Vertex`.

## A worked example

`cortex/tests/test_new_space.py` builds a complete space and its three views
against a synthetic ten-element geometry, and exercises construction, movies,
`.raw`, `to_json`, HDF round-tripping, rendering through `quickflat`, and both
outcomes of `pack_for_webgl`. Copy from there rather than from `VolumeSpace` or
`SurfaceSpace`, which carry volume- and surface-specific detail you do not need.
