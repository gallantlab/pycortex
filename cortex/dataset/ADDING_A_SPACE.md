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
constructor arguments besides `subject`, `("xfmname",)` for `VolumeSpace` and empty
for `SurfaceSpace`. `BrainSpace.from_spec` reads it to require each of them before
constructing, which is how the composite views build a space when they are handed
raw arrays rather than channel objects. Each of the four used to be passed a
`lambda` naming the space class with `_require` applied to each of its arguments,
*and* a dict of those same keys to validate channel objects against -- so "a
volumetric space is subject plus xfmname, and both are mandatory" was written into
four view constructors instead of into `VolumeSpace`.

Five more members are deliberately *not* in the list above, because all five are
concrete on `BrainSpace` and most spaces should inherit them:

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

### What the space owns, and why

The rule is that anything depending on the *geometry* belongs to the space, so
that adding a space does not mean editing the columns. Seven things moved there,
each because it was the same question asked once per space:

| was | now | it was duplicated as |
| --- | --- | --- |
| `db.get_xfm(...).shape` vs `SurfaceSpace(...).nverts` | `template_shape` | 4 methods (`empty`/`random` x 2 classes) |
| `shape` vs `split`/`frames` in the simple JSON | `describe_layout` | 4 `to_json` overrides, now one each on `ScalarView` and `DataviewRGB` |
| slicing at `llen`, branching on movie-ness | `SurfaceSpace.split_hemispheres` | 4 properties (`left`/`right` on `Vertex` and `VertexRGB`) |
| whether two masked arrays can be compared elementwise | `align` | `Volume2D.raw` at 15 lines against `Vertex2D.raw` at 2 |
| "subject and xfmname are both mandatory" | `spec_keys` + `from_spec` | a lambda plus a dict in each of 4 constructors |
| the transform name a volumetric view reports | `xfmname`, now concrete on `VolumetricView` | 3 overrides, two of which reached through a channel |
| how an array reaches the browser | `pack_for_webgl` | 3 `isinstance(brain, SurfaceView)` forks in `webgl/data.py`, plus a guard for "neither" |

`split_hemispheres` also removed the one place a view reached *through a channel*
for geometry: `VertexRGB.left` read `self.red.llen` because `DataviewRGB.space` is
typed `BrainSpace`. `VertexRGB` now narrows `space` to `SurfaceSpace` the way
`Vertex` does -- sound because `DataviewRGB[Vertex]` fixes the channel type, which
the generic base cannot state.

What deliberately did **not** move:

- `Volume.map`, `Vertex.map`, `Vertex.volume` -- these transform *between* spaces
  via `cortex.utils.get_mapper`, so they are a property of a space *pair*, and
  putting them on `BrainSpace` would drag the mapper into `_space.py`.
- `Volume.save_nii` -- the affine it needs is a space fact, but writing a file is
  not; only the lookup is a candidate, and it is one line.
- `__repr__` -- the mask description is space knowledge, but the payoff is
  cosmetic.

`MySpatial` is the spatial interface — `VolumetricView` if the data samples through a
transform, `SurfaceView` if it is per-vertex, or a new subclass of
`RenderableView` supplying `renderer_data` if it is sampled some other way. A view
that inherits neither is still a perfectly good `Dataview`; it just cannot be
passed to the flatmap renderers, and `as_renderable` will say so.

`register_space` puts it in the registry that `_from_hdf_data`, `_from_hdf_view`
and `normalize` dispatch through, so HDF round-tripping needs no edits. Order
matters: the registry is consulted in order and the first space whose `from_hdf`
returns non-`None` wins.

Order is by **`fallback`**, not by arrival. A fallback space claims any node no
other space wanted, and exists only because legacy files carry no space
discriminator: `SurfaceSpace` sets it, and accepts anything without a transform,
which is how a pre-registry file is recognised. `register_space` inserts a
non-fallback space ahead of every fallback one, so the catch-alls stay last
however many spaces are added. Leave `fallback` alone; test in `from_hdf` for
something you write yourself in `write_hdf_attrs`, and you will be consulted
ahead of the built-in catch-all.

Pinned by `test_a_third_space_registers_ahead_of_the_catch_all`, which also checks
that a *second* fallback still sorts behind every real space.

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

    @property                            # REQUIRED
    def renderer_data(self) -> npt.NDArray:
        """The array a renderer samples, with a leading frame axis.

        Published as `volume` or `vertices` too, if MySpatial is one of the two
        built-in spatial interfaces. Do *not* also implement those.
        """

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

Filled in against a synthetic 10-element space, all three of these work: scalar
construction from an array and from a movie, `renderer_data`, the arithmetic
operators, `empty`/`random`, `raw` (which round-trips through `space.wrap_rgb`),
2D construction from both arrays and views, RGB construction including a movie,
`to_json` in both modes, `uniques()` collapsed and expanded, `name`,
`as_renderable`, and a full HDF round trip -- all three columns save and reload as
their own classes, with the space rebuilt from the discriminator its
`write_hdf_attrs` wrote. Pinned by
`test_the_documented_skeleton_for_a_new_space_actually_works`, which is this
skeleton filled in and exercised, so the doc cannot rot into describing an
extension point that no longer exists. It lives in `cortex/tests/test_new_space.py`
along with every other test that defines a space or a view this package does not
ship -- the ones cited by name elsewhere in this document included.

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

#### Why none of this is generated

The obvious next step is to generate the three classes from the space and let a new
one subclass only for extras. It was costed and rejected; the numbers are why.

Taking the surface family as one space's cost, of roughly 380 mechanical lines:

| | lines | generatable? |
| --- | --- | --- |
| the three `__init__` signatures | 90 | **no** |
| the three class docstrings | 123 | **no** |
| the space's own vocabulary accessors | 17 | no -- it is new by definition |
| genuinely new logic (`renderer_data`, cross-space maps, `__getitem__`) | 101 | no |
| narrowing properties (`space`, `raw` x 3) | 20 | **no** |
| `empty`/`random` and three `__repr__`s | 24 | yes |

- **`__init__`** cannot be generated because the space-identifying argument is
  passed *positionally* -- `cortex.Volume(arr, "S1", "fullhead")` -- so a generic
  `__init__(self, data, subject, **spec)` cannot put `myarg` in position 3. That is
  an API break, not a refactor. A `**kwargs` constructor also gives up parameter
  types and positional arity: mypy rejects `Vertex2D(a, a, vmin="lo")`,
  `VertexRGB(a, a, a, 3)`, a fourth positional to `Vertex`, and -- since no
  constructor ends in `**kwargs: Any` any more -- every misspelled keyword. A
  `**kwargs` constructor would give all of that back up, which is now the strongest
  argument against generating these. See INHERITANCE.md, "Unknown keywords".
- **Class docstrings** are rendered verbatim by `autoclass`, and they *are* the
  parameter documentation. Generating them means templating numpydoc, which is
  worse than writing it.
- **Narrowing properties** exist precisely to state a static type. A generated
  property has no narrowed type, so nothing is saved.
- **`empty`/`random`** genuinely could be `def random(cls, *args, **kwargs)`
  forwarding to `cls._space_cls(*args)`. But `Volume.empty("S1", "fullhead", 5)`
  passes `value` positionally today, and `*args` would swallow the `5` into the
  space constructor as `mask=5` -- silent breakage. Making `value` keyword-only
  fixes it and is an API change; and `*args: Any` means `Volume.empty("S1")`
  type-checks and fails at runtime.

So the ceiling is about 6% of the mechanical lines, half of it behind an API
change, in exchange for an `__init_subclass__` mechanism. If the per-space cost is
still the problem, the lever is the ~101 lines of genuinely new logic, and the only
help available there is this document.

### One channel resolver for both composite columns

`Dataview2D` and `DataviewRGB` both accept either already-built channel views or raw
arrays, and both have to reject the mixture. That validation -- is the first
argument a channel object; are the rest; do they agree on subject; do they agree on
the space's `spec_keys` -- was written out twice, in the same order, with different
wording. It is now `views._resolve_channels`, which returns the space plus the
channels if they arrived as views:

```python
space, views = _resolve_channels(
    [dim1, dim2], channel_cls=Volume, space_cls=VolumeSpace,
    subject=subject, spec={"xfmname": xfmname}, argnames=("dim1", "dim2"),
)
```

What is left in each column is only what genuinely differs: per-dimension
`vmin`/`vmax` for 2D, the colour basis and `color_voxels` for RGB. `argnames` exists
solely so the messages can name the offending argument, since one column calls them
`dim1`/`dim2` and the other `channel1`..`channel3`.

One quirk is preserved rather than unified: the RGB column requires `ndarray`
channels while the 2D column `np.asarray`'s whatever it is given. Loosening RGB
would start accepting lists there while `Volume(list, ...)` still fails with an
`AttributeError` from `coerce`, so the inconsistency is left where it is visible.

`space.wrap()` is the abstraction that keeps the rest space-agnostic: the channel
resolvers in `view2D.py` and `viewRGB.py`, and all three HDF factories, build views
through it and never name `Volume` or `Vertex`. A space is per-view, not shared:
`coerce()` records facts that depend on the particular array bound to it (which
mask a flattened array matches, which hemisphere a half-length array covered), so
`wrap()` uses `self` only as a template of parameters.


## What a new spatial kind must implement to be rendered

The two renderers ask for different amounts, and the difference is not a matter of
tidiness: `renderer_data` is enough to *draw a flatmap*, but not enough to *ship
data to a browser*, because the browser needs to know how the bytes are laid out.
Both are now open to a new spatial kind, but the webgl one is open only to the
extent of picking between the two layouts `dataset.js` understands.

### `quickflat` — nothing beyond the spatial ABC

Implement `renderer_data` and give the space an `xfmname` (`None` if the data is
not sampled through a transform) and `quickshow`/`make_flatmap_image` work. The
renderer never asks what it is holding. Pinned by
`test_a_third_spatial_kind_needs_no_change_to_the_renderer`.

Three flatmap *decorations* legitimately require a transform and will reject a kind
that has none, with a `TypeError` naming the class: `with_dropout`,
`with_connected_vertices`, and `add_connected_vertices`. That is a capability
requirement, not a closed-world assumption — there is nothing a transformless kind
could do with them.

### `webgl` / `webshow` — `space.pack_for_webgl`, returning one of two encodings

`webgl/data.py`'s `Package` reads `renderer_data` like everything else, and then asks
the space to encode it: `space.pack_for_webgl(renderer_data, raw=...)` returns a
`WebGLPayload`, of which exactly two exist because `dataset.js` can read exactly
two. `raw` says the array is 4-channel uint8 from an RGB view rather than scalar
floats.

| | `MosaicTexture` | `VertexAttributes` |
| --- | --- | --- |
| returned by | `VolumeSpace` | `SurfaceSpace` |
| array shape | `(frames, z, y, x[, 4])` | `(frames, nverts[, 4])` |
| packing | `volume.mosaic()` per frame → PNG | raw `.npy` bytes |
| JSON (`describe()`) | `raw`, and `mosaic` set to the tile shape | `raw` only, no `mosaic` key |
| slot 7 | `[xfmname]` | `null` |
| vertex order | `reorder` is a no-op | **must** be permuted by `Package.reorder` into the CTM's order |
| JS path | texture, sampled through the transform | per-vertex attribute |
| premultiplied alpha | no — Three.js premultiplies on texture upload | **yes**, done in Python |

Both live in `_webgl.py`, next to each other, because the packing, the `mosaic`
key, the reordering and the premultiply are one decision. `Package` used to make it
three times over —
`isinstance(brain, SurfaceView)` for the premultiply, again for the packing, again
in `reorder` — plus a fourth `not isinstance(brain, VolumetricView)` guard for
"neither". So a new spatial kind had to inherit `VolumetricView` or `SurfaceView`
*for the webgl path to work*, even though `quickflat` accepts a bare
`RenderableView`, and that was the one place the spatial axis was genuinely not
open.

It is open now, in the sense that matters here: a space picks an encoding, so it
reaches the browser without any edit to `webgl/data.py`. Pinned by
`test_a_third_space_packs_its_own_webgl_encoding`. What is *not* open is the set of
encodings — that is a constraint of the browser code, not of this package.
`dataset.js` selects the path by testing `mosaic === undefined`, so a kind wanting a
third layout still has to add a matching branch there (and to `shaderlib.js`, if it
samples differently).

`BrainSpace.pack_for_webgl` is concrete rather than abstract, and its default
raises a `TypeError` naming the space, both encodings and the JS file. A space with
no browser representation is a legitimate thing to have — `quickflat` needs only
`renderer_data` — so this is a capability the space declines rather than one it
forgets. Pinned by `test_a_space_with_no_webgl_encoding_says_so_once`, which also
records what the old shape cost: a third kind fell through to `volume.mosaic` and
died on "Invalid data shape", several frames deep, naming neither the view nor the
real problem.

Between `pack_for_webgl` and `describe_layout` the whole wire contract is now
stated by the space. `mosaic` used to be the exception — assembled by the consumer
while its sibling keys `shape`, `split` and `frames` came from the space — and it
is `MosaicTexture.describe()` that closed that gap. Keys must be *absent* rather
than null when they do not apply, since the JS tests `mosaic === undefined`.

Treat `_webgl.py` as a compatibility surface with its own tests, not as a place to
tidy up: the premultiplied-alpha asymmetry above is a fact about Three.js
(`tex.premultiplyAlpha` on upload for the texture path, nothing at all for vertex
attributes), and it is pinned in both directions by
`test_vertexrgb_alpha_is_premultiplied_in_package` and
`test_volumergb_alpha_is_NOT_premultiplied_in_package`.

The geometry is always a surface mesh, whatever the spatial kind: `brainctm.py`
builds it from `db.get_surf`, so volumetric data is rendered by sampling its texture
at each vertex's coordinates. A kind whose data cannot be evaluated per-vertex has no webgl
representation at all.

`Package` iterates `uniques(collapse=True)`, which decomposes 2D views into their
scalar channels, so only the scalar and RGB columns ever reach it. That is why
nothing there mentions `Dataview2D`.

### Consumers that are deliberately narrower

These name a concrete class because they need a specific capability, and a new kind
will be rejected rather than mis-drawn:

| consumer | requires | why |
| --- | --- | --- |
| `volume.show_slice` | `Volume` | slices the 3D array against an anatomical reference |
| `blender.add_cutdata` | `Volume` or `Vertex` | samples one colormapped value per vertex, so needs a single `cmap` |
| `Mapper.__call__` | `Vertex` | splits per-hemisphere by vertex count |
