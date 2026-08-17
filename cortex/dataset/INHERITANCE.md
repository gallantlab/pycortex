# `cortex.dataset` class hierarchy

Reference for the `Dataview` object graph. See
[TYPING_ALTERNATIVES.md](TYPING_ALTERNATIVES.md) for the restructuring options that
were considered and why this one was chosen.

## Two axes, two mechanisms

The six public classes are a 2x3 grid: two **spaces** crossed with three **channel
layouts**. The two axes are deliberately given different mechanisms.

|  | scalar (1 channel + 1D cmap) | 2D (2 channels + 2D cmap) | RGB (3 channels + alpha) |
| --- | --- | --- | --- |
| **volumetric** | `Volume` | `Volume2D` | `VolumeRGB` |
| **surface** | `Vertex` | `Vertex2D` | `VertexRGB` |
| *abstract base* | `ScalarView` | `Dataview2D` | `DataviewRGB` |

- **Channel layout is the inheritance axis.** All the shared logic -- colormapping,
  HDF, JSON, NaN handling, alpha -- lives in the three abstract bases.
- **Space is a component**, `view.space`, described by a `BrainSpace`. This is the
  open axis: adding a new kind of brain data means adding a space, not
  reimplementing the three bases.

There is no multiple inheritance anywhere. `BrainData` and `Dataview` used to be
unrelated base classes joined only by MI in `Volume` and `Vertex`, and that MI was
load-bearing: `BrainData.to_json` and `VolumeData.copy` called `super()` methods
that existed nowhere in their own ancestry, resolving only because `Volume`'s MRO
threaded through `Dataview`.

```mermaid
classDiagram
    direction TB

    class Dataview {
        <<abstract>>
        +space: BrainSpace
        +subject
        +state
        +attrs
        +description
        +priority
        +raw()*
        +uniques()*
        +to_json()
        +get_cmapdict()
        +_write_view_node()
        +_write_hdf()*
        +from_hdf()$
    }
    class ScalarView {
        <<abstract>>
        +data
        +movie
        +shape
        +cmap
        +vmin
        +vmax
        +name
        +copy()*
        +exp()
        +_colormap_to_rgba()
        +_write_data_hdf()
        +__add__ __sub__ __neg__ ...
    }
    class Dataview2D~ScalarT~ {
        <<abstract>>
        +dim1: ScalarT
        +dim2: ScalarT
        +cmap
        +vmin
        +vmax
        +vmin2
        +vmax2
        +copy()
        +_to_raw()
    }
    class DataviewRGB~ScalarT~ {
        <<abstract>>
        +red: ScalarT
        +green: ScalarT
        +blue: ScalarT
        +alpha: ScalarT
        +copy()
        +color_voxels()$
        +_nan_mask
    }

    class Volume {
        +xfmname
        +linear
        +mask
        +mask_name
        +volume
        +masked
        +map()
        +save_nii()
        +empty()$
        +random()$
    }
    class Vertex {
        +llen
        +rlen
        +nverts
        +hem
        +vertices
        +left
        +right
        +volume()
        +map()
        +blend_curvature()
        +empty()$
        +random()$
    }
    class Volume2D
    class Vertex2D
    class VolumeRGB
    class VertexRGB
    class Multiview {
        <<unimplemented>>
    }

    Dataview <|-- ScalarView
    Dataview <|-- Dataview2D
    Dataview <|-- DataviewRGB
    Dataview <|-- Multiview

    ScalarView <|-- Volume
    ScalarView <|-- Vertex
    Dataview2D <|-- Volume2D
    Dataview2D <|-- Vertex2D
    DataviewRGB <|-- VolumeRGB
    DataviewRGB <|-- VertexRGB
```

```mermaid
classDiagram
    direction TB
    class BrainSpace {
        <<abstract>>
        +subject
        +hdf_key$
        +coerce()*
        +is_movie()*
        +spatial_shape*
        +wrap()*
        +to_json()*
        +write_hdf_attrs()*
        +view_xfmname*
        +from_hdf()$*
        +views()$*
    }
    class VolumeSpace {
        +xfmname
        +mask
        +mask_name
        +mask_spec
        +linear
        +unmask()
    }
    class SurfaceSpace {
        +llen
        +rlen
        +nverts
        +hem
    }
    BrainSpace <|-- VolumeSpace
    BrainSpace <|-- SurfaceSpace
```

Source locations:

| Class | Location |
| --- | --- |
| `Dataview`, `ScalarView`, `Volume`, `Vertex`, `Multiview`, `_masker` | `views.py` |
| `Dataview2D`, `Volume2D`, `Vertex2D` | `view2D.py` |
| `DataviewRGB`, `VolumeRGB`, `VertexRGB`, `Colors` | `viewRGB.py` |
| `BrainSpace`, `VolumeSpace`, `SurfaceSpace`, the space registry | `_space.py` |
| `Dataset` | `dataset.py` |
| `_hash`, `_hdf_write`, `_find_mask` | `_hdf.py` |
| `VolumetricView` / `SurfaceView` (row ABCs) | `views.py` |
| `Renderable`, `as_renderable`, `space_of` | `_typing.py` |

`braindata.py` is a compatibility shim: `BrainData`, `VolumeData` and `VertexData`
are aliases for `ScalarView`, `Volume` and `Vertex`. The aliases preserve
`isinstance` exactly -- `isinstance(x, VolumeData)` was only ever true for
`Volume`, since `Volume2D` and `VolumeRGB` never inherited it. Two module paths are
pinned by external code and still resolve: `cortex.dataset.views.Vertex`
(`cortex/database.py`) and `dataset.braindata.VertexData` (`cortex/blender/`).

## Generics: one covariant TypeVar

`Dataview2D` and `DataviewRGB` are generic in their channel type:

```python
ScalarT = TypeVar("ScalarT", bound=ScalarView, covariant=True)

class Volume2D(Dataview2D[Volume]): ...
class VolumeRGB(DataviewRGB[Volume]): ...
```

So `Volume2D.dim1` is a `Volume` and `VertexRGB.alpha` is a `Vertex` without either
class re-declaring anything. That last one is why the TypeVar earns its keep: **a
property's return type cannot be narrowed by re-annotation, only by re-implementing
the property.** `alpha` used to exist twice, ~25 near-identical lines in each RGB
class, purely because of that.

Covariance is sound here because the channels are read-only properties backed by
private fields, set once in `__init__`. It buys off the usual generics tax:
`Dataview2D[ScalarView]` accepts a `Volume2D`, which an invariant parameter would
reject.

Two limits worth knowing rather than discovering:

- mypy allows a covariant TypeVar in `__init__` parameters and in return position,
  but **not in ordinary method parameters**. The `alpha` setter therefore takes the
  base `ChannelLike`, not `ScalarT`.
- `isinstance(x, DataviewRGB[Volume])` is a runtime `TypeError`. Unsubscripted works
  and narrows to `DataviewRGB[Any]`, which is what the back-compat aliases rely on.

Narrowing a **bare annotation** in a subclass is accepted by mypy; narrowing an
**assigned** ClassVar is not. That asymmetry is why `space` is exposed as a
read-only property over a narrowed private `_space`, and it is exactly what made
the old `_cls` untypeable.

One thing the type system cannot express here: nothing stops
`DataviewRGB[Vertex]` from being declared with a `VolumeSpace`. Tying the space to
the channel would need higher-kinded types, so it stays a runtime constructor
check.

## Adding a new kind of brain data

Four declarations. The three bases supply everything else.

```python
@register_space
class MySpace(BrainSpace):
    hdf_key = "myspace"
    def coerce(self, data): ...           # validate; record per-array geometry
    def is_movie(self, data): ...         # does it have a leading time axis
    @property
    def spatial_shape(self): ...
    def wrap(self, data, **kw): ...       # build a MyView over `data`
    def to_json(self): ...
    def write_hdf_attrs(self, h5, node): ...
    @property
    def view_xfmname(self): ...           # slot 7 of the /views record
    @classmethod
    def from_hdf(cls, attrs, *, subject, xfmname, mask): ...
    @classmethod
    def views(cls):
        return SpaceViews(scalar=MyView, twod=MyView2D, rgb=MyViewRGB)

class MyView(ScalarView): ...             # + space-specific accessors
class MyView2D(Dataview2D[MyView]): ...   # + a constructor forwarding space kwargs
class MyViewRGB(DataviewRGB[MyView]): ...
```

`register_space` puts it in the registry that `_from_hdf_data`, `_from_hdf_view`
and `normalize` dispatch through, so HDF round-tripping needs no edits. Order
matters: the registry is consulted in registration order and the first space whose
`from_hdf` returns non-`None` wins. `SurfaceSpace` is deliberately last, because it
accepts anything without a transform -- that is how legacy files, which carry no
space discriminator, are detected. A new space should test for something it writes
itself in `write_hdf_attrs`, and will be consulted ahead of the built-ins.

`space.wrap()` is the abstraction that keeps the rest space-agnostic: the channel
resolvers in `view2D.py` and `viewRGB.py`, and all three HDF factories, build views
through it and never name `Volume` or `Vertex`. A space is per-view, not shared:
`coerce()` records facts that depend on the particular array bound to it (which
mask a flattened array matches, which hemisphere a half-length array covered), so
`wrap()` uses `self` only as a template of parameters.

## Narrowing the grid

Both axes are real classes, so every narrowing test is a nominal `isinstance`.

|  | volumetric | surface |
| --- | --- | --- |
| **row** (space) | `VolumetricView` | `SurfaceView` |
| **column** (channels) | `ScalarView` / `Dataview2D` / `DataviewRGB` | same |

The columns were always classes; only the rows lacked a type, which is why
consumers duck-typed `hasattr(braindata, "xfmname")`. `VolumetricView` declares
`xfmname` and `volume`; `SurfaceView` declares `vertices`. Each of the six views
inherits exactly one, which is what makes the volumetric/surface fork total.

Narrowing needs no machinery — `isinstance` against a union member subtracts it
from the `else` branch:

```python
def make_flatmap_image(braindata: Renderable, ...):
    if isinstance(braindata, SurfaceView):
        data = braindata.vertices          # SurfaceView
    else:
        data = braindata.volume            # VolumetricView
```

No `TypeGuard`, no `TypeIs`, no `Protocol`, no `typing_extensions`, no predicate
helpers. Earlier iterations had all of those; `TypeIs` is only needed to carry
that narrowing *across a function-call boundary*, i.e. only if the check lives in
a named helper rather than inline.

**Which mechanism for which question.** Protocols and ABCs are both used here, for
different questions, and the split is deliberate:

| question | mechanism | why |
| --- | --- | --- |
| "what kind of view is this?" | ABC — `VolumetricView`, `SurfaceView` | asked at *runtime*, so the check must be sound; only a nominal base gives that |
| "what does this function need?" | Protocol — `HasSubject`, `SupportsCurvatureBlend` | a *static* contract on a parameter; never `isinstance`d |

The Protocols are deliberately **not** `runtime_checkable`, so `isinstance` against
them raises `TypeError` — which mechanically prevents the presence-only check from
creeping back in. A test pins that.

The payoff for the Protocol half is that a function can claim exactly what it
touches. Every one of `add_curvature`, `add_rois`, `add_sulci`, `add_custom` and
`add_cutout` reads nothing but `dataview.subject`, yet they were annotated
`Dataview` (or a `Union[Vertex, Volume, Dataview]` that collapses to it). They now
take `HasSubject`.

**Why abstract bases rather than `Protocol` for the rows.** A `runtime_checkable`
protocol's
`isinstance` tests only for the *presence* of the member names — it cannot tell a
property from a method or check any types — so an object carrying an unrelated
`subject`/`xfmname`/`volume` satisfies it. Composing several protocols does not
help; the check is still per-name `hasattr`. Nominal bases give a real class
check, and because the row members are abstract, a view that forgets one cannot be
instantiated. The trade is that conformance is explicit opt-in: a third-party view
must inherit one of the rows rather than merely happening to have the attributes.
Both properties are pinned by tests.

The rows subclass `Dataview`, so narrowing to a row keeps every `Dataview` member
available. Do *not* convert a `Dataview` into some separate renderable type and
carry it around — an intermediate design did that with protocols and broke
`get_cmapdict`, `add_curvature`, `add_rois`, `add_sulci` and `add_custom`, because
protocols are structural where `Dataview` is nominal.

**`isinstance` tuples must be inline literals.** `isinstance(v, (ScalarView,
Dataview2D))` narrows; hoisting that tuple into a module constant annotated
`tuple[type, ...]` does not, because the annotation loses the members. The same
trap applies to any such constant.

This is not a return to the multiple inheritance the package was restructured to
remove. The rows are stateless interfaces: no `__init__`, no attributes, no
cooperative `super()` chain. `BrainData`/`Dataview` were pathological because each
carried state and called `super()` methods that resolved only through a subclass's
MRO. The MROs here linearize cleanly, e.g.
`Volume2D -> Dataview2D -> VolumetricView -> Dataview -> ABC -> Generic -> object`.

`Volume2D.volume` and `VolumeRGB.xfmname` exist to make the rows implementable.
`Volume2D` had no `volume` at all, so consumers special-cased it to reach
`.raw.volume`; `VolumeRGB.xfmname` was a stored copy of `red.xfmname`, which an
abstract property will not accept, so it is now derived. Like `VolumeRGB.volume`,
`Volume2D.volume` returns uint8 RGBA rather than the scalar array
`Volume.volume` returns — the same split `vertices` already had across `Vertex`
and `Vertex2D`.

## The wire format is a hard interface

`webgl/resources/js/dataset.js` dispatches **structurally**; no Python class name
reaches the browser.

| JS test | Meaning |
| --- | --- |
| `mosaic === undefined` | surface data |
| `json.data[i] instanceof Array` | 2D view |
| `json.raw` | RGB, 4-channel uint8 texture path |
| `json.vmin[0] instanceof Array` | 2D ranges |

`module.makeFrom(dvx, dvy)` also synthesizes a 2D view client-side. The eight-slot
`/views` record and the `to_json` key shapes must therefore be preserved exactly,
including the `"null"` written into slots 2-4 for RGB views. Any change here should
be checked by diffing the saved HDF and `to_json()` output before and after: a
shape change breaks the viewer silently rather than noisily.

Slot layout, written by `Dataview._write_view_node`:

| Slot | Contents |
| --- | --- |
| 0 | data node name(s); a nested list means a 2D or RGB view |
| 1 | description |
| 2 | `[cmap]`, or `"null"` for RGB |
| 3 | `[vmin]`, or `[[vmin, vmin2]]` for 2D, or `"null"` for RGB |
| 4 | `[vmax]`, likewise |
| 5 | state |
| 6 | attrs |
| 7 | `space.view_xfmname` -- `[xfmname]` for volumes, `null` for surfaces |

An RGB view built with `alpha=None` writes `null` in the alpha position of slot 0
rather than a synthesized fully-opaque channel, which is what makes
`_from_hdf_view`'s `data[3] is not None` branch reachable.

## Numerics note

`ScalarView._resolve_percentiles` keeps `np.percentile`'s `np.float64` rather than
converting to a Python `float`. Under NEP 50 a numpy scalar is a *strong* operand,
so `float32_channel -= vmin` computes in float64 and rounds once, where a weak
Python float computes in float32. The difference is a single LSB on a handful of
voxels -- but channel names are content hashes, so it silently changes on-disk node
identity. `np.float64` subclasses `float`, so the annotation still holds.

## Known bug: `mapper.py`

`Mapper.__call__` does:

```python
if isinstance(data, dataset.Vertex):
    llen = self.masks[0].shape[0]
    if data.raw:                       # <-- always truthy
        left, right = data.data[..., :llen, :], data.data[..., llen:, :]
    else:
        left, right = data[..., :llen], data[..., llen:]
```

`Vertex.raw` is a property that builds and returns a `VertexRGB`, so it is always
truthy and the `else` branch is dead. Every scalar `Vertex` takes the RGB path,
which indexes as if the data had a trailing channel axis. This looks like a
leftover from a time when `raw` was a boolean flag.

Not fixed here, because correcting it changes runtime behaviour and needs a
regression test. It is now visible to mypy: `Vertex.__getitem__` returns `Self`
rather than `Any`, so the checker reports that the dead branch yields `Vertex`
objects where arrays are expected.

## Other known issues, outside this package

- `quickflat/composite.py` reads `dataview.xfmname` then tests `if xfmname is None`
  to emit a "you seem to have provided vertex data" message -- but `Vertex` has no
  `xfmname`, so it raises `AttributeError` before reaching the intended
  `ValueError`.
- `quickflat/view.py` reads `dataview.xfmname` unguarded on the `with_dropout` path.
- `webgl/view.py`'s `JSMixer.addData` references `Dataset` and `_convert_dataset`,
  neither of which exists; it is permanently broken and xfailed.
- `volume.py`'s `epi2anatspace_fsl` calls `normalize(...).data` then `.subject` on
  the resulting array. Unreachable -- the function raises `NotImplementedError`
  first.
- `Dataview2D.to_json` ignores its `simple` flag. Preserved deliberately: the webgl
  packer only ever calls `to_json(simple=True)` on the scalar channels yielded by
  `uniques()`, never on a 2D view.
- `Database.get_cache` builds its hash from `auxfile.h5.filename`; `md5` needs
  bytes, so this raised an uncaught `TypeError` until it was given an `.encode()`.
