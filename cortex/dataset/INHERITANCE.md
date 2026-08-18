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

Each concrete class has exactly two bases: its column, which carries all the state
and behaviour, and its row, which is a stateless interface. That is not the
multiple inheritance this package was restructured to remove. `BrainData` and
`Dataview` used to be unrelated base classes joined only by MI in `Volume` and
`Vertex`, and that MI was load-bearing: `BrainData.to_json` and `VolumeData.copy`
called `super()` methods that existed nowhere in their own ancestry, resolving
only because `Volume`'s MRO threaded through `Dataview`. The rows have no
`__init__`, no attributes and no cooperative `super()` chain, so nothing depends
on how they linearize. See [Narrowing the grid](#narrowing-the-grid).

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

    class HasSubject {
        <<protocol>>
        +subject
    }
    class Packable {
        <<abstract>>
        +name*
    }
    class RenderableView {
        <<abstract>>
        +sampling_data*
        +raw()*
    }
    class VolumetricView {
        <<abstract>>
        +xfmname*
        +volume*
        +sampling_data
        +raw() VolumeRGB
    }
    class SurfaceView {
        <<abstract>>
        +vertices*
        +sampling_data
        +raw() VertexRGB
        +blend_curvature()
    }

    HasSubject <|.. Dataview
    Dataview <|-- Packable
    Dataview <|-- Dataview2D
    Dataview <|-- Multiview
    Dataview <|-- RenderableView
    Packable <|-- ScalarView
    Packable <|-- DataviewRGB
    RenderableView <|-- VolumetricView
    RenderableView <|-- SurfaceView

    ScalarView <|-- Volume
    ScalarView <|-- Vertex
    Dataview2D <|-- Volume2D
    Dataview2D <|-- Vertex2D
    DataviewRGB <|-- VolumeRGB
    DataviewRGB <|-- VertexRGB

    VolumetricView <|-- Volume
    VolumetricView <|-- Volume2D
    VolumetricView <|-- VolumeRGB
    SurfaceView <|-- Vertex
    SurfaceView <|-- Vertex2D
    SurfaceView <|-- VertexRGB
```

Each concrete class reads down two edges: its column (left) and its row (right).
`blend_curvature` is drawn on `SurfaceView` because that is the only place it is
defined; `Vertex` inherits it rather than declaring its own.

### `Packable`: the unit of transport

`Packable` sits between `Dataview` and the two columns that own an array, and it is
what `uniques()` yields. It answers a different question from the row:

| base | question | who has it |
| --- | --- | --- |
| `Packable` | "is this **one addressable array**?" | scalar and RGB columns |
| `RenderableView` | "can a renderer **sample** this?" | every row |

Neither implies the other, which is the point of keeping them separate. A 2D view
is renderable but **not** packable: it owns no array, only the two channels it
decomposes into, which is exactly why it has no `name`. Conversely a bare
`ScalarView` subclass is packable but has no row.

The one member is `name`, a content hash. That is what makes `Dataset.uniques()` a
`set` rather than a list -- two views over identical data collapse to one entry and
are stored and shipped once.

`uniques()` was annotated `Iterator[Dataview]`, which is wider than the truth and
lacks the one member every consumer reaches for first. `webgl.data.Package`
type-checked only because the list it built was `Any`; nothing warned that
`Dataview` has no `name`.

**Do not hoist `name` onto the row.** `VolumeRGB.name` and `VertexRGB.name` are both
exactly `_hash(self.sampling_data)`, which makes a single definition on
`RenderableView` look free. It is not: `ScalarView.name` hashes the *stored* array,
and for a masked `Volume` that is the flat masked array, not the unmasked 3-D
`sampling_data`. Unifying them silently renames every existing HDF node. Pinned by
`test_packable_name_is_not_hoisted_onto_the_row`. What *can* collapse is the two RGB
definitions into one on `DataviewRGB`, since for RGB the stored array is the sampled
one -- but that needs `DataviewRGB` to see a row member.

What `name` addresses is also asymmetric, which is why `Packable` promises only the
name and not what it hashes: for a scalar view it is both the HDF node name and the
browser key, whereas an RGB view writes its channels as four separate HDF nodes and
uses its own `name` only as the browser key.

```mermaid
classDiagram
    direction TB
    class BrainSpace {
        <<abstract>>
        +subject
        +hdf_key$
        +xfmname*
        +coerce()*
        +is_movie()*
        +spatial_shape*
        +wrap()*
        +wrap_rgb()*
        +to_json()*
        +write_hdf_attrs()*
        +view_xfmname
        +from_hdf()$*
        +views()$*
    }
    class VolumeSpace {
        +xfmname: str
        +mask
        +mask_name
        +mask_spec
        +linear
        +unmask()
    }
    class SurfaceSpace {
        +xfmname: None
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
| `HasSubject`, `Packable`, `RenderableView`, `VolumetricView`, `SurfaceView` | `views.py` |
| `Dataview2D`, `Volume2D`, `Vertex2D` | `view2D.py` |
| `DataviewRGB`, `VolumeRGB`, `VertexRGB`, `Colors` | `viewRGB.py` |
| `BrainSpace`, `VolumeSpace`, `SurfaceSpace`, the space registry | `_space.py` |
| `Dataset` | `dataset.py` |
| `_hash`, `_hdf_write`, `_find_mask` | `_hdf.py` |
| `Renderable`, `Packable` (re-export), `ColormappedView`, `as_renderable`, `space_of` | `_typing.py` |

The row ABCs live in `views.py`, not `_typing.py`, because `Volume` and `Vertex`
inherit them; `_typing.py` imports from `views.py` and re-exports, so it holds only
the boundary helpers and aliases. `cortex/dataset/__init__.py` must import `.views`
first — it resolves its circular dependency on `view2D`/`viewRGB` with deferred
imports at the bottom of its own module, so anything reaching those earlier sees a
partially initialised module. A test pins the import order.

`braindata.py` is a compatibility shim: `BrainData`, `VolumeData` and `VertexData`
are aliases for `ScalarView`, `Volume` and `Vertex`. The aliases preserve
`isinstance` exactly -- `isinstance(x, VolumeData)` was only ever true for
`Volume`, since `Volume2D` and `VolumeRGB` never inherited it.

Nothing inside the package reaches through `braindata` any more. `cortex/blender/`
used to (`isinstance(braindata, dataset.braindata.VertexData)`) and now tests
`dataset.Vertex` directly; what remains are `cortex/tests/test_braindata.py`, which
imports `_hash` from there, and `test_dataset.py`, which imports the three aliases
precisely to keep this shim honest. It is kept for code outside the repo. One
submodule path is still pinned internally and must keep resolving:
`cortex.dataset.views.Vertex`, imported by `cortex/database.py`.

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

class MyView(ScalarView, MyRow): ...      # + space-specific accessors
class MyView2D(Dataview2D[MyView], MyRow): ...   # + a ctor forwarding space kwargs
class MyViewRGB(DataviewRGB[MyView], MyRow): ...
```

`view_xfmname` is deliberately *not* in that list: it is the one concrete member
on `BrainSpace`, derived as `None if self.xfmname is None else [self.xfmname]`, so
implementing `xfmname` gets slot 7 right for free. Override it only if a space
needs a slot-7 value that is not just its transform name.

`MyRow` is the row interface — `VolumetricView` if the data samples through a
transform, `SurfaceView` if it is per-vertex, or a new subclass of
`RenderableView` supplying `sampling_data` if it is sampled some other way. A view
that inherits no row is still a perfectly good `Dataview`; it just cannot be
passed to the flatmap renderers, and `as_renderable` will say so.

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
| both rows | `RenderableView` | |
| **column** (channels) | `ScalarView` / `Dataview2D` / `DataviewRGB` | same |

The columns were always classes; only the rows lacked a type, which is why
consumers duck-typed `hasattr(braindata, "xfmname")`.

### Nothing branches on the row

An `if volumetric / else surface` fork silently encodes "there are exactly two
rows": a third would take the `else` and then fail, or draw the wrong thing. So the
renderer asks for the two facts it needs instead of asking what it is holding:

- `view.space.xfmname` — the transform to sample *through*, or `None`. Owned by the
  space, and HDF slot 7 derives from the same value.
- `view.sampling_data` — the array to sample. Owned by the row: `VolumetricView`
  points it at `volume`, `SurfaceView` at `vertices`.

```python
def make_flatmap_image(braindata: RenderableView, ...):
    mask, extents = get_flatmask(braindata.subject, ...)
    pixmap = get_flatcache(braindata.subject, braindata.space.xfmname, ...)
    data = braindata.sampling_data
```

`as_renderable` is likewise one `isinstance(view, RenderableView)` rather than a
tuple of known rows. Adding a row therefore touches neither. Pinned by
`test_a_third_row_needs_no_change_to_the_renderer`, which renders through a row
this package has never seen.

Tests *for* a specific capability stay as positive `isinstance` checks and are
still correct — `with_dropout` needs a transform, so
`isinstance(dataview, VolumetricView)` is exactly the right question, and it
correctly rejects a third row that has none.

**Which mechanism for which question.** Protocols and ABCs are both used here, for
different questions, and the split is deliberate:

| question | mechanism | why |
| --- | --- | --- |
| "what kind of view is this?" | ABC — `VolumetricView`, `SurfaceView` | asked at *runtime*, so the check must be sound; only a nominal base gives that |
| "what does this function need?" | Protocol — `HasSubject` | a *static* contract on a parameter; never `isinstance`d |

`HasSubject` is the only Protocol left. There was a second, `SupportsCurvatureBlend`,
which vanished when `blend_curvature` moved onto `SurfaceView`: once the method had
one home, nothing needed a structural name for "things it can be called on".

`HasSubject` is deliberately **not** `runtime_checkable`, so `isinstance` against
it raises `TypeError` — which mechanically prevents the presence-only check from
creeping back in. A test pins that.

It is also declared exactly once, in `views.py`, and re-exported from `_typing.py`.
That matters more than it looks: two structurally identical Protocols type-check
interchangeably, so a duplicate declaration is invisible to mypy *and* to the test
asserting `Dataview` claims it, and the two copies then drift apart silently.

The payoff for the Protocol half is that a function can claim exactly what it
touches. Every one of `add_curvature`, `add_rois`, `add_sulci`, `add_custom` and
`add_cutout` reads nothing but `dataview.subject`, yet they were annotated
`Dataview` (or a `Union[Vertex, Volume, Dataview]` that collapses to it). They now
take `HasSubject`.

### Which class implements what

Every claim below is declared in the class's bases, not left to be rediscovered
structurally, so mypy checks it and a missing member makes the class abstract.
Pinned by `test_protocol_implementations_are_declared_not_merely_structural`.

| class | row (ABC) | column (ABC) | `HasSubject` | `blend_curvature` |
| --- | --- | --- | :-: | :-: |
| `Volume` | `VolumetricView` | `ScalarView` | ✓ | |
| `Volume2D` | `VolumetricView` | `Dataview2D[Volume]` | ✓ | |
| `VolumeRGB` | `VolumetricView` | `DataviewRGB[Volume]` | ✓ | |
| `Vertex` | `SurfaceView` | `ScalarView` | ✓ | ✓ |
| `Vertex2D` | `SurfaceView` | `Dataview2D[Vertex]` | ✓ | ✓ |
| `VertexRGB` | `SurfaceView` | `DataviewRGB[Vertex]` | ✓ | ✓ |

`HasSubject` is claimed once, by `Dataview`, so it covers every view including any
added later. `blend_curvature` is defined once, as a concrete method on
`SurfaceView` — curvature blending is a surface-only contract, so all three surface
views inherit one implementation. It used to be three identical forwarding methods
delegating to a module-level function typed against a `SupportsCurvatureBlend`
protocol; hoisting it removed the three copies, the function and the protocol.

The rows also narrow `raw`: `VolumetricView.raw` returns `VolumeRGB` and
`SurfaceView.raw` returns `VertexRGB`, rather than the base's `DataviewRGB`. So
`view.raw.volume` and `view.raw.left` resolve without a further cast.

Note that inheriting a Protocol explicitly does **not** re-enable `isinstance`
against it — that still requires `@runtime_checkable`, which these deliberately
lack. The claim is checked statically; the runtime guard stays shut.

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
MRO. The MROs here linearize cleanly:

```
Volume    -> ScalarView  -> Packable    -> VolumetricView -> RenderableView
          -> Dataview -> HasSubject -> Protocol -> Generic -> ABC -> object
VertexRGB -> DataviewRGB -> Packable    -> SurfaceView    -> RenderableView
          -> Dataview -> HasSubject -> Protocol -> Generic -> ABC -> object
Volume2D  -> Dataview2D  -> VolumetricView -> RenderableView -> Dataview
          -> HasSubject  -> Protocol -> Generic -> ABC -> object
```

Column before row in all three, because the column is listed first in the bases and
carries the implementations; the row contributes only defaults such as
`sampling_data` and `blend_curvature`, which nothing overrides.

That order is load-bearing for `Packable`. It declares `name` abstract, and
`Packable` precedes the row in the MRO, so had it also declared `sampling_data` the
abstract stub would have shadowed the row's working implementation. `name` is safe
because both columns define it ahead of `Packable`. Anything a row implements must
therefore stay off `Packable`.

`Volume2D.volume` and `VolumeRGB.xfmname` exist to make the rows implementable.
`Volume2D` had no `volume` at all, so consumers special-cased it to reach
`.raw.volume`; `VolumeRGB.xfmname` was a stored copy of `red.xfmname`, which an
abstract property will not accept, so it is now derived. Like `VolumeRGB.volume`,
`Volume2D.volume` returns uint8 RGBA rather than the scalar array
`Volume.volume` returns — the same split `vertices` already had across `Vertex`
and `Vertex2D`.

## What a new row must implement to be rendered

The two renderers are not equally open, and the difference is not a matter of
tidiness: `sampling_data` is enough to *draw a flatmap*, but not enough to *ship
data to a browser*, because the browser needs to know how the bytes are laid out.

### `quickflat` — nothing beyond the row ABC

Implement `sampling_data` and give the space an `xfmname` (`None` if the data is
not sampled through a transform) and `quickshow`/`make_flatmap_image` work. The
renderer never asks what it is holding. Pinned by
`test_a_third_row_needs_no_change_to_the_renderer`.

Three flatmap *decorations* legitimately require a transform and will reject a row
that has none, with a `TypeError` naming the class: `with_dropout`,
`with_connected_vertices`, and `add_connected_vertices`. That is a capability
requirement, not a closed-world assumption — there is nothing a transformless row
could do with them.

### `webgl` / `webshow` — pick one of exactly two wire encodings

`webgl/data.py`'s `Package` reads `sampling_data` like everything else, but it must
then choose how the array reaches the browser, and only two encodings exist:

| | volumetric encoding | surface encoding |
| --- | --- | --- |
| array shape | `(frames, z, y, x[, 4])` | `(frames, nverts[, 4])` |
| packing | `volume.mosaic()` per frame → PNG | raw `.npy` bytes |
| JSON | sets `mosaic` to the tile shape | no `mosaic` key |
| slot 7 | `[xfmname]` | `null` |
| vertex order | n/a | **must** be permuted by `Package.reorder` into the CTM's order |
| JS path | texture, sampled through the transform | per-vertex attribute |
| premultiplied alpha | no — Three.js premultiplies on texture upload | **yes**, done in Python |

A new row must therefore inherit `VolumetricView` or `SurfaceView` *for the webgl
path to work*, even though `quickflat` would have accepted a bare `RenderableView`.
That is the one place where the row axis is genuinely not open, and it is a
constraint of the browser code, not of this package: `dataset.js` selects the path
by testing `mosaic === undefined`, so a row wanting a third layout has to add a
matching branch there (and to `shaderlib.js`, if it samples differently).

The geometry is always a surface mesh, whatever the row: `brainctm.py` builds it
from `db.get_surf`, so a volumetric row is rendered by sampling its texture at each
vertex's coordinates. A row whose data cannot be evaluated per-vertex has no webgl
representation at all.

`Package` iterates `uniques(collapse=True)`, which decomposes 2D views into their
scalar channels, so only the scalar and RGB columns ever reach it. That is why
nothing there mentions `Dataview2D`.

### Consumers that are deliberately narrower

These name a concrete class because they need a specific capability, and a new row
will be rejected rather than mis-drawn:

| consumer | requires | why |
| --- | --- | --- |
| `volume.show_slice` | `Volume` | slices the 3D array against an anatomical reference |
| `blender.add_cutdata` | `Volume` or `Vertex` | samples one colormapped value per vertex, so needs a single `cmap` |
| `Mapper.__call__` | `Vertex` | splits per-hemisphere by vertex count |

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

- `quickflat/composite.py`'s `add_connected_vertices` still reads
  `dataview.xfmname` and tests `if xfmname is None` to emit a "you seem to have
  provided vertex data" message. That branch is now dead rather than broken: the
  parameter is typed `VolumetricView`, whose `xfmname` is a `str`, and the sole
  caller guards with `isinstance`. It used to raise `AttributeError` before
  reaching the intended `ValueError`, because `Vertex` has no `xfmname` at all.
- `webgl/view.py`'s `JSMixer.addData` references `Dataset` and `_convert_dataset`,
  neither of which exists; it is permanently broken and xfailed.
- `volume.py`'s `epi2anatspace_fsl` calls `normalize(...).data` then `.subject` on
  the resulting array. Unreachable -- the function raises `NotImplementedError`
  first.
- `Dataview2D.to_json` ignores its `simple` flag. Preserved deliberately: the webgl
  packer only ever calls `to_json(simple=True)` on the scalar channels yielded by
  `uniques()`, never on a 2D view.
Fixed in passing while typing the callers, and recorded here only so the change is
not mistaken for an unrelated edit:

- `Database.get_cache` built its hash from `auxfile.h5.filename`; `md5` needs
  bytes, so this raised an uncaught `TypeError` until it was given an `.encode()`.
- `quickflat/view.py` read `dataview.xfmname` unguarded on both `with_dropout`
  paths, raising `AttributeError` on surface data. Both now check
  `isinstance(dataview, VolumetricView)` and raise a `TypeError` naming the class.
- `Dataset.get_surf` and `Database.get_surf` take `merge` and `nudge` keyword-only
  in every overload. Previously only the `merge=True` overload did, which made the
  calling convention depend on the argument's value: a positional
  `get_surf(s, t, 'both', True)` ran but matched no overload, and a positional
  dynamic flag bound the wrong one. No caller in this repo passed either
  positionally, but it is a break for anyone outside it who did.
