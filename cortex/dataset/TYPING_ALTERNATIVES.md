# `cortex.dataset` restructuring options

Design record for the type-system restructure of this package. See
[INHERITANCE.md](INHERITANCE.md) for the map of the *current* class graph and the specific
problems each option below is trying to solve.

Four options were considered. **Option A was chosen, delivered in two stages.** The others are
recorded here with their tradeoffs so the reasoning does not have to be reconstructed.

## The problem in one paragraph

`BrainData` and `Dataview` are unrelated base classes joined only by multiple inheritance in
`Volume` and `Vertex`. That MI is load-bearing — `BrainData.to_json` and `VolumeData.copy` call
`super()` methods that exist nowhere in their own ancestry and resolve only because `Volume`'s
MRO threads through `Dataview`. The concrete classes form a 2x3 grid (volumetric/surface x
scalar/2D/RGB) but only its columns exist as classes; there is no type meaning "any volumetric
view". Baseline: **66 mypy errors inside `cortex/dataset/`**, plus ~8 downstream
`"Dataview" has no attribute "subject"/"xfmname"` in `quickflat/` and `export/`.

Requirements the options were judged against:

1. Keep the six public classes: `Volume`, `Vertex`, `Volume2D`, `Vertex2D`, `VolumeRGB`, `VertexRGB`.
2. Zero mypy errors in `cortex/dataset/` under the existing config (defaults +
   `allow_redefinition`, `disable_error_code = "import-untyped"`).
3. Make the space axis **open** — adding a new kind of brain data should not mean
   reimplementing colormapping, HDF and JSON logic three more times.
4. Old non-public names stay importable; attribute renames get deprecation shims.
5. Python >= 3.10.

---

## Option A — linear core, channel count as the inheritance axis *(chosen)*

Split the two axes and give them different mechanisms.

- **Channel count becomes the inheritance axis.** One root `Dataview`, then `ScalarView`
  (1 channel + 1D colormap), `TwoDView` (2 channels + 2D colormap), `RGBView` (3 channels +
  alpha, no colormap). All shared logic lives here. No multiple inheritance anywhere.
- **Space becomes a component**, `BrainSpace`, holding `subject` plus the geometry metadata
  (`xfmname`/`mask` for volumes, `llen`/`rlen`/`hem` for surfaces), with a registry keyed on an
  HDF discriminator so the factories extend without edits.
- **One covariant TypeVar** parameterizes the composite views by their channel type.

```python
ScalarT = TypeVar("ScalarT", bound=ScalarView, covariant=True)

class Dataview(ABC):                      # subject, state, attrs, description, priority
    space: BrainSpace
class ScalarView(Dataview): ...           # data, cmap, vmin, vmax
class TwoDView(Dataview, Generic[ScalarT]):
    dim1: ScalarT
    dim2: ScalarT
class RGBView(Dataview, Generic[ScalarT]):
    red: ScalarT
    green: ScalarT
    blue: ScalarT

class Volume(ScalarView):        space: VolumeSpace
class Volume2D(TwoDView[Volume]): space: VolumeSpace
class VolumeRGB(RGBView[Volume]): space: VolumeSpace
```

### Why one TypeVar and not two

The first sketch parameterized on space *and* channel: `TwoDView[VolumeSpace, Volume]`. That was
dropped. mypy accepts narrowing a **bare annotation** in a subclass, so `space: VolumeSpace` over
`space: BrainSpace` costs one line per concrete class and needs no type parameter. Verified by
prototype — `t.space` reveals as `VolumeSpace` with zero errors.

Note the asymmetry that motivated the current code's problems: mypy *does* flag narrowing an
**assigned** ClassVar. That is precisely the `_cls` error (`type[BrainData]` narrowed to
`type[VolumeData]`), and it is why `_cls` cannot be typed as written.

### What the TypeVar actually buys

Not saved annotation lines — roughly 10. The payoff is that **a property's return type cannot be
narrowed by re-annotation, only by re-implementing the property.** That is the direct cause of
the worst duplication in the package today: `alpha` exists twice, ~25 near-identical lines in
each RGB class, and the four-branch channel-resolution logic exists eight times. With `ScalarT`,
`alpha`, `uniques`, `copy` and the channel loops in `_write_hdf`/`volume`/`vertices` are written
once and resolve precisely per subclass.

Verified by prototype: `r.alpha` -> `Volume`, `t.dim1` -> `Volume`, `v.dim1` -> `Vertex`,
`t.uniques()` -> `tuple[Volume, Volume]`, `t.dim1.copy(...)` -> `Volume`.

It also deletes `_cls` outright. The six `self._cls._write_hdf(self.red, h5)` sites exist only to
reach the data-only write while skipping the view half; a real `_write_data_hdf` method on
`ScalarView` does that by construction.

### Costs

- **Invariance.** With an invariant TypeVar, `TwoDView[ScalarView]` rejects `Volume2D`
  (confirmed: `arg-type`). Bought off by making `dim1`/`dim2`/`red`/`green`/`blue` **read-only
  properties** and marking the TypeVar covariant — sound, because nothing in the codebase or
  tests rebinds them (`blend_curvature` mutates `blended.red.data`, it does not rebind `red`).
  The residual cost is that `v2d.dim1 = x` becomes an error, a silent break for any downstream
  code that does it. Nothing in-tree does.
- **The space/channel pairing is unenforceable.** `RGBView[Vertex]` declared with
  `space: VolumeSpace` type-checks fine. Expressing "the space must match the channel's space"
  needs higher-kinded types, which Python does not have, so it stays a runtime constructor check.
  Only reachable by a library author adding a space.
- `isinstance(x, RGBView[Volume])` is a runtime `TypeError`. Unsubscripted works and narrows to
  `RGBView[Any]`, which is what the back-compat aliases rely on.
- Sphinx renders generic bases noisily. Low impact: `docs/api_reference_flat.rst:65-71`
  autosummaries only the six concrete classes.

### Honest assessment of complexity

The generics are not the hard part. One TypeVar with a plain bound is *less* machinery than the
`_cls` indirection and eight duplicated constructor branches it replaces, and the package already
operates at this level — `_masker(Generic[T_masker])` and the constrained `ColorDtype` TypeVar
already exist here.

The hard part is extracting `BrainSpace`. `VolumeData._check_size` sets `linear`, `movie`,
`shape`, `_mask` and `mask` in a single pass, mixing data-derived facts (`linear`, `movie` — both
just `data.ndim` tests) with space-derived ones (`shape`, `mask`). That untangling carries the
most behaviour risk in the whole plan, and it carries it identically in Option B.

Hence the staging: **Stage 1 is Option A with no generics and no `BrainSpace`** — single root,
colormapping off the root, `raw` renamed, `_cls` deleted, explicit operators. That clears the
large majority of the 66 errors on its own. **Stage 2** adds `BrainSpace` and `Generic[ScalarT]`
against a green baseline.

### Cost of adding a new kind of brain data

`KSpace(BrainSpace)` + `K(ScalarView)` + `K2D(TwoDView[K])` + `KRGB(RGBView[K])`. The two
composites need only a constructor forwarding their space kwargs; all colormapping, HDF, JSON,
NaN and alpha logic is inherited. The registry means the factories and `normalize` need no edit.

---

## Option B — same restructure, no generics

Identical single-root fix and identical `BrainSpace` extraction, but zero TypeVars. Each
concrete class re-declares its channel types:

```python
class Volume2D(TwoDView):
    space: VolumeSpace
    dim1: Volume
    dim2: Volume

class VolumeRGB(RGBView):
    space: VolumeSpace
    red: Volume
    green: Volume
    blue: Volume
```

**For:** nothing to reason about. No variance, no `Any` leaking through a TypeVar bound, no
subscripted-isinstance footgun, clean autodoc. Anyone who can read the current file can read this.

**Against:** re-declaration is ~8 annotations per space family instead of 1, and — the real cost —
**`alpha` must stay duplicated.** Re-annotating a property does not narrow its return type, so
`RGBView.alpha` would return `ScalarView` and each RGB class would have to re-implement the
property to return `Volume`/`Vertex`. That is exactly the duplication the restructure is meant to
remove. Also no way to write a function generic over space: `def f(v: TwoDView[Volume])` is
inexpressible.

**Why not chosen:** it pays the full cost of the risky part (`BrainSpace`) while declining the
main benefit. If the generics had turned out to be a problem on 3.10 this was the fallback.

---

## Option C — protocol overlay, minimal churn

Keep the current class graph and the multiple inheritance. Add one cooperative base so `super()`
resolves, and express the cross-cutting concepts structurally:

```python
class _ViewRoot:                      # the "common inheritance" the TODO at views.py asked for
    def to_json(self, simple: bool = False) -> dict: return {}
    def copy(self, *a, **k): raise NotImplementedError

class BrainData(_ViewRoot): ...
class Dataview(_ViewRoot): ...
class Volume(VolumeData, Dataview): ...    # MI stays

# _typing.py
class VolumetricView(Protocol):
    xfmname: str
    @property
    def volume(self) -> NDArray: ...

VolumeLike = Volume | Volume2D | VolumeRGB
```

**For:** by far the smallest diff and the lowest risk. Keeps all twelve classes and every
attribute exactly where it is. Would resolve the `super()` errors and let consumers in
`quickflat/` annotate against something narrowable instead of `hasattr(braindata, "xfmname")`.

**Against:** it treats the symptoms. `raw`'s two incompatible meanings still collide (the tuple
form on the base, an RGB object on all six subclasses), so the six `override` errors stay unless
`raw` is renamed anyway. `_cls` unbound dispatch stays. The RGB constructors stay duplicated
eightfold. The unions are closed, so requirement 3 is not met — a new kind of brain data still
means three hand-written classes with the logic copied again, plus edits to every `VolumeLike`
union and every factory branch.

**Why not chosen:** fails the extensibility requirement, and the parts it does fix (`super()`
resolution, protocols for consumers) are a subset of what Stage 1 delivers anyway.

---

## Option D — stub-only `.pyi` overlay

Leave the runtime completely untouched. Hand-write `braindata.pyi`, `views.pyi`, `view2D.pyi`,
`viewRGB.pyi` declaring the intersection types a checker cannot infer.

**For:** fastest route to a clean mypy run, and zero behaviour risk by construction — useful if
the only goal were a green CI gate.

**Against:** the stubs drift from the implementation with nothing to detect it, and this package
is under active change. More fundamentally, the two hardest facts cannot be stated honestly in a
stub: `raw` genuinely has two incompatible types depending on which class you are looking at, and
`cmap`/`vmin`/`vmax` genuinely do not exist on RGB views — a stub that declares them would make
the `except AttributeError` blocks at `views.py:253` and `:311` look like dead code to every
reader and every checker. None of the developer-facing messiness improves.

**Why not chosen:** it would make the package *look* typed while leaving it exactly as hard to
work on, and it actively lies about two things the type system should be telling you.

---

## Python >= 3.10 constraints

Verified by prototype, executed clean on CPython 3.10.20 and 3.14.7. Any of the options above
must respect:

| Feature | Available from | Verdict |
| --- | --- | --- |
| PEP 695 `class C[T]:`, `type X = ...` | 3.12 | **unusable** — use `TypeVar` + `Generic[T]` |
| PEP 696 TypeVar defaults | 3.13 | **unusable** |
| `typing.TypeIs` | 3.13 | **unusable** — confirmed absent from both 3.10 and 3.12 |
| `typing.TypeGuard` | 3.10 | usable |
| `typing.Self` | 3.11 | usable via the `sys.version_info < (3, 11)` shim already in this package |
| `ABC` + `Generic[T]` together | 3.7 | usable, no metaclass conflict |

`TypeIs` is worth calling out because [INHERITANCE.md](INHERITANCE.md) proposes `is_volume_view` /
`is_rgb_view` helpers built on it. Adopting `TypeIs` would force `typing_extensions` from its
current `python_version < '3.11'` marker to an unconditional runtime dependency. `TypeGuard` is in
3.10 stdlib and is sufficient here, at the cost of not narrowing the negative branch.

Three files in this package lack `from __future__ import annotations` (`braindata.py`,
`dataset.py`, `view2D.py`) and need it for forward references and `X | Y` under 3.10. It must
**not** be removed from `views.py`, whose bottom-of-file circular imports of `viewRGB`/`view2D`
depend on it.

---

## Bugs found while evaluating the options

Recorded because they constrain any restructure, and because several are invisible until you go
looking. None are caused by the restructure; each changes runtime behaviour to fix and wants its
own regression test.

- **`Vertex2D` does not survive an HDF round-trip.** `_from_hdf_view` does `xfmname[0]`
  unconditionally, but slot 7 is `null` for surface data, so reload raises `TypeError`. Worse,
  `Dataset.from_file` catches `Exception` and calls `traceback.print_exc()`, so the view is
  **silently dropped** — 13 views saved, 12 reloaded. Verified against a freshly written file.
- **`mapper.py:65`** — `if data.raw:` is always truthy, because `Vertex.raw` is a property
  returning a `VertexRGB`. Every scalar `Vertex` takes the RGB indexing path and the `else` is
  dead. Looks like a leftover from when `raw` was a boolean flag.
- **The RGB classes have no working `copy()`.** `Dataview.copy` splats `cmap=`/`vmin=`/`vmax=`
  into `self.__class__(...)`, which the RGB constructors do not accept. This is why
  `blend_curvature` reaches for `deepcopy`.
- **`alpha` mutates its input on every read.** When `_alpha` is a `Volume`/`Vertex`, the getter
  writes into the caller's object (`alpha.volume[mask] = alpha.vmin`), re-deriving and re-mutating
  on each access. `uniques()` and `_write_hdf` each read it more than once and get different
  objects; they agree only because names are content hashes.
- **`alpha` is never `None`**, so `if self.alpha is not None` in `uniques` and `_write_hdf` is
  always true, the HDF `alpha=None` slot is never written, and `_from_hdf_view`'s
  `if data[3] is not None` never sees `None` for files pycortex wrote.
- **The 2D and RGB families disagree on `subject=`.** `Volume2D`/`Vertex2D` reject a non-`None`
  `subject` when given channel objects; the RGB classes accept a matching one. Since
  `_from_hdf_view` always forwards `subject=`, `Dataview.from_hdf(node, subject="X")` on a 2D view
  raises `TypeError`.
- **`_apply_nan_mask` uses `hasattr(alpha, "volume")` to mean "is a volume".** `volume` is a
  property on `Volume` but a *method* on `Vertex`, so the test is true for both and
  `alpha.volume.shape` would raise on a `Vertex`.
- **`color_voxels` does `alpha[mask] = 0`** (carrying its own `# TODO: this seems like an actual
  issue`): raises `TypeError` for a `Volume`/`Vertex` alpha, mutates a caller-owned ndarray
  otherwise, and hardcodes `0` instead of the object's `vmin`.
- **`Dataview2D.to_json` ignores its `simple` flag entirely** and uses
  `self.vmin or d1js['vmin'][0]`, so an explicit `vmin=0` is silently replaced by a percentile.
  Masked today only because `Dataview2D.uniques` yields the dims, so the webgl packer never calls
  `to_json(simple=True)` on a 2D view.
- **`__neg__` and `__abs__` are generated with a binary signature** by
  `BrainData._add_numpy_methods`, along with a dead `__div__` (Python 2 only).
- `quickflat/composite.py:508-510` and `quickflat/view.py:170` read `.xfmname` unguarded on a
  `Dataview`; `webgl/view.py:730` references `Dataset` and `_convert_dataset`, neither of which
  exists (`addData` is permanently broken, xfailed).

## Constraint that bounds all options: the wire format

`webgl/resources/js/dataset.js` dispatches **structurally, not nominally** — no Python class name
reaches the browser:

- `mosaic === undefined` => surface data
- `json.data[i] instanceof Array` => 2D view
- `json.raw` => RGB (4-channel uint8 texture path)
- `json.vmin[0] instanceof Array` => 2D again
- `module.makeFrom(dvx, dvy)` synthesizes a 2D view client-side

So the eight-slot `/views` layout and the `to_json` key shapes are a hard interface, including the
`"null"` written into slots 2-4 for RGB views. Any restructure must diff the saved HDF and
`to_json` output before and after; a shape change breaks the viewer silently rather than noisily.
