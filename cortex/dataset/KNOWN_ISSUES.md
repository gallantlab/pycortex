# Known issues in and around `cortex.dataset`

Bugs the restructure of this package found or left behind, kept out of
[INHERITANCE.md](INHERITANCE.md) so that document describes only what the class
graph *is*. Nothing here is required reading to use or extend the package.

Three kinds are collected: defects that are still live, defects outside this
package, and defects fixed during the restructure -- the last so that a fix does
not read as an unrelated edit, and so the reason a rule exists is recoverable
without putting it in the rule.

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
- `quickflat.make_movie` is `raise NotImplementedError` on its first line, with a
  whole body behind it that has never run. That is true of `main` and `types-orig`
  too, and no branch has ever had a test for it, so its contents have decayed
  against the API around them unnoticed. It is nonetheless exported and documented
  in `api_reference_flat.rst`.

  Two of its calls had rotted. The `make_flatmap_image` one is now fixed: it passed
  `(data, subject, xfmname, ...)` to a signature that has been
  `(braindata, height, recache, nanmean, **kwargs)` since at least `main`, so
  `subject` bound to `height` and the keywords then collided outright. Rebinding
  alone was not enough — `make_flatmap_image` renders one frame and raises on a
  longer leading axis, so a 4D dataset must go through it frame by frame, which is
  what it now does.

  `make_figure(ims[0], subject, vmin=..., vmax=...)` is still wrong and cannot be
  repaired by rebinding: it takes a `Dataview` and renders it itself, so it cannot
  accept a pre-rendered image; `subject` lands on `recache`; and it has neither
  `vmin`/`vmax` nor `**kwargs`, because a view's colour range now lives on the
  view. Fixing it is a design choice about how the movie should be assembled, and
  any choice is unverifiable while the body is unreachable, so it is left annotated
  in place.

  mypy catches none of this: `make_movie` is unannotated, so its body is unchecked
  — the same blind spot that hid the `renderer_data` fork in `webgl/data.py`.

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



## Fixed during the restructure

### `quickflat.add_cutout` used two removed APIs

`make_figure(..., cutout=...)` raised on any current install. Two APIs, both on the
path that clips a layer to a cutout region:

- `from scipy.misc import imresize`, removed in SciPy 1.3 (2019). Now
  `PIL.Image.fromarray(mask, mode="F").resize(..., Image.Resampling.BILINEAR)`,
  which resizes the float32 mask in place of a round trip through uint8 --
  `imresize` bytescaled a float input, so the old code divided by 255 to undo it.
  Two traps: PIL takes `(width, height)` where the layer's shape is
  `(height, width)`, and `Image.BILINEAR` is a deprecated alias that Pillow's stubs
  do not declare, so the `Resampling` spelling is the one that type-checks.
- `np.cast['float32']`, removed in NumPy 2.0. Now
  `np.asarray(im).astype(np.float32)`, which is what `np.cast[t]` was defined as.

Neither was visible to mypy until `add_cutout` gained a signature: an unannotated
body is not checked. Nothing covered it either, which is why it went unnoticed
across two dependency removals. Two tests now do:
`test_cutout_composites_a_figure` calls `make_figure(cutout=...)` end to end, and
`test_cutout_resizes_a_layer_that_is_one_pixel_off` drives the resize branch --
reachable only when two layers disagree by a pixel -- and asserts through a spy
that the resize ran on float32, since a wrong dtype there would change transparency
silently rather than raise.

One incidental change: the `print(f'Resizing! ...')` on that branch is gone. It was
a debug print to stdout from library code.

### `uniques()` promised half of what its consumer needs

`Packable` declared only `name`, but `webgl.data.Package` reads `name` *and*
`renderer_data` off each unique. It went unnoticed because `Package.__init__` was
unannotated, so mypy skipped its body -- the same blind spot that still hides
`quickflat.make_movie`. Annotating the signature reported
`"Packable" has no attribute "renderer_data"` at once.

Fixed by making `Packable` a subclass of `RenderableView`, which every packable
column already was in fact.

### A third-party space was never reached, and its data vanished

`register_space` appended, so a space registered by a third party -- necessarily
after `cortex.dataset` registers its own two -- landed *behind* `SurfaceSpace`,
which claims any node without a transform. `SurfaceSpace` took the node, `wrap`
built a `Vertex`, `coerce` raised on the vertex count, and `cortex.load` swallowed
that per view and returned an **empty** Dataset. Writing was unaffected
throughout, so the file was intact and the load was silently empty.

Unnoticed because both built-in spaces are registered inside `_space.py` in an
order chosen by hand, so append semantics were never exercised by a third
registration. Fixed by inserting non-fallback spaces ahead of every fallback one.

### Surface RGB movies were unconstructible

`VertexRGB` sized its default alpha from `vertices.shape[1]` -- the vertex count
alone, which carries no frame axis -- so a multi-frame `VertexRGB` raised. Fixed by
sizing from the channel's stored array instead.

`VolumeRGB` sized from `volume`, the same values with the frame axis already
prepended, so its auto-alpha came out wrapped as a one-frame movie. That unmasks to
an identical array, and an auto-generated alpha is never written to HDF nor shipped
on its own, so nothing outside the class could observe it.

### A float32 view could not be saved at all

`ScalarView._write_cmap_slots` dumps `[self.vmin]`, and a defaulted `vmin` on
float32 data is an `np.float32`, which `json.dumps` refuses:

```
TypeError: Object of type float32 is not JSON serializable
```

float64 data worked purely by the accident that `np.float64` subclasses `float`.
The four view types carrying cmap bounds were affected (`Volume`, `Vertex`,
`Volume2D`, `Vertex2D`); the RGB views were not, having no `vmin`/`vmax` and
writing their channels as bare data nodes. The same root cause broke `make_static`
and `show`, where `to_json(simple=False)` puts the same defaulted bounds into the
browser payload; only `serve.py`'s websocket path had an encoder (`NPEncode`) that
coped.

Fixed by routing every write-path `json.dumps` through `views._dumps` -- see
[INHERITANCE.md](INHERITANCE.md) for why the conversion belongs at that boundary
and not in `_resolve_percentiles`.
