# Working branches and the order to land them

Notes for moving the in-progress type-annotation work onto `main`, or onto each
other. Read this before rebasing: the branches are a stack in places and
independent in others, and the difference matters.

## The stack

```
main
 └── types-orig ............... pre-existing annotation work (openctm/formats stubs, etc.)
      └── types-data .......... dataset restructure, unions + TypeIs narrowing
      └── types-data-protocols  dataset restructure, row ABCs  <-- current
                                (an alternative to types-data, not on top of it)

main
 └── visual-regression-tests .. render-consistency tests, independent of all the above
```

`types-data` and `types-data-protocols` are **alternatives**, not a sequence. They
share history up to the restructure and then diverge on how the volumetric/surface
distinction is expressed in the type system; see
`cortex/dataset/TYPING_ALTERNATIVES.md` for what was tried and why. Land one, not
both.

## Order of operations

1. **Rebase `types-orig` onto `main` first.** Everything in `types-data*` sits on
   top of it, so rebasing a dataset branch before its base only creates conflicts
   twice.
2. **Then rebase the chosen dataset branch** (`types-data-protocols`, or
   `types-data`) onto the rebased `types-orig`.
3. **Then cherry-pick the testing changes** from `visual-regression-tests`. They
   are deliberately *not* part of the dataset branches' own history — see below.

## Why the testing changes are cherry-picked rather than merged

`visual-regression-tests` forks directly from `main` and touches only
`cortex/tests/test_webgl_headless.py`, `setup.py` and
`cortex/tests/reference_images/`. It shares no files with the dataset work, so it
rebases and cherry-picks onto anything without conflict, and it passes standalone
on `main` — verified, not assumed.

That independence is the point. The reference images were rendered **on `main`**,
so the test asserts that a given branch reproduces main's output pixel for pixel.
Keeping it on its own branch means:

- it can be reviewed and landed on its own schedule, before or after the
  restructure;
- it can be cherry-picked onto *any* branch to check that branch against main;
- the restructure's history does not contain the very test used to judge it.

On `types-data-protocols` it is applied with `git cherry-pick -x`, so each commit
records the branch commit it came from. If `visual-regression-tests` is amended,
drop the cherry-picks and re-apply rather than editing them in place.

## Gotchas

- **Regenerating reference images is how a real regression gets blessed.** They are
  a `main` baseline on purpose. If a rebase makes the visual test fail, that is
  the test doing its job; read `cortex/tests/reference_images/README.md` before
  reaching for `REGENERATE_REFERENCE_IMAGES=1`.
- The test skips, rather than fails, when the reference images are absent, so the
  first of the two testing commits is safe to land alone.
- The images are kept out of the wheel and in the source tarball; if you move them,
  keep `exclude_package_data` in `setup.py` and `MANIFEST.in` in agreement.
