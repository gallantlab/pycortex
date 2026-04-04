import os
from typing import Any, Literal, overload

import numpy as _py_np
import numpy.typing as npt

_Path = str | os.PathLike[str]

PY3: bool

def read(globname: str) -> tuple[npt.NDArray[_py_np.floating], npt.NDArray[_py_np.integer]]: ...

# `polys` is float64 rather than integer for a file declaring zero faces, since
# `np.array([])` is float64. Same caveat on `read_obj` below.
def read_off(filename: _Path) -> tuple[npt.NDArray[_py_np.floating], npt.NDArray[_py_np.integer]]: ...

# dtypes are whatever was stored in the archive.
def read_npz(filename: _Path) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...
def read_gii(filename: _Path) -> tuple[npt.NDArray[_py_np.floating], npt.NDArray[_py_np.integer]]: ...
def read_stl(filename: _Path) -> tuple[npt.NDArray[_py_np.floating], npt.NDArray[_py_np.uint32]]: ...

# 2-tuple only when both flags are literal `False`; unsound for non-literal `False, False`.
@overload
def read_obj(  # type: ignore[overload-overlap]  # overlaps the bool fallback below
    filename: _Path, norm: Literal[False] = False, uv: Literal[False] = False
) -> tuple[npt.NDArray[_py_np.floating], npt.NDArray[_py_np.integer]]: ...
@overload
def read_obj(
    filename: _Path, norm: bool = False, uv: bool = False
) -> tuple[
    npt.NDArray[_py_np.floating],
    npt.NDArray[_py_np.integer],
    list[list[float]] | None,
    list[list[float]] | None,
]: ...

def read_vtk(filename: _Path) -> tuple[npt.NDArray[_py_np.float64], npt.NDArray[_py_np.uint32]]: ...

# The writers require real arrays, not just ArrayLike: they read `polys.dtype`
# and `pts.astype`, and index with `pts[polys]`.
def write_vtk(
    filename: _Path,
    pts: npt.NDArray[Any],
    polys: npt.NDArray[Any],
    norms: npt.NDArray[Any] | None = None,
) -> None: ...
def write_off(filename: _Path, pts: npt.NDArray[Any], polys: npt.NDArray[Any]) -> None: ...
def write_stl(filename: _Path, pts: npt.NDArray[Any], polys: npt.NDArray[Any]) -> None: ...
def write_gii(filename: _Path, pts: npt.NDArray[Any], polys: npt.NDArray[Any]) -> None: ...
def write_obj(
    filename: _Path,
    pts: npt.NDArray[Any],
    polys: npt.NDArray[Any],
    colors: npt.NDArray[Any] | None = None,
) -> None: ...
