"""Backwards-compatibility shim for the pre-restructure class names.

``BrainData``, ``VolumeData`` and ``VertexData`` used to be separate abstract
classes that were joined to ``Dataview`` by multiple inheritance in ``Volume``
and ``Vertex``. That multiple inheritance was load-bearing -- ``BrainData``
called ``super().to_json()`` and ``super().copy()``, which only resolved because
``Volume``'s MRO threaded through ``Dataview`` -- and it is what made the package
impossible to type. The hierarchy is now linear, so the three names are aliases.

The aliases are chosen so that ``isinstance`` behaves exactly as before:
``isinstance(x, VolumeData)`` was only ever true for ``Volume``, because
``Volume2D`` and ``VolumeRGB`` never inherited from ``VolumeData``.

Nothing in the package reaches through this module any more.
``cortex/blender/__init__.py`` used to
(``isinstance(braindata, dataset.braindata.VertexData)``) and now tests
``dataset.Vertex`` directly. What is left is ``cortex/tests/test_braindata.py``,
which imports ``_hash`` from here, and ``test_dataset.py``, which imports the
three aliases to keep the shim honest. It stays for code outside the repo.
"""

from __future__ import annotations

from ._hdf import _find_mask, _hash, _hdf_write
from .views import ScalarView, Vertex, Volume, _masker

#: Any single array of scalar values living in some brain space.
BrainData = ScalarView
#: Volumetric scalar data. Formerly a separate abstract base of ``Volume``.
VolumeData = Volume
#: Surface scalar data. Formerly a separate abstract base of ``Vertex``.
VertexData = Vertex

__all__ = [
    "BrainData",
    "VolumeData",
    "VertexData",
    "_masker",
    "_hash",
    "_hdf_write",
    "_find_mask",
]
