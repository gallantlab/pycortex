"""Makes an interactive viewer for viewing data in a browser"""

from typing import TYPE_CHECKING

from ..utils import DocLoader

if TYPE_CHECKING:
    from cortex.webgl.view import show as _show
    from cortex.webgl.view import show_multi as _show_multi
    from cortex.webgl.view import make_static as _static
else:
    _show = None
    _show_multi = None
    _static = None

show = DocLoader("show", ".view", "cortex.webgl", actual_func=_show)
show_multi = DocLoader("show_multi", ".view", "cortex.webgl", actual_func=_show_multi)
make_static = DocLoader("make_static", ".view", "cortex.webgl", actual_func=_static)
