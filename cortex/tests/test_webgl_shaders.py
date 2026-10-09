"""Tests that every webgl shader variant compiles and links.

WebGL only guarantees a maximum number of vertex attribute slots (usually
16). If a shader asks for too many, it compiles but fails at *link* time. This
shows up in the viewer as an unexplained black screen. This is how
``Vertex2D`` broke (gh-714).

Linking a shader needs no subject database or render, just Chromium, so it
costs milliseconds instead of seconds per case -- cheap enough to cover every
combination of options here, including the equivolume shaders that no render
test enables.
"""

import itertools
import os
from typing import TYPE_CHECKING, Callable, Iterator, TypedDict

import pytest

import cortex.webgl
from cortex.export.headless import SWIFTSHADER_CHROMIUM_ARGS
from cortex.tests.testing_utils import has_playwright

if TYPE_CHECKING:
    # What pytest.param returns; pytest does not export it publicly.
    from _pytest.mark.structures import ParameterSet
    from playwright.sync_api import Page

pytestmark = pytest.mark.skipif(
    not has_playwright, reason="playwright and chromium are required"
)

JS_PATH = os.path.join(os.path.dirname(cortex.webgl.__file__), "resources", "js")

# Loads the viewer's shader library into a page and exposes a hook that builds
# one shader variant through THREE.ShaderMaterial -- the same constructor
# DataView.getShader() uses in dataset.js -- and renders it once. That way
# THREE.WebGLProgram builds the real prefix (precision, light/morph defines,
# attribute declarations) itself instead of a copy hand-kept here going stale.
PAGE = """
<html><body><canvas id="c" width="32" height="32"></canvas>
<script src="file://__JSDIR__/jquery-2.1.1.min.js"></script>
<script src="file://__JSDIR__/dat.gui.min.js"></script>
<script src="file://__JSDIR__/three.js"></script>
<script src="file://__JSDIR__/shaderlib.js"></script>
<script src="file://__JSDIR__/movement.js"></script>
<script src="file://__JSDIR__/figure.js"></script>
<script src="file://__JSDIR__/axes3d.js"></script>
<script>
// Use the viewer's own renderer, camera, lights and scene. The stubs stand in
// for what mriview.js's Viewer provides.
var axes = Object.create(jsplot.Axes3D.prototype);
axes.canvas = $('#c');
axes.figure = {register: function() {}};
axes.setFrame = function() {};
jsplot.Axes3D.call(axes, axes.figure);
var renderer = axes.renderer, camera = axes.camera;
var scene = axes.setGrid(1, 1, 0);

function linkResult(material) {
    var gl = renderer.context;
    var program = material.program;
    var vs = program.vertexShader, fs = program.fragmentShader, gp = program.program;
    var linked = !!gl.getProgramParameter(gp, gl.LINK_STATUS);
    var uniforms = [];
    var nuniforms = linked ? gl.getProgramParameter(gp, gl.ACTIVE_UNIFORMS) : 0;
    for (var i = 0; i < nuniforms; i++)
        uniforms.push(gl.getActiveUniform(gp, i).name);
    return {
        linked: linked,
        log: [gl.getShaderInfoLog(vs), gl.getShaderInfoLog(fs),
              gl.getProgramInfoLog(gp)].join("\\n"),
        uniforms: uniforms,
    };
}

var ITEM_SIZE = {f: 1, v2: 2, v3: 3, v4: 4};

// Link one {vertex, fragment, attrs} object, as Shaders[name](opts) returns.
window.linkCode = function(code) {
    // The pick shader returns one fragment shader per axis; any of them will
    // do, they all go with the vertex shader that holds the attributes.
    var frag = code.fragment instanceof Array ? code.fragment[0] : code.fragment;

    // A handful of dummy vertices are enough to let every attribute the
    // shader declares (position/normal/uv/uv2 plus whatever custom ones
    // code.attrs lists) bind to a buffer, which is all link status needs.
    var nverts = 3;
    var geometry = new THREE.BufferGeometry();
    geometry.addAttribute('position', new THREE.BufferAttribute(new Float32Array(nverts * 3), 3));
    geometry.addAttribute('normal', new THREE.BufferAttribute(new Float32Array(nverts * 3), 3));
    geometry.addAttribute('uv', new THREE.BufferAttribute(new Float32Array(nverts * 2), 2));
    geometry.addAttribute('uv2', new THREE.BufferAttribute(new Float32Array(nverts * 2), 2));
    for (var name in code.attrs) {
        var itemSize = ITEM_SIZE[code.attrs[name].type] || 1;
        geometry.addAttribute(name, new THREE.BufferAttribute(new Float32Array(nverts * itemSize), itemSize));
    }

    // The viewer also passes lights:true, which needs its full uniform set.
    // Linking doesn't: the light count comes from the scene either way.
    var material = new THREE.ShaderMaterial({
        vertexShader: code.vertex,
        fragmentShader: frag,
        attributes: code.attrs,
    });
    var mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    scene.add(mesh);
    renderer.render(scene, camera);
    scene.remove(mesh);

    return linkResult(material);
};
</script></body></html>
"""


class LinkResult(TypedDict):
    """What the page's ``linkCode`` hook returns."""

    linked: bool
    log: str
    uniforms: list[str]


# Link one Shaders[name](opts) variant / raw (vertex, fragment) GLSL source.
LinkShader = Callable[[str, dict[str, object]], LinkResult]
LinkRawShader = Callable[[str, str], LinkResult]


# The options the viewer generates surface shaders with. ``morphs`` is the
# number of surfaces to mix between (anatomical, inflated and flat), ``volume``
# says the subject has a white matter surface; the rest come from the dataview
# and from the surface menu.
SURFACE_OPTS = dict(morphs=3, volume=1, layers=1, rois=True, extratex=False,
                    halo=False, dither=False, voxline=False, sampler="nearest")


def _surface_variants() -> Iterator["ParameterSet"]:
    """Every (shader, opts) pair the viewer can ask for a surface shader.

    On top of the dataview and surface options, gh-695 added three that
    together decide how much GLSL the volume-sampling shader carries:
    ``dataalpha`` adds a second pair of samplers and the alpha-map arithmetic,
    ``nanmean`` changes how layer samples are combined, and ``layers`` decides
    how many of those sampling blocks are emitted.

    ``voxline`` (the ``[webgl_viewopts] voxlines`` debug grid) only appends a
    fixed blend to ``surface_pixel`` that reads the cortical sheet position and
    the final color -- no attributes or samplers -- so it is checked once per
    color type on the largest variant instead of doubling the whole product.
    """
    bools = (False, True)
    for shader, rgb, twod, hasflat, equivolume, dataalpha, nanmean, layers in itertools.product(
        ("surface_vertex", "surface_pixel"), bools, bools, bools, bools,
        bools, bools, (1, 32),
    ):
        if rgb and twod:
            continue  # RGB data has no second dimension
        if shader == "surface_vertex" and (dataalpha or not nanmean or layers > 1):
            # Vertex data folds its alpha map into the ``nanmask`` attribute
            # precisely because it has no attribute slot to spare, and it has
            # no cortical depth to average over: none of these reach it.
            continue
        if rgb and dataalpha:
            # RGB dataviews carry their alpha in the texture's own fourth
            # channel; the viewer never asks for a separate alpha map.
            continue
        opts = dict(SURFACE_OPTS, rgb=rgb, twod=twod, hasflat=hasflat,
                    equivolume=equivolume, dataalpha=dataalpha,
                    nanmean=nanmean, layers=layers)
        name = "%s-%s%s%s%s%s%s%s" % (
            shader,
            "rgb" if rgb else "cmap",
            "-2d" if twod else "",
            "-flat" if hasflat else "",
            "-equivolume" if equivolume else "",
            "-dataalpha" if dataalpha else "",
            "" if nanmean else "-no_nanmean",
            "-%dlayer" % layers if layers > 1 else "",
        )
        yield pytest.param(shader, opts, id=name)

    # `voxline`
    for rgb, twod in ((False, False), (False, True), (True, False)):
        opts = dict(SURFACE_OPTS, rgb=rgb, twod=twod, hasflat=True,
                    equivolume=True, dataalpha=not rgb, nanmean=True,
                    layers=32, voxline=True)
        name = "surface_pixel-%s%s-voxline" % ("rgb" if rgb else "cmap",
                                              "-2d" if twod else "")
        yield pytest.param("surface_pixel", opts, id=name)


def _main_variants() -> Iterator["ParameterSet"]:
    """Every (shader, opts) pair the slice planes ask ``main`` for.

    The slice planes (sliceplane.js) are the only live users of ``main``; they
    only show volume data and fix every option except ``raw`` and ``twod``.
    """
    for raw, twod in ((False, False), (False, True), (True, False)):
        opts = dict(sampler="nearest", raw=raw, twod=twod, voxline=False,
                    viewspace=True)
        name = "main-%s%s" % ("rgb" if raw else "cmap", "-2d" if twod else "")
        yield pytest.param("main", opts, id=name)


def _variants() -> Iterator["ParameterSet"]:
    yield from _surface_variants()
    yield from _main_variants()
    # The shaders the picker renders with; they morph the same geometry but
    # carry no data.
    yield pytest.param("pick", dict(morphs=3, volume=1), id="pick")
    yield pytest.param("depth", dict(morphs=3, volume=1), id="depth")


@pytest.fixture(scope="module")
def webgl_page(tmp_path_factory: pytest.TempPathFactory) -> Iterator["Page"]:
    """Load the shader-linking page into a real GL context, shared by both hooks.

    Skips without WebGL, but fails on any page script error.
    """
    from playwright.sync_api import sync_playwright

    page_path = tmp_path_factory.mktemp("shaders") / "shaders.html"
    page_path.write_text(PAGE.replace("__JSDIR__", JS_PATH))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, args=SWIFTSHADER_CHROMIUM_ARGS
        )
        try:
            page = browser.new_page()
            errors: list[str] = []
            page.on("pageerror",
                    lambda error: errors.append(error.stack or str(error)))
            page.goto("file://%s" % page_path, wait_until="load", timeout=60000)
            if not page.evaluate(
                "() => !!document.createElement('canvas').getContext('webgl')"
            ):
                pytest.skip("no WebGL context available in this browser")
            if errors or not page.evaluate("() => !!window.linkCode"):
                pytest.fail(
                    "the shader-linking page failed to load:\n%s"
                    % "\n".join(errors or ["window.linkCode is not defined"])
                )
            yield page
        finally:
            browser.close()


@pytest.fixture(scope="module")
def link_shader(webgl_page: "Page") -> LinkShader:
    """Return a function linking one shader variant in a real GL context."""
    return lambda shader, opts: webgl_page.evaluate(
        "args => window.linkCode(Shaders[args[0]](args[1]))", [shader, opts]
    )


@pytest.fixture(scope="module")
def link_raw_shader(webgl_page: "Page") -> LinkRawShader:
    """Return a function linking raw GLSL source in the same GL context."""
    return lambda vertex, fragment: webgl_page.evaluate(
        "args => window.linkCode({vertex: args[0], fragment: args[1], attrs: {}})",
        [vertex, fragment],
    )


@pytest.mark.parametrize("shader,opts", list(_variants()))
def test_shader_links(
    shader: str, opts: dict[str, object], link_shader: LinkShader
) -> None:
    """Each shader variant has to compile *and* link.

    A variant that uses more vertex attributes than the driver has slots for
    compiles fine and fails to link, which leaves the viewer showing nothing at
    all. Checking ``linked`` alone covers both, since a compile failure also
    fails to link.
    """
    result = link_shader(shader, opts)
    assert result["linked"], "%s failed to compile or link:\n%s" % (
        shader, result["log"]
    )
    if shader not in ("pick", "depth"):
        assert "directionalLightColor[0]" in result["uniforms"], (
            "%s compiled without its lighting code" % shader
        )


def test_shader_link_catches_compile_error(
    link_raw_shader: LinkRawShader
) -> None:
    """A shader that fails to *compile* must fail ``linked`` too.

    No production shader is broken this way, so nothing above exercises this
    path; this proves the assumption behind checking ``linked`` alone (a
    compile failure always fails the subsequent link) actually holds in a
    real GL context, not just per the GLSL/WebGL spec.
    """
    result = link_raw_shader(
        "this is not valid glsl;",
        "void main() { gl_FragColor = vec4(1.0); }",
    )
    assert not result["linked"], "invalid GLSL should not link:\n%s" % result["log"]
