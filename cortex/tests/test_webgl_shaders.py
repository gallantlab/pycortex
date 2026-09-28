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
from typing import Any, Callable, Iterator

import pytest

import cortex.webgl
from cortex.export.headless import SWIFTSHADER_CHROMIUM_ARGS
from cortex.tests.testing_utils import has_playwright

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
<script src="file://__JSDIR__/three.js"></script>
<script src="file://__JSDIR__/shaderlib.js"></script>
<script>
var renderer = new THREE.WebGLRenderer({
    canvas: document.getElementById('c'), antialias: false
});
// THREE.WebGLRenderer asks for these too; the fragment shaders use fwidth
// (derivatives) and float textures.
renderer.context.getExtension('OES_standard_derivatives');
renderer.context.getExtension('OES_texture_float');
var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);

var ITEM_SIZE = {f: 1, v2: 2, v3: 3, v4: 4};

window.linkShader = function(shadername, opts) {
    var code = Shaders[shadername](opts);
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

    // The real viewer passes lights:true (dataset.js's getShader), but that
    // only makes THREE.WebGLRenderer refresh built-in light uniforms against
    // the material's uniforms object -- which needs the viewer's full merged
    // uniform set to exist. None of these shaders reference THREE's light
    // uniforms (they compute shading themselves), and the MAX_*_LIGHTS
    // defines this test cares about come from the renderer's light count
    // regardless of this flag, so leaving it out avoids that crash for free.
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

    var gl = renderer.context;
    var program = material.program;
    var vs = program.vertexShader, fs = program.fragmentShader, gp = program.program;
    return {
        linked: !!gl.getProgramParameter(gp, gl.LINK_STATUS),
        log: [gl.getShaderInfoLog(vs), gl.getShaderInfoLog(fs),
              gl.getProgramInfoLog(gp)].join("\\n"),
    };
};

// Same path as linkShader, but from raw GLSL source instead of a Shaders[]
// lookup, so a test can hand it deliberately invalid GLSL.
window.linkRawShader = function(vertexShader, fragmentShader) {
    var geometry = new THREE.BufferGeometry();
    geometry.addAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3));
    var material = new THREE.ShaderMaterial({vertexShader: vertexShader, fragmentShader: fragmentShader});
    var mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    scene.add(mesh);
    renderer.render(scene, camera);
    scene.remove(mesh);

    var gl = renderer.context;
    var program = material.program;
    var vs = program.vertexShader, fs = program.fragmentShader, gp = program.program;
    return {
        linked: !!gl.getProgramParameter(gp, gl.LINK_STATUS),
        log: [gl.getShaderInfoLog(vs), gl.getShaderInfoLog(fs),
              gl.getProgramInfoLog(gp)].join("\\n"),
    };
};
</script></body></html>
"""

# The options the viewer generates surface shaders with. ``morphs`` is the
# number of surfaces to mix between (anatomical, inflated and flat), ``volume``
# says the subject has a white matter surface; the rest come from the dataview
# and from the surface menu.
SURFACE_OPTS = dict(morphs=3, volume=1, layers=1, rois=True, extratex=False,
                    halo=False, dither=False, voxline=False, sampler="nearest")


def _surface_variants() -> Iterator[Any]:
    """Every (shader, opts) pair the viewer can ask for a surface shader."""
    bools = (False, True)
    for shader, rgb, twod, hasflat, equivolume in itertools.product(
        ("surface_vertex", "surface_pixel"), bools, bools, bools, bools
    ):
        if rgb and twod:
            continue  # RGB data has no second dimension
        opts = dict(SURFACE_OPTS, rgb=rgb, twod=twod,
                    hasflat=hasflat, equivolume=equivolume)
        name = "%s-%s%s%s%s" % (
            shader,
            "rgb" if rgb else "cmap",
            "-2d" if twod else "",
            "-flat" if hasflat else "",
            "-equivolume" if equivolume else "",
        )
        yield pytest.param(shader, opts, id=name)


def _variants() -> Iterator[Any]:
    yield from _surface_variants()
    # The shaders the picker renders with; they morph the same geometry but
    # carry no data.
    yield pytest.param("pick", dict(morphs=3, volume=1), id="pick")
    yield pytest.param("depth", dict(morphs=3, volume=1), id="depth")


@pytest.fixture(scope="module")
def webgl_page(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """Load the shader-linking page into a real GL context, shared by both hooks."""
    from playwright.sync_api import sync_playwright

    page_path = tmp_path_factory.mktemp("shaders") / "shaders.html"
    page_path.write_text(PAGE.replace("__JSDIR__", JS_PATH))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, args=SWIFTSHADER_CHROMIUM_ARGS
        )
        page = browser.new_page()
        page.goto("file://%s" % page_path, wait_until="load", timeout=60000)
        if not page.evaluate("() => !!window.linkShader"):
            browser.close()
            pytest.skip("no WebGL context available in this browser")
        yield page
        browser.close()


@pytest.fixture(scope="module")
def link_shader(webgl_page: Any) -> Callable[[str, dict], Any]:
    """Return a function linking one shader variant in a real GL context."""
    return lambda shader, opts: webgl_page.evaluate(
        "args => window.linkShader(args[0], args[1])", [shader, opts]
    )


@pytest.fixture(scope="module")
def link_raw_shader(webgl_page: Any) -> Callable[[str, str], Any]:
    """Return a function linking raw GLSL source in the same GL context."""
    return lambda vertex, fragment: webgl_page.evaluate(
        "args => window.linkRawShader(args[0], args[1])", [vertex, fragment]
    )


@pytest.mark.parametrize("shader,opts", list(_variants()))
def test_shader_links(
    shader: str, opts: dict, link_shader: Callable[[str, dict], Any]
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


def test_shader_link_catches_compile_error(
    link_raw_shader: Callable[[str, str], Any]
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
