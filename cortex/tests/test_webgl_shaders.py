"""Tests that every webgl shader variant compiles and links.

WebGL only guarantees 16 vertex attribute slots (``MAX_VERTEX_ATTRIBS``), and
the surface shaders use nearly all of them. A shader that asks for one slot too
many still compiles: it fails at *link* time, which three.js only reports on
the browser console and which shows up in the viewer as an unexplained black
screen. That is how ``Vertex2D`` data broke (gh-714), so every combination of
options the viewer generates shaders with is linked here.

These tests only need Chromium; no subject database or viewer is involved.
"""

import json
import os

import pytest

import cortex.webgl
from cortex.tests.testing_utils import has_playwright

pytestmark = pytest.mark.skipif(
    not has_playwright, reason="playwright and chromium are required"
)

JS_PATH = os.path.join(os.path.dirname(cortex.webgl.__file__), "resources", "js")

# The declarations THREE.WebGLProgram prepends to every shader it builds
# (three.js r69, resources/js/three.js). Only the ones the surface shaders
# actually rely on are listed; a missing one shows up as a compile error rather
# than as a silently passing test.
VERTEX_PREFIX = """
precision highp float;
precision highp int;
#define MAX_DIR_LIGHTS 3
#define MAX_POINT_LIGHTS 0
#define MAX_SPOT_LIGHTS 0
#define MAX_HEMI_LIGHTS 0
#define MAX_SHADOWS 0
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform vec3 cameraPosition;
attribute vec3 position;
attribute vec3 normal;
attribute vec2 uv;
attribute vec2 uv2;
"""

FRAGMENT_PREFIX = """
precision highp float;
precision highp int;
#define MAX_DIR_LIGHTS 3
#define MAX_POINT_LIGHTS 0
#define MAX_SPOT_LIGHTS 0
#define MAX_HEMI_LIGHTS 0
#define MAX_SHADOWS 0
uniform mat4 viewMatrix;
uniform vec3 cameraPosition;
"""

# Loads the viewer's shader library into a page and exposes a hook that builds
# one shader variant and links it, the way THREE.WebGLProgram does.
PAGE = """
<html><body><canvas id="c" width="32" height="32"></canvas>
<script src="file://__JSDIR__/three.js"></script>
<script src="file://__JSDIR__/shaderlib.js"></script>
<script>
var gl = document.getElementById('c').getContext('webgl');
// THREE.WebGLRenderer asks for these too; the fragment shaders use fwidth
// (derivatives) and float textures.
gl.getExtension('OES_standard_derivatives');
gl.getExtension('OES_texture_float');
var VERTEX_PREFIX = __VERTEX_PREFIX__;
var FRAGMENT_PREFIX = __FRAGMENT_PREFIX__;

function compile(type, source) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    return shader;
}

window.linkShader = function(shadername, opts) {
    var code = Shaders[shadername](opts);
    // The pick shader returns one fragment shader per axis; any of them will
    // do, they all go with the vertex shader that holds the attributes.
    var frag = code.fragment instanceof Array ? code.fragment[0] : code.fragment;
    var vs = compile(gl.VERTEX_SHADER, VERTEX_PREFIX + code.vertex);
    var fs = compile(gl.FRAGMENT_SHADER, FRAGMENT_PREFIX + frag);
    var program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    var result = {
        compiled: !!(gl.getShaderParameter(vs, gl.COMPILE_STATUS) &&
                     gl.getShaderParameter(fs, gl.COMPILE_STATUS)),
        linked: !!gl.getProgramParameter(program, gl.LINK_STATUS),
        log: [gl.getShaderInfoLog(vs), gl.getShaderInfoLog(fs),
              gl.getProgramInfoLog(program)].join("\\n"),
        max_attributes: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
        attributes: [],
    };
    var nattr = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES) || 0;
    for (var i = 0; i < nattr; i++)
        result.attributes.push(gl.getActiveAttrib(program, i).name);
    return result;
};
</script></body></html>
"""

# The options the viewer generates surface shaders with. ``morphs`` is the
# number of surfaces to mix between (anatomical, inflated and flat), ``volume``
# says the subject has a white matter surface; the rest come from the dataview
# and from the surface menu.
SURFACE_OPTS = dict(morphs=3, volume=1, layers=1, rois=True, extratex=False,
                    halo=False, dither=False, voxline=False, sampler="nearest")


def _surface_variants():
    """Every (shader, opts) pair the viewer can ask for a surface shader."""
    for shader in ("surface_vertex", "surface_pixel"):
        for rgb in (False, True):
            for twod in (False, True):
                if rgb and twod:
                    continue  # RGB data has no second dimension
                for hasflat in (False, True):
                    for equivolume in (False, True):
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


def _variants():
    yield from _surface_variants()
    # The shaders the picker renders with; they morph the same geometry but
    # carry no data.
    yield pytest.param("pick", dict(morphs=3, volume=1), id="pick")
    yield pytest.param("depth", dict(morphs=3, volume=1), id="depth")


@pytest.fixture(scope="module")
def link_shader(tmp_path_factory):
    """Return a function linking one shader variant in a real GL context."""
    from playwright.sync_api import sync_playwright

    page_path = tmp_path_factory.mktemp("shaders") / "shaders.html"
    page_path.write_text(
        PAGE.replace("__JSDIR__", JS_PATH)
        .replace("__VERTEX_PREFIX__", json.dumps(VERTEX_PREFIX))
        .replace("__FRAGMENT_PREFIX__", json.dumps(FRAGMENT_PREFIX))
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--enable-webgl", "--use-gl=swiftshader", "--no-sandbox",
                  "--disable-dev-shm-usage"],
        )
        page = browser.new_page()
        page.goto("file://%s" % page_path, wait_until="load", timeout=60000)
        if not page.evaluate("() => !!window.linkShader"):
            browser.close()
            pytest.skip("no WebGL context available in this browser")
        yield lambda shader, opts: page.evaluate(
            "args => window.linkShader(args[0], args[1])", [shader, opts]
        )
        browser.close()


@pytest.mark.parametrize("shader,opts", list(_variants()))
def test_shader_links(shader, opts, link_shader):
    """Each shader variant has to compile *and* link.

    A variant that uses more vertex attributes than the driver has slots for
    compiles fine and fails to link, which leaves the viewer showing nothing at
    all.
    """
    result = link_shader(shader, opts)
    assert result["compiled"], "%s did not compile:\n%s" % (shader, result["log"])
    assert result["linked"], (
        "%s compiled but did not link, using %d of the %d available vertex "
        "attributes:\n%s" % (shader, len(result["attributes"]),
                             result["max_attributes"], result["log"])
    )
    assert len(result["attributes"]) <= result["max_attributes"]
