var Shaderlib = (function() {

    var utils = {
        rand: [
            "float rand(vec2 co){",
            "    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);",
            "}",
        ].join("\n"),
        edge: [
            // from http://codeflow.org/entries/2012/aug/02/easy-wireframe-display-with-barycentric-coordinates/
            "float edgeFactor(vec3 edge) {",
                "vec3 d = fwidth(edge)+voxlineWidth;",
                "vec3 a3 = smoothstep(vec3(0.), d, edge);",
                "return min(min(a3.x, a3.y), a3.z);",
            "}",
        ].join("\n"),

        colormap: [
            "vec4 colorlut(vec4 values) {",
                "float range = vmax[0] - vmin[0];",
                "float norm0 = (values.x - vmin[0]) / range;",
                "float norm1 = (values.y - vmin[0]) / range;",
                "float fnorm0 = mix(norm0, norm1, framemix);",
            "#ifdef TWOD",
                "range = vmax[1] - vmin[1];",
                "norm0 = (values.z - vmin[1]) / range;",
                "norm1 = (values.w - vmin[1]) / range;",
                "float fnorm1 = mix(norm0, norm1, framemix);",
                "vec2 cuv = vec2(clamp(fnorm0, 0., 1.), clamp(fnorm1, 0., 1.) );",
            "#else",
                "vec2 cuv = vec2(clamp(fnorm0, 0., 1.), 0.);",
            "#endif",

                "vec4 vColor = texture2D(colormap, cuv);",
                "bvec4 valid = notEqual(lessThanEqual(values, vec4(0.)), lessThan(vec4(0.), values));",
                "return all(valid) ? vColor : vec4(0.);",
            "}"
        ].join("\n"),

        pack: [
            "vec4 pack_float( const in float depth ) {",
                "const vec4 bit_shift = vec4( 256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0 );",
                "const vec4 bit_mask  = vec4( 0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0 );",
                "vec4 res = fract( depth * bit_shift );",
                "res -= res.xxyz * bit_mask;",
                "return floor(res * 256.) / 256. + 1./512.;",
            "}",
        ].join("\n"),

        samplers: [
            "vec2 make_uv_x(vec2 coord, float slice) {",
                "vec2 pos = vec2(mod(slice, mosaic[0].x), slice / mosaic[0].x);",
                "vec2 offset = (floor(pos) * (dshape[0]+1.)) + 1.;",
                "vec2 imsize = (mosaic[0] * (dshape[0]+1.)) + 1.;",
                "return (2.*(offset+coord)+1.) / (2.*imsize);",
            "}",

            "vec4 trilinear_x(sampler2D data, vec3 coord) {",
                "vec4 tex1 = texture2D(data, make_uv_x(coord.xy, floor(coord.z)));",
                "vec4 tex2 = texture2D(data, make_uv_x(coord.xy, ceil(coord.z)));",
                'return mix(tex1, tex2, fract(coord.z));',
            "}",

            "vec4 nearest_x(sampler2D data, vec3 coord) {",
                "return texture2D(data, make_uv_x(coord.xy, floor(coord.z+.5)));",
            "}",

            "vec4 debug_x(sampler2D data, vec3 coord) {",
                "return vec4(coord / vec3(136., 136., 38.), 1.);",
            "}",

            "vec2 make_uv_y(vec2 coord, float slice) {",
                "vec2 pos = vec2(mod(slice, mosaic[1].x), floor(slice / mosaic[1].x));",
                "vec2 offset = (pos * (dshape[1]+1.)) + 1.;",
                "vec2 imsize = (mosaic[1] * (dshape[1]+1.)) + 1.;",
                "return (2.*(offset+coord)+1.) / (2.*imsize);",
            "}",

            "vec4 trilinear_y(sampler2D data, vec3 coord) {",
                "vec4 tex1 = texture2D(data, make_uv_y(coord.xy, floor(coord.z)));",
                "vec4 tex2 = texture2D(data, make_uv_y(coord.xy, ceil(coord.z)));",
                'return mix(tex1, tex2, fract(coord.z));',
            "}",

            "vec4 nearest_y(sampler2D data, vec3 coord) {",
                "return texture2D(data, make_uv_y(coord.xy, floor(coord.z+.5)));",
            "}",

            "vec4 debug_y(sampler2D data, vec3 coord) {",
                "return vec4(coord / vec3(100., 100., 32.), 1.);",
            "}",
        ].join("\n"),

        standard_frag_vars: [
            "uniform vec3 diffuse;",
            "uniform vec3 ambient;",
            "uniform vec3 emissive;",
            "uniform vec3 specular;",
            "uniform float shininess;",

            "uniform sampler2D colormap;",
            "uniform float vmin[2];",
            "uniform float vmax[2];",
            "uniform float framemix;",

            "uniform vec3 voxlineColor;",
            "uniform float voxlineWidth;",
            "uniform float dataAlpha;",

            "uniform vec2 mosaic[2];",
            "uniform vec2 dshape[2];",
            "uniform sampler2D data[4];",
        ].join("\n"),

        mixer: function(morphs) {
            var glsl = [
            "uniform float mix;",
            "attribute vec3 mixSurfs["+morphs+"];",
            "attribute vec3 mixNorms["+morphs+"];",
            "vec3 mixfunc(vec3 basepos) {",
                "float smix = mix * "+morphs";",
                "float clamp(1.-smix, 0., 1.);",
            "}",
            ].join("\n");
            return glsl;
        }
    }

    var module = function() {

    };
    module.prototype = {
        constructor: module,
        main: function(sampler, raw, twod, voxline) {
            //Creates shader code with all the parameters
            //sampler: which sampler to use, IE nearest or trilinear
            //raw: whether the dataset is raw or not
            //voxline: whether to show the voxel lines
            
            var header = "";
            if (voxline)
                header += "#define VOXLINE\n";
            if (raw)
                header += "#define RAWCOLORS\n";
            if (twod)
                header += "#define TWOD\n";

            var vertShade =  [
            THREE.ShaderChunk[ "lights_phong_pars_vertex" ],

            "uniform mat4 volxfm[2];",

            "attribute vec4 auxdat;",

            "varying vec3 vViewPosition;",
            "varying vec3 vNormal;",

            "varying vec3 vPos_x;",

        "#ifdef TWOD",
            "varying vec3 vPos_y;",
        "#endif",

            "void main() {",

                "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",
                "vViewPosition = -mvPosition.xyz;",

                //Find voxel positions with both transforms (2D colormap x and y datasets)
                "vPos_x = (volxfm[0]*vec4(position,1.)).xyz;",
        "#ifdef TWOD",
                "vPos_y = (volxfm[1]*vec4(position,1.)).xyz;",
        "#endif",

                "vNormal = normalMatrix * normal;",
                "gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0);",

            "}"
            ].join("\n");

            var fragShade = [
            "#extension GL_OES_standard_derivatives: enable",
            "#extension GL_OES_texture_float: enable",

            "varying vec3 vPos_x;",
            "varying vec3 vPos_y;",

            THREE.ShaderChunk[ "lights_phong_pars_fragment" ],
            
            utils.standard_frag_vars,
            utils.rand,
            utils.edge,
            utils.colormap,
            utils.samplers,

            "void main() {",
            "#ifdef RAWCOLORS",
                "vec4 color[2]; color[0] = vec4(0.), color[1] = vec4(0.);",
            "#else",
                "vec4 values = vec4(0.);",
            "#endif",
        
        "#ifdef RAWCOLORS",
                "color[0] += "+sampler+"_x(data[0], vPos_x);",
                "color[1] += "+sampler+"_x(data[1], vPos_x);",
        "#else",
                "values.x += "+sampler+"_x(data[0], vPos_x).r;",
                "values.y += "+sampler+"_x(data[1], vPos_x).r;",
            "#ifdef TWOD",
                "values.z += "+sampler+"_y(data[2], vPos_y).r;",
                "values.w += "+sampler+"_y(data[3], vPos_y).r;",
            "#endif",
        "#endif",

            "#ifdef RAWCOLORS",
                "vec4 vColor = mix(color[0], color[1], framemix);",
            "#else",
                "vec4 vColor = colorlut(values);",
            "#endif",
                "vColor *= dataAlpha;",

        "#ifdef VOXLINE",
                "vec3 coord = vPos_x[0];",
                "vec3 edge = abs(fract(coord) - vec3(0.5));",
                "vColor = mix(vec4(voxlineColor, 1.), vColor, edgeFactor(edge*1.001));",
        "#endif",

                "if (vColor.a < .001) discard;",
                "gl_FragColor = vColor;",

                THREE.ShaderChunk[ "lights_phong_fragment" ],
            "}"
            ].join("\n");

            return {vertex:header+vertShade, fragment:header+fragShade};
        },
        surface: function(sampler, raw, twod, voxline, morphs, volume, rois) {
            if (volume === undefined)
                volume = 0;

            var header = "";
            if (voxline)
                header += "#define VOXLINE\n";
            if (raw)
                header += "#define RAWCOLORS\n";
            if (volume > 0)
                header += "#define CORTSHEET\n";
            if (twod)
                header += "#define TWOD\n";
            if (rois)
                header += "#define ROI_RENDER\n";


            var vertShade =  [
            THREE.ShaderChunk[ "lights_phong_pars_vertex" ],
            "uniform mat4 volxfm[2];",

            "uniform float thickmix;",
            "attribute vec3 position2;",

            "attribute vec4 auxdat;",
            // "attribute vec3 wmnorm;",
            // "attribute float dropout;",
            
            "varying vec2 vUv;",
            "varying float vCurv;",
            // "varying float vDrop;",
            "varying float vMedial;",

            "varying vec3 vViewPosition;",
            "varying vec3 vNormal;",

            "varying vec3 vPos_x[2];",
            "varying vec3 vPos_y[2];",

            utils.mixer(morphs),

            "void main() {",

                "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",
                "vViewPosition = -mvPosition.xyz;",

                //Find voxel positions with both transforms (2D colormap x and y datasets)
                "vPos_x[0] = (volxfm[0]*vec4(position,1.)).xyz;",
            "#ifdef TWOD",
                "vPos_y[0] = (volxfm[1]*vec4(position,1.)).xyz;",
            "#endif",
        "#ifdef CORTSHEET",
                "vPos_x[1] = (volxfm[0]*vec4(position2,1.)).xyz;",
            "#ifdef TWOD",
                "vPos_y[1] = (volxfm[1]*vec4(position2,1.)).xyz;",
            "#endif",
                "vec3 npos = mix(position, position2, thickmix);",
        "#else",
                "vec3 npos = position;",
        "#endif",

            "#ifdef ROI_RENDER",
                //Overlay
                "vUv = uv;",
            "#endif",

                // "vDrop = dropout;",
                "vMedial = auxdat.x;",
                "vCurv = auxdat.y;",



                "gl_Position = projectionMatrix * modelViewMatrix * vec4( morphed, 1.0 );",

            "}"
            ].join("\n");

            var fragHead = [
            "#extension GL_OES_standard_derivatives: enable",
            "#extension GL_OES_texture_float: enable",

            THREE.ShaderChunk[ "map_pars_fragment" ],
            THREE.ShaderChunk[ "lights_phong_pars_fragment" ],

            utils.standard_frag_vars,

            "uniform int hide_mwall;",
            "uniform float curvAlpha;",
            "uniform float curvScale;",
            "uniform float curvLim;",
            "uniform float hatchAlpha;",
            "uniform vec3 hatchColor;",

            "uniform sampler2D hatch;",
            "uniform vec2 hatchrep;",

            "varying float vCurv;",
            // "varying float vDrop;",
            "varying float vMedial;",

            "uniform float thickmix;",

            "varying vec3 vPos_x[2];",
            "varying vec3 vPos_y[2];",

            utils.rand,
            utils.edge,
            utils.colormap,
            utils.samplers,

            "void main() {",
                //Curvature Underlay
                "float curv = clamp(vCurv / curvScale  + .5, curvLim, 1.-curvLim);",
                "vec4 cColor = vec4(vec3(curv) * curvAlpha, curvAlpha);",

                "vec3 coord_x, coord_y;",
            "#ifdef RAWCOLORS",
                "vec4 color[2]; color[0] = vec4(0.), color[1] = vec4(0.);",
            "#else",
                "vec4 values = vec4(0.);",
            "#endif",
                "",
            ].join("\n");

            
            var factor = volume > 1 ? (1/volume).toFixed(6) : "1.";
            var sampling = [
        "#ifdef RAWCOLORS",
                "color[0] += "+factor+"*"+sampler+"_x(data[0], coord_x);",
                "color[1] += "+factor+"*"+sampler+"_x(data[1], coord_x);",
        "#else",
                "values.x += "+factor+"*"+sampler+"_x(data[0], coord_x).r;",
                "values.y += "+factor+"*"+sampler+"_x(data[1], coord_x).r;",
            "#ifdef TWOD",
                "values.z += "+factor+"*"+sampler+"_y(data[2], coord_y).r;",
                "values.w += "+factor+"*"+sampler+"_y(data[3], coord_y).r;",
            "#endif",
        "#endif",
            ].join("\n");

            var fragMid = "";
            if (volume == 0) {
                fragMid += [
                    "coord_x = vPos_x[0];",
                "#ifdef TWOD",
                    "coord_y = vPos_y[0];",
                "#endif",
                    sampling,
                    ""
                ].join("\n");
            } else if (volume == 1) {
                fragMid += [
                    "coord_x = mix(vPos_x[0], vPos_x[1], thickmix);",
                "#ifdef TWOD",
                    "coord_y = mix(vPos_y[0], vPos_y[1], thickmix);",
                "#endif",
                    sampling,
                    "",
                ].join("\n");
            } else {
                fragMid += "vec2 rseed;\nfloat randval;\n";
                for (var i = 0; i < volume; i++) {
                    fragMid += [
                        "rseed = gl_FragCoord.xy + vec2(2.34*"+i.toFixed(3)+", 3.14*"+i.toFixed(3)+");",
                        "randval = rand(rseed);",
                        "coord_x = mix(vPos_x[0], vPos_x[1], randval);",
                    "#ifdef TWOD",
                        "coord_y = mix(vPos_y[0], vPos_y[1], randval);",
                    "#endif",
                        sampling,
                        "", 
                    ].join("\n");
                }
            }

            var fragTail = [
            "#ifdef RAWCOLORS",
                "vec4 vColor = mix(color[0], color[1], framemix);",
            "#else",
                "vec4 vColor = colorlut(values);",
            "#endif",
                "vColor *= dataAlpha;",

        "#ifdef VOXLINE",
            "#ifdef CORTSHEET",
                "vec3 coord = mix(vPos_x[0], vPos_x[1], 0.5);",
            "#else",
                "vec3 coord = vPos_x[0];",
            "#endif",
                "vec3 edge = abs(fract(coord) - vec3(0.5));",
                "vColor = mix(vec4(voxlineColor, 1.), vColor, edgeFactor(edge*1.001));",
        "#endif",

                //Cross hatch / dropout layer
                // "float hw = gl_FrontFacing ? hatchAlpha*vDrop : 1.;",
                // "vec4 hColor = hw * vec4(hatchColor, 1.) * texture2D(hatch, vUv*hatchrep);",

                //roi layer
            "#ifdef ROI_RENDER",
                "vec4 rColor = texture2D(map, vUv);",
            "#endif",          

        "#ifdef SUBJ_SURF",
                "if (vMedial < .999) {",
                    "gl_FragColor = cColor;",
                    "gl_FragColor = vColor + (1.-vColor.a)*gl_FragColor;",
                    // "gl_FragColor = hColor + (1.-hColor.a)*gl_FragColor;",
            "#ifdef ROI_RENDER",
                    "gl_FragColor = rColor + (1.-rColor.a)*gl_FragColor;",
            "#endif",
                "} else if (hide_mwall == 1) {",
                    "discard;",
                "} else {",
                    "gl_FragColor = cColor;",
                "}",
        "#else",
                "if (vColor.a < .01) discard;",
                "gl_FragColor = vColor;",
        "#endif",
            "} else if (hide_mwall == 1) {",
                "discard;",
            "} else {",
                "gl_FragColor = cColor;",
            "}",

                THREE.ShaderChunk[ "lights_phong_fragment" ],
            "}"
            ].join("\n");


            return {vertex:header+vertShade, fragment:header+fragHead+fragMid+fragTail};
        },
        pick: function() {
            var vertShade = [
                "attribute vec4 auxdat;",
                "varying vec3 vPos;",
                "varying float vMedial;",
                THREE.ShaderChunk[ "morphtarget_pars_vertex" ],
                "void main() {",
                    "vPos = position;",
                    "vMedial = auxdat.x;",
                    THREE.ShaderChunk[ "morphtarget_vertex" ],
                "}",
            ].join("\n");

            var fragShades = [];
            var dims = ['x', 'y', 'z'];
            for (var i = 0; i < 3; i++){
                var dim = dims[i];
                var shade = [
                "uniform vec3 min;",
                "uniform vec3 max;",
                "uniform int hide_mwall;",
                "varying vec3 vPos;",
                "varying float vMedial;",
                utils.pack,
                "void main() {",
                    "float norm = (vPos."+dim+" - min."+dim+") / (max."+dim+" - min."+dim+");", 
                    "if (vMedial > .999 && hide_mwall == 1)",
                        "discard;",
                    "else",
                        "gl_FragColor = pack_float(norm);",
                "}"
                ].join("\n");
                fragShades.push(shade);
            }

            return {vertex:vertShade, fragment:fragShades};
        },
        data: function(thick) {
            var vertex = [
                "uniform mat4 volxfm;",

                "attribute vec3 wm;",
                "attribute vec3 flatpos;",
                "attribute vec4 auxdat;",

                "varying vec3 vPos[2];",
                "varying float vMedial;",

                "void main() {",
                    "vMedial = auxdat.x;",
                    "vPos[0] = (volxfm*vec4(position, 1.)).xyz;",
                    "#ifdef CORTSHEET",
                    "vPos[1] = (volxfm*vec4(wm, 1.)).xyz;",
                    "#endif",
                    "gl_Position = projectionMatrix * modelViewMatrix * vec4( flatpos, 1.0 );",
                "}",
            ].join("\n");
            var fragment = [
                "varying vec3 vPos[2];",
                "varying float vMedial;",

                "uniform vec2 mosaic[2];",
                "uniform vec2 dshape[2];",
                "uniform sampler2D data;",

                utils.samplers,

                "void main() {",
                    "gl_FragColor = vec4(vPos[0] / vec3(100., 100., 32.), 1.);",
                    "if (vMedial > .999) discard;",
                    // "gl_FragColor = vec4(vec3(.5), 1.);",
                "}",
            ].join("\n");
            return {vertex:vertex, fragment:fragment};
        }
    };

    return module;
})();

var Shaders = new Shaderlib();