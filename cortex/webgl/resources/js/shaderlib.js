var Shaderlib = (function() {
    var samplers = (function() {
        var shader = "";
        var dims = ["x", "y"];
        for (var val = 0; val < 2; val++) {
            var dim = dims[val];
            shader += [
            "vec2 make_uv_"+dim+"(vec2 coord, float slice) {",
                "vec2 pos = vec2(mod(slice, mosaic["+val+"].x), floor(slice / mosaic["+val+"].x));",
                "vec2 offset = (pos * (dshape["+val+"]+1.)) + 1.;",
                "vec2 imsize = (mosaic["+val+"] * (dshape["+val+"]+1.)) + 1.;",
                "return (2.*(offset+coord)+1.) / (2.*imsize);",
            "}",

            "vec4 trilinear_"+dim+"(sampler2D data, vec3 coord) {",
                "vec4 tex1 = texture2D(data, make_uv_"+dim+"(coord.xy, floor(coord.z)));",
                "vec4 tex2 = texture2D(data, make_uv_"+dim+"(coord.xy, ceil(coord.z)));",
                'return mix(tex1, tex2, fract(coord.z));',
            "}",

            "vec4 nearest_"+dim+"(sampler2D data, vec3 coord) {",
                "vec4 c = texture2D(data, make_uv_"+dim+"(coord.xy, ceil(coord.z)));",
                "vec4 f = texture2D(data, make_uv_"+dim+"(coord.xy, floor(coord.z)));",
                // "return fract(coord.z) < .5 ? f : c;",
                "return mix(f, c, step(0.5, fract(coord.z)));",
            "}\n",

            "vec4 debug_"+dim+"(sampler2D data, vec3 coord) {",
                "return vec4(coord / vec3(100., 100., 32.), 1.);",
                // "return vec4(coord ,1.);",
            "}",


            ].join("\n");
        }
        return shader;
    })();

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
                "float norm1 = (values.z - vmin[0]) / range;",
                "float fnorm0 = mix(norm0, norm1, framemix);",

                "range = vmax[1] - vmin[1];",
                "norm0 = (values.y - vmin[1]) / range;",
                "norm1 = (values.w - vmin[1]) / range;",
                "float fnorm1 = mix(norm0, norm1, framemix);",
                "vec2 cuv = vec2(clamp(fnorm0, 0., .999), clamp(fnorm1, 0., .999) );",

                "vec4 vColor = texture2D(colormap, cuv);",
                "bvec4 valid = notEqual(lessThanEqual(values, vec4(0.)), lessThan(vec4(0.), values));",
                "vColor = all(valid) ? vColor : vec4(0.);",
                "return vColor;",
            "}"
        ].join("\n"),

    }

    var module = function() {

    };
    module.prototype = {
        constructor: module,
        main: function(sampler, raw, voxline, volume) {
            //Creates shader code with all the parameters
            //sampler: which sampler to use, IE nearest or trilinear
            //raw: whether the dataset is raw or not
            //voxline: whether to show the voxel lines
            //volume: the number of volume samples to take

            if (volume === undefined)
                volume = 0;

            var header = "";
            if (voxline)
                header += "#define VOXLINE\n";
            if (raw)
                header += "#define RAWCOLORS\n";
            if (volume > 0)
                header += "#define CORTSHEET\n";

            var vertShade =  [
            THREE.ShaderChunk[ "map_pars_vertex" ], 
            THREE.ShaderChunk[ "lights_phong_pars_vertex" ],
            THREE.ShaderChunk[ "morphtarget_pars_vertex" ],

            "uniform float thickmix;",
            "uniform mat4 volxfm[2];",

            "attribute vec4 auxdat;",
            "attribute vec3 wm;",
            // "attribute vec3 wmnorm;",
            // "attribute float dropout;",

            "varying vec3 vViewPosition;",
            "varying vec3 vNormal;",
            "varying float vCurv;",
            // "varying float vDrop;",
            "varying float vMedial;",
            "varying vec3 vPos_x[2];",
            "varying vec3 vPos_y[2];",

            "void main() {",

                "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",
                "vViewPosition = -mvPosition.xyz;",

                THREE.ShaderChunk[ "map_vertex" ],

                // "vDrop = dropout;",
                "vCurv = auxdat.y;",
                "vMedial = auxdat.z;",

                //Find voxel positions with both transforms (2D colormap x and y datasets)
                "vPos_x[0] = (volxfm[0]*vec4(position,1.)).xyz;",
                "vPos_y[0] = (volxfm[1]*vec4(position,1.)).xyz;",
                "#ifdef CORTSHEET",
                "vPos_x[1] = (volxfm[0]*vec4(wm,1.)).xyz;",
                "vPos_y[1] = (volxfm[1]*vec4(wm,1.)).xyz;",
                "vec3 npos = mix(position, wm, thickmix);",
                "#else",
                "vec3 npos = position;",
                "#endif",

                THREE.ShaderChunk[ "morphnormal_vertex" ],
                "vNormal = transformedNormal;",

                "vec3 morphed = vec3( 0.0 );",
                "morphed += ( morphTarget0 - npos ) * morphTargetInfluences[ 0 ];",
                "morphed += ( morphTarget1 - npos ) * morphTargetInfluences[ 1 ];",
                "morphed += ( morphTarget2 - npos ) * morphTargetInfluences[ 2 ];",
                "morphed += ( morphTarget3 - npos ) * morphTargetInfluences[ 3 ];",
                "morphed += npos;",
                "gl_Position = projectionMatrix * modelViewMatrix * vec4( morphed, 1.0 );",

            "}"
            ].join("\n");

            var factor = volume > 1 ? (1/volume).toFixed(6) : "1.";

            var fragHead = [
            "#extension GL_OES_standard_derivatives: enable",
            "#extension GL_OES_texture_float: enable",
            THREE.ShaderChunk[ "map_pars_fragment" ],
            THREE.ShaderChunk[ "lights_phong_pars_fragment" ],

            "uniform int hide_mwall;",

            "uniform vec3 diffuse;",
            "uniform vec3 ambient;",
            "uniform vec3 emissive;",
            "uniform vec3 specular;",
            "uniform float shininess;",

            "uniform sampler2D colormap;",
            "uniform float vmin[2];",
            "uniform float vmax[2];",
            "uniform float framemix;",

            "uniform float curvAlpha;",
            "uniform float curvScale;",
            "uniform float curvLim;",
            "uniform float dataAlpha;",
            "uniform float hatchAlpha;",
            "uniform vec3 hatchColor;",
            "uniform vec3 voxlineColor;",
            "uniform float voxlineWidth;",

            "uniform sampler2D hatch;",
            "uniform vec2 hatchrep;",

            "uniform vec2 mosaic[2];",
            "uniform vec2 dshape[2];",
            "uniform sampler2D data[4];",
            "uniform float thickmix;",

            "varying float vCurv;",
            // "varying float vDrop;",
            "varying float vMedial;",
            "varying vec3 vPos_x[2];",
            "varying vec3 vPos_y[2];",

            utils.rand,
            utils.edge,
            utils.colormap,
            samplers,

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

            
            var sampling = [
        "#ifdef RAWCOLORS",
                "color[0] += "+factor+"*"+sampler+"_x(data[0], coord_x);",
                "color[1] += "+factor+"*"+sampler+"_x(data[2], coord_x);",
        "#else",
                "values.x += "+factor+"*"+sampler+"_x(data[0], coord_x).r;",
                "values.z += "+factor+"*"+sampler+"_x(data[2], coord_x).r;",
                "values.y += "+factor+"*"+sampler+"_y(data[1], coord_y).r;",
                "values.w += "+factor+"*"+sampler+"_y(data[3], coord_y).r;",
        "#endif",
            ].join("\n");

            var fragMid = "";
            if (volume == 0) {
                fragMid += [
                    "coord_x = vPos_x[0];",
                    "coord_y = vPos_y[0];",
                    sampling,
                    ""
                ].join("\n");
            } else if (volume == 1) {
                fragMid += [
                    "coord_x = mix(vPos_x[0], vPos_x[1], thickmix);",
                    "coord_y = mix(vPos_y[0], vPos_y[1], thickmix);",
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
                        "coord_y = mix(vPos_y[0], vPos_y[1], randval);",
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
                "vec4 rColor = texture2D(map, vUv);",

                "if (vMedial < .999) {",
                    "gl_FragColor = cColor;",
                    "gl_FragColor = vColor + (1.-vColor.a)*gl_FragColor;",
                    // "gl_FragColor = hColor + (1.-hColor.a)*gl_FragColor;",
                    "gl_FragColor = rColor + (1.-rColor.a)*gl_FragColor;",
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
                    "vMedial = auxdat.z;",
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
                "vec4 pack_float( const in float depth ) {",
                    "const vec4 bit_shift = vec4( 256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0 );",
                    "const vec4 bit_mask  = vec4( 0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0 );",
                    "vec4 res = fract( depth * bit_shift );",
                    "res -= res.xxyz * bit_mask;",
                    "return res;",
                "}",
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
        data: function() {
            var vertex = [
                "attribute vec3 position;",
                "attribute vec3 wm;",
                "attribute vec3 uv;",
                "varying vPos[2];",

                "void main() {",
                    "vPos[0] = position;",
                    "#ifdef CORTSHEET",
                    "vPos[1] = wm;",
                    "#endif",
                    "gl_Position = projectionMatrix * modelViewMatrix * vec4( uv, 1.0 );",
                "}",
            ].join("\n");
            var fragment = [
                "uniform mat4 volxfm[2];",
                "uniform vec2 mosaic[2];",
                "uniform vec2 shape[2];",
                "uniform sampler2D data[4];",
                samplers,
                "void main() {",
                    "gl_FragColor = vec4(vec3(.5), 1.);",
                "}",
            ].join("\n");
            return {vertex:vertex, fragment:fragment};
        }
    };

    return module;
})();

var Shaders = new Shaderlib();